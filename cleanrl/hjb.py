# HJB Value Iteration with Learned Control-Affine Dynamics (Neural ODE)
# and Quadratic Reward Approximation for Analytical Action
# Uses Normalization, Raw Buffer Storage, Model Validation, Early Stopping.
# --- Fix for incomplete eval_calculate_a_star_quad_approx ---

import os
import random
import time
import math
import copy
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Any, Union, Sequence, Callable, Generator


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
# Removed Normal distribution import
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# Required imports
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecEnv, VecNormalize, DummyVecEnv
try:
    from torch.func import grad, vmap, jacrev, hessian # <<< Need hessian >>>
    print("Imported grad, vmap, jacrev, hessian from torch.func")
    TORCH_FUNC_AVAILABLE = True
except ImportError:
    try:
        from functorch import grad, vmap, jacrev, hessian
        print("Imported grad, vmap, jacrev, hessian from functorch")
        TORCH_FUNC_AVAILABLE = True
    except ImportError:
        print("WARNING: torch.func / functorch required for HJB gradients/jacobians/hessians not available.")
        TORCH_FUNC_AVAILABLE = False
try:
    import torchode as to
    print("Imported torchode.")
    TORCHODE_AVAILABLE = True
except ImportError:
    print("FATAL: torchode not found (`pip install torchode`).")
    TORCHODE_AVAILABLE = False
    exit()

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.buffers")


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")] + "_hjb_vi_ode_quad"
    """the name of this experiment"""
    seed: int = 1; torch_deterministic: bool = True; cuda: bool = True
    track: bool = False; wandb_project_name: str = "cleanRL"; wandb_entity: str = None
    capture_video: bool = False; save_model: bool = True; upload_model: bool = False; hf_entity: str = ""

    # Algorithm specific arguments
    env_id: str = "InvertedPendulum-v4"; total_timesteps: int = 1000000 # Simpler env better for this complex method
    learning_rate: float = 1e-3 # LR for Critic (V function)
    buffer_size: int = int(1e6); gamma: float = 0.99
    batch_size: int = 256 # Agent update batch size
    learning_starts: int = 5000
    exploration_noise_std: float = 0.1
    grad_norm_clip: Optional[float] = 1.0

    # Model Training Args
    model_train_freq: int = 250
    model_dataset_size: int = 50_000
    dynamics_learning_rate: float = 1e-3
    reward_learning_rate: float = 1e-3 # <<< Re-added >>>
    dynamic_train_threshold: float = 0.01
    reward_train_threshold: float = 0.01 # <<< Re-added >>>
    model_val_ratio: float = 0.2; model_val_patience: int = 10
    model_val_delta: float = 1e-5; model_max_epochs: int = 50
    model_train_batch_size: int = 256; model_validation_freq: int = 5
    model_updates_per_epoch: int = 1

    # Model Rollout Args (Currently unused)
    model_rollout_freq: int = 10000000; model_rollout_length: int = 1; num_model_rollout_starts: int = 0

    # HJB Residual Args
    hjb_coef: float = 1.0
    use_hjb_loss: bool = True
    terminal_coeff: float = 1.0
    hessian_reg: float = 1e-3 # Regularization for Hessian inversion
    """Regularization added to the reward Hessian before inversion."""

    # Env Args
    num_envs: int = 1; env_dt: float = 0.02

    # Runtime filled
    minibatch_size: int = field(init=False); rho: float = field(init=False)


# --- Environment Creation ---
def make_env(env_id, seed, idx, capture_video, run_name): # (Same)
    def thunk():
        render_mode="rgb_array" if capture_video and idx==0 else None;
        try: env=gym.make(env_id,render_mode=render_mode)
        except Exception as e: print(f"Warning: Failed render_mode='{render_mode}'. Error: {e}. Defaulting."); env=gym.make(env_id)
        if capture_video and idx==0 and env.render_mode=="rgb_array": env=gym.wrappers.RecordVideo(env,f"videos/{run_name}",episode_trigger=lambda x:x%50==0)
        if isinstance(env.observation_space, gym.spaces.Dict): env=gym.wrappers.FlattenObservation(env)
        env=gym.wrappers.RecordEpisodeStatistics(env); env=gym.wrappers.ClipAction(env);
        env.action_space.seed(seed + idx); return env
    return thunk

# --- Utilities ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0): # (Same)
    torch.nn.init.orthogonal_(layer.weight, std); torch.nn.init.constant_(layer.bias, bias_const); return layer

# --- Dynamics Model (Neural ODE using TorchODE) ---
class ODEFunc(nn.Module): # (Same)
    def __init__(self, obs_dim, action_dim): super().__init__(); hidden_size=256; self.net=nn.Sequential(layer_init(nn.Linear(obs_dim+action_dim,hidden_size)), nn.SiLU(), layer_init(nn.Linear(hidden_size,hidden_size)), nn.SiLU(), layer_init(nn.Linear(hidden_size,hidden_size)), nn.SiLU(), layer_init(nn.Linear(hidden_size,obs_dim)),); print(f"ODEFunc: In={obs_dim+action_dim}, Out={obs_dim}")
    def forward(self, t, x_norm, a): return self.net(torch.cat([x_norm.float(), a.float()], dim=-1))
class DynamicModel(nn.Module): # (Same)
    def __init__(self, obs_dim, action_dim, dt: float, device: torch.device):
        super().__init__()
        self.ode_func = ODEFunc(obs_dim, action_dim)
        self.dt = dt
        self.device = device
        if not TORCHODE_AVAILABLE:
            raise ImportError("torchode not found.")
        self.term = to.ODETerm(self.ode_func, with_args=True)
        self.step_method = to.Euler(term=self.term)
        self.step_size_controller = to.FixedStepController()
        self.adjoint = to.AutoDiffAdjoint(step_method=self.step_method, step_size_controller=self.step_size_controller)
        print(f"DynamicModel: TorchODE (Solver: Euler, dt={self.dt})")
    def forward(self, initial_obs_norm, actions_norm): batch_size=initial_obs_norm.shape[0]; dt0=torch.full((batch_size,),self.dt/5,device=self.device); t_span_tensor=torch.tensor([0.0,self.dt],device=self.device); t_eval=t_span_tensor.unsqueeze(0).repeat(batch_size,1); problem=to.InitialValueProblem(y0=initial_obs_norm.float(),t_eval=t_eval,); t_eval_actual, sol_ys = to.odeint(self.ode_func, initial_obs_norm.float(), t_eval[0], solver=self.step_method, args=(actions_norm.float(),), dt0=dt0[0]); final_state_pred_norm=sol_ys[1]; return final_state_pred_norm

# --- Reward Model ---
class RewardModel(nn.Module): # Re-introduced
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        hidden_size=128
        self.net=nn.Sequential(
            layer_init(nn.Linear(obs_dim+action_dim,hidden_size)), nn.ReLU(),
            layer_init(nn.Linear(hidden_size,hidden_size)), nn.ReLU(),
            layer_init(nn.Linear(hidden_size,1))
        )
        print(f"Initialized RewardModel: Input {obs_dim+action_dim}, Output 1")
    def forward(self, obs_norm, action):
        # Predicts normalized reward (scalar output)
        return self.net(torch.cat([obs_norm.float(), action.float()], dim=1)).squeeze(-1) # Output shape [batch]

# --- Agent Network Definitions ---
class ValueNetwork(nn.Module): # Renamed from HJBCritic
    def __init__(self, env: VecEnv): super().__init__(); obs_dim=np.array(env.observation_space.shape).prod(); self.net=nn.Sequential(nn.Linear(obs_dim,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,1),); print("Initialized ValueNetwork (Critic).")
    def forward(self, x_norm): return self.net(x_norm).squeeze(-1)

# --- Utility Functions ---
def calculate_metrics(preds, targets): # (Same)
    mse=F.mse_loss(preds,targets).item(); mae=F.l1_loss(preds,targets).item(); ss_res=torch.sum((targets-preds)**2); ss_tot=torch.sum((targets-torch.mean(targets))**2); r2=(1-ss_res/ss_tot).item() if ss_tot>1e-8 else -float('inf'); return{"mse":mse,"mae":mae,"r2":r2}

def train_model_epoch(model, optimizer, train_loader, device, writer, epoch, global_step, model_name, is_dynamic_model): # Updated
    model.train(); total_loss=0.0; num_batches=0;
    for batch_idx,batch_data in enumerate(train_loader):
        obs_norm, actions, targets_norm = [d.to(device) for d in batch_data];
        preds_norm = model(obs_norm, actions) # Both models take obs, action
        loss = F.mse_loss(preds_norm, targets_norm); optimizer.zero_grad(); loss.backward(); optimizer.step(); total_loss+=loss.item(); num_batches+=1;
        if batch_idx%50==0: writer.add_scalar(f"losses/{model_name}_batch_mse",loss.item(),global_step+epoch*len(train_loader)+batch_idx)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0; return avg_loss

def validate_model(model, val_loader, device, writer, global_step, model_name, is_dynamic_model): # Updated
    model.eval(); all_preds_norm=[]; all_targets_norm=[];
    with torch.no_grad():
        for batch_data in val_loader:
            obs_norm, actions, targets_norm = [d.to(device) for d in batch_data];
            preds_norm = model(obs_norm, actions)
            all_preds_norm.append(preds_norm); all_targets_norm.append(targets_norm)
    if not all_preds_norm: return float('inf'), {"mse":float('inf'),"mae":float('inf'),"r2":-float('inf')}
    all_preds_norm=torch.cat(all_preds_norm,dim=0); all_targets_norm=torch.cat(all_targets_norm,dim=0); val_metrics=calculate_metrics(all_preds_norm,all_targets_norm); val_loss=val_metrics["mse"];
    writer.add_scalar(f"losses/{model_name}_val_mse",val_metrics["mse"],global_step); writer.add_scalar(f"metrics/{model_name}_val_mae",val_metrics["mae"],global_step); writer.add_scalar(f"metrics/{model_name}_val_r2",val_metrics["r2"],global_step)
    return val_loss, val_metrics

# --- Main Execution ---
if __name__ == "__main__":
    if not TORCHODE_AVAILABLE: exit()
    if not TORCH_FUNC_AVAILABLE: print("FATAL: torch.func or functorch required for HJB gradients."); exit()
    args = tyro.cli(Args)

    # Calculate dependent args
    if args.model_rollout_length > 0: args.num_model_rollout_starts = args.num_model_rollout_starts
    else: args.num_model_rollout_starts = 0; print("Warning: model_rollout_length <= 0.")
    args.rho = -math.log(args.gamma) if args.gamma > 0 and args.gamma < 1 else 0.0
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # --- Logging Setup ---
    if args.track: try: import wandb; wandb.init(project=args.wandb_project_name,entity=args.wandb_entity,sync_tensorboard=True,config=vars(args),name=run_name,monitor_gym=True,save_code=True,); print("WandB enabled.") except ImportError: print("WARNING: wandb not installed.");args.track=False
    writer = SummaryWriter(f"runs/{run_name}"); writer.add_text("hyperparameters","|param|value|\n|-|-|\n%s"%("\n".join([f"|{key}|{value}|"for key,value in vars(args).items()])))
    print(f"Run name: {run_name}"); print(f"Arguments: {vars(args)}")

    # --- Seeding & Device ---
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.backends.cudnn.deterministic=args.torch_deterministic;
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu"); print(f"Using device: {device}")

    # --- Environment Setup with VecNormalize ---
    print("Setting up environment...");
    envs = DummyVecEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    norm_envs = VecNormalize(envs, gamma=args.gamma, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0); print("VecNormalize enabled (Obs=True, Reward=True).") # Keep reward norm
    try: env_dt = norm_envs.get_attr("dt")[0]; print(f"Detected env dt: {env_dt}")
    except Exception: print(f"Warning: Could not detect env dt. Using default: {args.env_dt}"); env_dt=args.env_dt
    args.env_dt = env_dt
    # ctrl_cost_weight removed from args, not needed for this version
    obs_space = norm_envs.observation_space; action_space = norm_envs.action_space
    obs_dim = np.array(obs_space.shape).prod(); action_dim = np.prod(action_space.shape)
    action_space_low_t = torch.tensor(action_space.low, dtype=torch.float32, device=device)
    action_space_high_t = torch.tensor(action_space.high, dtype=torch.float32, device=device)

    # --- Agent, Models, Optimizers ---
    critic = ValueNetwork(norm_envs).to(device);
    critic_optimizer = optim.AdamW(critic.parameters(), lr=args.learning_rate);
    dynamic_model = DynamicModel(obs_dim, action_dim, args.env_dt, device).to(device);
    dynamics_optimizer = optim.AdamW(dynamic_model.ode_func.parameters(), lr=args.dynamics_learning_rate);
    reward_model = RewardModel(obs_dim, action_dim).to(device); # Re-added
    reward_optimizer = optim.AdamW(reward_model.parameters(), lr=args.reward_learning_rate); # Re-added

    # --- Replay Buffer for RAW Data ---
    raw_obs_space = norm_envs.unwrapped.observation_space; raw_action_space = norm_envs.unwrapped.action_space; raw_obs_space.dtype = np.float32; print(f"Replay buffer storing RAW data. Obs Shape: {raw_obs_space.shape}"); sb3_buffer_device="cpu";
    rb = ReplayBuffer(args.buffer_size, raw_obs_space, raw_action_space, device=sb3_buffer_device, n_envs=args.num_envs, handle_timeout_termination=True,)
    print("Replay buffer configured with handle_timeout_termination=True")

    # --- Continuous Discount Rate ---
    rho = -torch.log(torch.tensor(args.gamma, device=device)); print(f"Continuous discount rate (rho): {rho.item():.4f}")

    # --- Training Start ---
    start_time = time.time()
    norm_envs.seed(args.seed)
    obs = norm_envs.reset()
    obs = obs.astype(np.float32)
    dynamic_model_accurate = False; reward_model_accurate = False; global_step = 0 # Added reward model flag

    # --- vmap/grad/hessian/jacrev setup ---
    compute_value_grad_func = None
    compute_reward_grad_func = None
    compute_reward_hessian_func = None
    compute_f_jac_func = None

    if args.use_hjb_loss and TORCH_FUNC_AVAILABLE:
        try:
            # Grad V(s)
            def compute_scalar_value_critic(s):
                if s.dim() == 1: s = s.unsqueeze(0); return critic(s).squeeze()
            compute_value_grad_func = grad(compute_scalar_value_critic)

            # Grad R(s, a) w.r.t a
            def reward_model_wrapper_for_grad(s, a):
                if s.dim() == 1: s = s.unsqueeze(0)
                if a.dim() == 1: a = a.unsqueeze(0)
                return reward_model(s, a).squeeze()
            compute_reward_grad_func = grad(reward_model_wrapper_for_grad, argnums=1)

            # Hessian R(s, a) w.r.t a
            compute_reward_hessian_func = hessian(reward_model_wrapper_for_grad, argnums=1)

            # Jacobian f(s, a) w.r.t a
            func_for_jac = dynamic_model.ode_func
            compute_jac_f_a = jacrev(func_for_jac, argnums=2)
            def get_f2_transpose(s_norm_batch):
                 zero_actions = torch.zeros(s_norm_batch.shape[0], action_dim, device=s_norm_batch.device)
                 def compute_jac_for_single_s(s_single, a_single):
                      s_batch = s_single.unsqueeze(0); a_batch = a_single.unsqueeze(0)
                      jacobian_matrix = compute_jac_f_a(torch.tensor(0.0), s_batch, a_batch);
                      if jacobian_matrix.dim()>2: jacobian_matrix=jacobian_matrix.squeeze(0)
                      if jacobian_matrix.dim()>2: jacobian_matrix=jacobian_matrix.squeeze(0)
                      return jacobian_matrix
                 f2_matrices = vmap(compute_jac_for_single_s)(s_norm_batch, zero_actions)
                 f2_transpose = torch.permute(f2_matrices, (0, 2, 1))
                 return f2_transpose
            compute_f_jac_func = get_f2_transpose

            print("Gradient/Jacobian/Hessian functions for HJB created.")

        except Exception as e: print(f"WARNING: Failed grad/jac/hess func creation: {e}. HJB disabled."); args.use_hjb_loss = False
    elif args.use_hjb_loss: print("WARNING: HJB requested but torch.func unavailable. HJB disabled."); args.use_hjb_loss = False

    # --- Helper Function for HJB Calculation ---
    def calculate_a_star_quad_approx(dVdx_norm, f2_transpose, c1, c2_reg):
        # a* = - c2_reg^{-1} * (c1 + f2^T * dVdx^T)
        if f2_transpose is None or c1 is None or c2_reg is None: return None
        dVdx_col = dVdx_norm.unsqueeze(-1)
        f2T_dVdx = torch.bmm(f2_transpose, dVdx_col).squeeze(-1)
        term1 = c1 + f2T_dVdx
        try:
            # Use torch.linalg.solve
            a_star = torch.linalg.solve(c2_reg, -term1.unsqueeze(-1)).squeeze(-1)
        except torch._C._LinAlgError as e:
             print(f"WARN: Hessian inversion failed during a* calc: {e}. Using pseudo-inverse.")
             try:
                  c2_reg_pinv = torch.linalg.pinv(c2_reg)
                  a_star = torch.bmm(c2_reg_pinv, -term1.unsqueeze(-1)).squeeze(-1)
             except Exception as e2:
                  print(f"ERROR: Hessian pseudo-inverse also failed: {e2}. Cannot compute a*.")
                  return None
        return a_star


    # ========================================================================
    # <<< Main Training Loop >>>
    # ========================================================================
    print(f"Starting training loop for {args.total_timesteps} timesteps...")
    for global_step in range(args.total_timesteps):
        iter_start_time = time.time()

        # --- Action Selection & Environment Interaction ---
        if global_step < args.learning_starts:
            actions = np.array([norm_envs.action_space.sample() for _ in range(norm_envs.num_envs)])
        else:
            with torch.no_grad():
                obs_tensor = torch.Tensor(obs).to(device)
                actions_star = torch.zeros(args.num_envs, action_dim).to(device) # Default
                # Calculate a* using quadratic approximation
                if compute_value_grad_func is not None and compute_reward_grad_func is not None \
                   and compute_reward_hessian_func is not None and compute_f_jac_func is not None:
                    try:
                        dVdx = vmap(compute_value_grad_func)(obs_tensor)
                        f2_T = compute_f_jac_func(obs_tensor)
                        zero_actions_obs = torch.zeros_like(actions_star)
                        c1 = -vmap(compute_reward_grad_func)(obs_tensor, zero_actions_obs)
                        c2 = -vmap(compute_reward_hessian_func)(obs_tensor, zero_actions_obs)
                        c2_reg = c2 + torch.eye(action_dim, device=device) * args.hessian_reg
                        actions_star_unclamped = calculate_a_star_quad_approx(dVdx, f2_T, c1, c2_reg)
                        if actions_star_unclamped is not None:
                            actions_star = actions_star_unclamped # Use unclamped for noise
                    except Exception as e: print(f"WARN: a* calc failed in action selection: {e}")

                noise = torch.normal(0, args.exploration_noise_std, size=actions_star.shape, device=device)
                actions_noisy = actions_star + noise
                actions_clipped = torch.max(torch.min(actions_noisy, action_space_high_t), action_space_low_t)
                actions = actions_clipped.cpu().numpy()

        # Step environment
        next_obs_norm, rewards_raw, dones_combined_np, infos = norm_envs.step(actions) # Get RAW reward
        next_obs_norm = next_obs_norm.astype(np.float32); rewards_raw = rewards_raw.astype(np.float32)
        terminations = np.array([infos[i].get("TimeLimit.truncated", False) == False and dones_combined_np[i] for i in range(args.num_envs)])

        # Log real env returns (use raw reward now)
        if "final_info" in infos: final_infos = infos["final_info"]
        else: final_infos = [i for i in infos if i is not None]
        for info in final_infos:
             if info and "episode" in info:
                  episode_info=info['episode']; print(f"GStep={global_step}, EpReturn={episode_info['r']:.2f}, EpLen={episode_info['l']}");
                  writer.add_scalar("charts/episodic_return",episode_info['r'],global_step); writer.add_scalar("charts/episodic_length",episode_info['l'],global_step); break

        # Store RAW data in replay buffer
        raw_obs = norm_envs.get_original_obs();
        real_next_obs_raw = norm_envs.unnormalize_obs(next_obs_norm);
        for idx, done in enumerate(dones_combined_np):
             try:
                 is_truncated = infos[idx].get("TimeLimit.truncated", False)
                 if done and is_truncated and infos[idx].get("final_observation") is not None:
                     real_next_obs_raw[idx] = infos[idx]["final_observation"].astype(np.float32)
             except IndexError: pass
        rb.add(raw_obs, real_next_obs_raw, actions, rewards_raw, dones_combined_np, infos) # Store RAW reward

        obs = next_obs_norm # Update agent state

        # --- Model Training/Validation (Periodic) ---
        if global_step > args.learning_starts and global_step % args.model_train_freq == 0:
             print(f"\n--- GStep {global_step}: Checking/Training Models ---"); model_train_start_time=time.time();
             buffer_data_raw_tensors = rb.sample(args.model_dataset_size, env=None)
             obs_raw_np=buffer_data_raw_tensors.observations.cpu().numpy(); next_obs_raw_np=buffer_data_raw_tensors.next_observations.cpu().numpy(); actions_np=buffer_data_raw_tensors.actions.cpu().numpy(); rewards_raw_np=buffer_data_raw_tensors.rewards.cpu().numpy(); dones_term_only_np=buffer_data_raw_tensors.dones.cpu().numpy();
             # Normalize states, use raw rewards for reward model target
             obs_norm_np=norm_envs.normalize_obs(obs_raw_np); next_obs_norm_np=norm_envs.normalize_obs(next_obs_raw_np);
             obs_norm_t=torch.tensor(obs_norm_np,dtype=torch.float32).to(device); next_obs_norm_target_t=torch.tensor(next_obs_norm_np,dtype=torch.float32).to(device); actions_t=torch.tensor(actions_np,dtype=torch.float32).to(device);
             rewards_raw_target_t=torch.tensor(rewards_raw_np,dtype=torch.float32).to(device).squeeze(-1); # Target is raw reward
             dones_term_only_t=torch.tensor(dones_term_only_np,dtype=torch.float32).to(device);

             # --- Dynamic Model ---
             non_terminal_mask_dyn=dones_term_only_t.squeeze(-1)==0; dyn_obs_t=obs_norm_t[non_terminal_mask_dyn]; dyn_acts_t=actions_t[non_terminal_mask_dyn]; dyn_targets_t=next_obs_norm_target_t[non_terminal_mask_dyn];
             if len(dyn_obs_t)<2: print("Warn:Not enough samples for dyn model."); dynamic_model_accurate=False
             else:
                 indices=torch.randperm(len(dyn_obs_t),device=device);split=int(len(dyn_obs_t)*(1-args.model_val_ratio));train_idx,val_idx=indices[:split],indices[split:]; train_dataset=TensorDataset(dyn_obs_t[train_idx],dyn_acts_t[train_idx],dyn_targets_t[train_idx]);val_dataset=TensorDataset(dyn_obs_t[val_idx],dyn_acts_t[val_idx],dyn_targets_t[val_idx]); train_loader=DataLoader(train_dataset,batch_size=args.model_train_batch_size,shuffle=True);val_loader=DataLoader(val_dataset,batch_size=args.model_train_batch_size); print(f"DynModel:Tr={len(train_idx)},Vl={len(val_idx)}");best_val_loss=float('inf');patience_counter=0;dynamic_model_accurate=False;best_dyn_state_dict=None;final_model_epoch=0;dynamic_model.train()
                 for epoch in range(args.model_max_epochs):
                     final_model_epoch=epoch; train_loss=0; num_train_batches=0;
                     for _ in range(args.model_updates_per_epoch): epoch_train_loss=train_model_epoch(dynamic_model,dynamics_optimizer,train_loader,device,writer,epoch,global_step,"dynamic",True); train_loss+=epoch_train_loss; num_train_batches+=1;
                     if (epoch+1)%args.model_validation_freq==0:
                         val_loss,val_metrics=validate_model(dynamic_model,val_loader,device,writer,global_step+epoch+1,"dynamic",True); print(f" DynEp {epoch+1}:TrLs={train_loss/num_train_batches if num_train_batches>0 else 0:.5f},VlLs={val_loss:.5f},VlR2={val_metrics['r2']:.3f}")
                         if val_loss<best_val_loss-args.model_val_delta: best_val_loss=val_loss;patience_counter=0;best_dyn_state_dict=copy.deepcopy(dynamic_model.ode_func.state_dict());
                         else: patience_counter+=args.model_validation_freq;
                         if patience_counter>=args.model_val_patience: print(f" Early stop dyn @ ep {epoch+1}."); break
                 if best_dyn_state_dict: dynamic_model.ode_func.load_state_dict(best_dyn_state_dict); print(f" Loaded best dyn model(VlLs:{best_val_loss:.5f})"); final_validation_loss_state = best_val_loss
                 else: print(" No improve dyn valid."); dynamic_model.eval(); final_val_loss, _ = validate_model(dynamic_model, val_loader, device, writer, global_step, "dynamic_final_eval", True); dynamic_model.train(); final_validation_loss_state = final_val_loss
                 dynamic_model.eval(); _, final_val_metrics = validate_model(dynamic_model, val_loader, device, writer, global_step, "dynamic_final_metrics", True); dynamic_model.train(); validation_r2_score = final_val_metrics['r2']; validation_loss_state = final_validation_loss_state; dynamic_model_accurate=(validation_loss_state <= args.dynamic_train_threshold); writer.add_scalar("losses/dynamics_model_validation_loss_final", validation_loss_state, global_step); writer.add_scalar("losses/dynamics_model_R2_final", validation_r2_score, global_step); print(f"DynModel Complete.FinalVlLs:{validation_loss_state:.5f}.FinR2:{validation_r2_score:.3f}.Acc:{dynamic_model_accurate}"); writer.add_scalar("charts/dynamic_model_accurate", float(dynamic_model_accurate), global_step)

             # --- Reward Model ---
             rew_obs_t=obs_norm_t; rew_acts_t=actions_t; rew_targets_t=rewards_raw_target_t; # Target is RAW reward
             if len(rew_obs_t)<2: print("Warn:Not enough samples for rew model."); reward_model_accurate=False
             else:
                 indices=torch.randperm(len(rew_obs_t),device=device);split=int(len(rew_obs_t)*(1-args.model_val_ratio));train_idx,val_idx=indices[:split],indices[split:];
                 train_dataset=TensorDataset(rew_obs_t[train_idx],rew_acts_t[train_idx],rew_targets_t[train_idx]);val_dataset=TensorDataset(rew_obs_t[val_idx],rew_acts_t[val_idx],rew_targets_t[val_idx]);
                 train_loader=DataLoader(train_dataset,batch_size=args.model_train_batch_size,shuffle=True);val_loader=DataLoader(val_dataset,batch_size=args.model_train_batch_size);
                 print(f"RewModel:Tr={len(train_idx)},Vl={len(val_idx)}");best_val_loss=float('inf');patience_counter=0;reward_model_accurate=False;best_rew_state_dict=None;final_model_epoch=0;reward_model.train()
                 for epoch in range(args.model_max_epochs):
                     final_model_epoch=epoch; train_loss=0; num_train_batches=0;
                     for _ in range(args.model_updates_per_epoch): epoch_train_loss=train_model_epoch(reward_model,reward_optimizer,train_loader,device,writer,epoch,global_step,"reward",False); train_loss+=epoch_train_loss; num_train_batches+=1;
                     if (epoch+1)%args.model_validation_freq==0:
                         val_loss,val_metrics=validate_model(reward_model,val_loader,device,writer,global_step+epoch+1,"reward",False); print(f" RewEp {epoch+1}:TrLs={train_loss/num_train_batches if num_train_batches>0 else 0:.5f},VlLs={val_loss:.5f},VlR2={val_metrics['r2']:.3f}")
                         if val_loss<best_val_loss-args.model_val_delta: best_val_loss=val_loss;patience_counter=0;best_rew_state_dict=copy.deepcopy(reward_model.state_dict());
                         else: patience_counter+=args.model_validation_freq;
                         if patience_counter>=args.model_val_patience: print(f" Early stop rew @ ep {epoch+1}."); break
                 if best_rew_state_dict: reward_model.load_state_dict(best_rew_state_dict); print(f" Loaded best rew model(VlLs:{best_val_loss:.5f})"); final_validation_loss_reward = best_val_loss
                 else: print(" No improve rew valid."); reward_model.eval(); final_val_loss, _ = validate_model(reward_model, val_loader, device, writer, global_step, "reward_final_eval", False); reward_model.train(); final_validation_loss_reward = final_val_loss
                 reward_model.eval(); _, final_val_metrics = validate_model(reward_model, val_loader, device, writer, global_step, "reward_final_metrics", False); reward_model.train(); validation_loss_reward = final_validation_loss_reward; validation_r2_score_reward = final_val_metrics['r2'];
                 reward_model_accurate=(validation_loss_reward <= args.reward_train_threshold); # Threshold check on raw reward MSE might need tuning
                 writer.add_scalar("losses/reward_model_validation_loss_final", validation_loss_reward, global_step); writer.add_scalar("losses/reward_model_R2_final", validation_r2_score_reward, global_step); print(f"RewModel Complete.FinalVlLs:{validation_loss_reward:.5f}.FinR2:{validation_r2_score_reward:.3f}.Acc:{reward_model_accurate}"); writer.add_scalar("charts/reward_model_accurate", float(reward_model_accurate), global_step);
             print(f"--- Model Check/Training Finished ---"); model_train_time = time.time() - model_train_start_time; writer.add_scalar("perf/model_train_time", model_train_time, global_step)

        # --- Phase 3: Model Rollout Generation ---
        # (Disabled)

        # --- Agent Training (Value Network Only) ---
        if global_step > args.learning_starts:
            proceed_with_update = True # Removed gating
            if not dynamic_model_accurate: # Only check dynamics accuracy now
                 if global_step % 1000 == 0: print(f"Info: Proceeding with agent update step {global_step}, but dynamics model INACCURATE")

            if proceed_with_update:
                data = rb.sample(args.batch_size, env=None) # Samples Tensors: raw data + term-only dones
                # Convert raw Tensors -> NumPy -> Normalize (OBS ONLY) -> Normalized Tensors
                obs_raw_np=data.observations.cpu().numpy(); actions_np=data.actions.cpu().numpy(); rewards_raw_np=data.rewards.cpu().numpy(); dones_term_only_np=data.dones.cpu().numpy();
                obs_norm_np=norm_envs.normalize_obs(obs_raw_np);
                mb_obs = torch.tensor(obs_norm_np, dtype=torch.float32).to(device)
                mb_actions = torch.tensor(actions_np, dtype=torch.float32).to(device) # Use RAW actions from buffer
                mb_rewards_raw = torch.tensor(rewards_raw_np, dtype=torch.float32).to(device).squeeze(-1) # Use RAW reward [batch]
                mb_dones = torch.tensor(dones_term_only_np, dtype=torch.float32).to(device).squeeze(-1) # Term-only dones [batch]

                # --- Critic Update ---
                terminations_mask = mb_dones.bool(); non_terminations_mask = ~terminations_mask
                mb_obs_critic = mb_obs.clone().requires_grad_(True)
                all_current_v = critic(mb_obs_critic) # V is value of normalized state

                # A. Terminal State Loss: V(s_term) = 0
                v_term=all_current_v[terminations_mask]; terminal_critic_loss=torch.tensor(0.0,device=device)
                if v_term.numel()>0: terminal_critic_loss=F.mse_loss(v_term,torch.zeros_like(v_term))

                # B. Non-Terminal State Loss (HJB)
                hjb_loss_non_term=torch.tensor(0.0,device=device)
                if non_terminations_mask.any() and args.use_hjb_loss and compute_value_grad_func is not None \
                   and compute_reward_grad_func is not None and compute_reward_hessian_func is not None \
                   and compute_f_jac_func is not None:

                    obs_non_term = mb_obs_critic[non_terminations_mask]
                    v_non_term = all_current_v[non_terminations_mask]
                    actions_buffer_raw = mb_actions[non_terminations_mask] # Raw actions from buffer
                    rewards_buffer_raw = mb_rewards_raw[non_terminations_mask] # Raw rewards from buffer

                    # Calculate dV/dx, f1, f2, c1, c2, a*
                    try:
                        dVdx_non_term = vmap(compute_value_grad_func)(obs_non_term)
                        f1_non_term = get_f1(obs_non_term) # f1(s_norm)
                        f2_T_non_term = compute_f_jac_func(obs_non_term) # f2(s_norm)^T

                        zero_actions_non_term = torch.zeros_like(actions_buffer_non_term)
                        c1 = -vmap(compute_reward_grad_func)(obs_non_term, zero_actions_non_term) # -grad_a R(s_norm, 0)
                        c2 = -vmap(compute_reward_hessian_func)(obs_non_term, zero_actions_non_term) # -hess_aa R(s_norm, 0)
                        c2_reg = c2 + torch.eye(action_dim, device=device) * args.hessian_reg # Regularize Hessian

                        if f2_T_non_term is not None and c1 is not None and c2_reg is not None:
                            a_star_non_term = calculate_a_star_quad_approx(dVdx_non_term, f2_T_non_term, c1, c2_reg) # Unclamped a*

                            if a_star_non_term is not None:
                                # Calculate H(a*) = R(s, a*) + <dVdx, f(s, a*)> using learned models
                                with torch.no_grad():
                                    # Need f(s, a*) = f1 + f2 * a*
                                    f_star_non_term = f1_non_term + torch.bmm(torch.permute(f2_T_non_term, (0, 2, 1)), a_star_non_term.unsqueeze(-1)).squeeze(-1)
                                    # Need R(s, a*) - predict using reward model (predicts raw reward)
                                    r_star_non_term = reward_model(obs_non_term, a_star_non_term)

                                hamiltonian_star = r_star_non_term + torch.einsum("bi,bi->b", dVdx_non_term, f_star_non_term)

                                # HJB residual: H(a*) - rho * V
                                hjb_residual = hamiltonian_star - rho * v_non_term
                                hjb_loss_non_term = 0.5 * (hjb_residual**2).mean()
                            else: print("WARN: HJB skipped due to a* calculation failure.")
                        else: print("WARN: HJB skipped due to f2/c1/c2 calculation failure.")
                    except Exception as e: print(f"HJB Error:{e}"); hjb_loss_non_term=torch.tensor(0.0,device=device)

                # C. Total Critic Loss & Update
                critic_loss = args.hjb_coef * hjb_loss_non_term + args.terminal_coeff * terminal_critic_loss
                critic_optimizer.zero_grad();
                critic_loss.backward()
                if args.grad_norm_clip is not None: nn.utils.clip_grad_norm_(critic.parameters(), args.grad_norm_clip)
                critic_optimizer.step()

                # Logging Critic losses
                writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
                writer.add_scalar("losses/critic_terminal", terminal_critic_loss.item(), global_step)
                writer.add_scalar("losses/critic_hjb_non_term", hjb_loss_non_term.item(), global_step)
                writer.add_scalar("metrics/critic_value_mean", all_current_v.mean().item(), global_step)

        # Log SPS occasionally
        if global_step > 0 and global_step % 1000 == 0:
             sps = int(global_step / (time.time() - start_time)); print(f"GStep: {global_step}, SPS: {sps}"); writer.add_scalar("charts/SPS", sps, global_step)

    # --- End of Training Loop ---
    # --- Saving & Evaluation ---
    critic_final = critic
    if args.save_model:run_folder=f"runs/{run_name}";os.makedirs(run_folder,exist_ok=True);
    critic_model_path = f"{run_folder}/{args.exp_name}_critic.cleanrl_model"; torch.save(critic_final.state_dict(), critic_model_path); print(f"Critic saved: {critic_model_path}");
    dynamics_ode_path = f"{run_folder}/{args.exp_name}_dynamics_odefunc.cleanrl_model"; torch.save(dynamic_model.ode_func.state_dict(), dynamics_ode_path); print(f"Dynamics ODEFunc saved: {dynamics_ode_path}");
    reward_model_path = f"{run_folder}/{args.exp_name}_reward_model.cleanrl_model"; torch.save(reward_model.state_dict(), reward_model_path); print(f"Reward model saved: {reward_model_path}");
    norm_stats_path = f"{run_folder}/{args.exp_name}_vecnormalize.pkl"; norm_envs.save(norm_stats_path); print(f"Normalization stats saved: {norm_stats_path}");
    if args.save_model:
        print("\nEvaluating agent performance...");eval_episodes=10;eval_seeds=range(args.seed+100,args.seed+100+eval_episodes);eval_returns_raw=[]
        # Evaluation needs policy derived from V and models
        eval_critic = ValueNetwork(norm_envs).to(device); eval_critic.load_state_dict(torch.load(critic_model_path, map_location=device)); eval_critic.eval()
        eval_dynamic_model = DynamicModel(obs_dim, action_dim, args.env_dt, device).to(device); eval_dynamic_model.ode_func.load_state_dict(torch.load(dynamics_ode_path, map_location=device)); eval_dynamic_model.eval()
        # <<< Re-add reward model loading for eval if needed by H? Not needed for a* >>>
        eval_reward_model = RewardModel(obs_dim, action_dim).to(device); eval_reward_model.load_state_dict(torch.load(reward_model_path, map_location=device)); eval_reward_model.eval()

        # Need grad/jac/hess functions for eval critic/models
        eval_compute_value_grad_func = None; eval_get_f2_transpose = None; eval_calculate_a_star_quad_approx = None
        eval_compute_reward_grad_func = None; eval_compute_reward_hessian_func = None
        if TORCH_FUNC_AVAILABLE:
             try:
                 def eval_compute_scalar_value_critic(s):
                      if s.dim()==1: s=s.unsqueeze(0); return eval_critic(s).squeeze()
                 eval_compute_value_grad_func = grad(eval_compute_scalar_value_critic)

                 def eval_dynamic_ode_func_wrapper(t, s, a): return eval_dynamic_model.ode_func(t, s, a)
                 eval_compute_jac_f_a = jacrev(eval_dynamic_ode_func_wrapper, argnums=2)
                 def eval_compute_jac_for_single_s(s_single):
                      s_batch=s_single.unsqueeze(0); a_zeros_single=torch.zeros(1,action_dim,device=s_single.device)
                      jacobian_matrix=eval_compute_jac_f_a(torch.tensor(0.0),s_batch,a_zeros_single);
                      if jacobian_matrix.dim()>2: jacobian_matrix=jacobian_matrix.squeeze(0)
                      if jacobian_matrix.dim()>2: jacobian_matrix=jacobian_matrix.squeeze(0)
                      return jacobian_matrix
                 def eval_get_f2_transpose(s_norm_batch): return torch.permute(vmap(eval_compute_jac_for_single_s)(s_norm_batch), (0,2,1))

                 # <<< Add reward grad/hessian funcs for eval >>>
                 def eval_reward_model_wrapper(s, a):
                      if s.dim()==1: s=s.unsqueeze(0)
                      if a.dim()==1: a=a.unsqueeze(0)
                      return eval_reward_model(s, a).squeeze()
                 eval_compute_reward_grad_func = grad(eval_reward_model_wrapper, argnums=1)
                 eval_compute_reward_hessian_func = hessian(eval_reward_model_wrapper, argnums=1)

                 # <<< Fix: Pass state batch to eval helper >>>
                 def eval_calculate_a_star_quad_approx(s_norm_batch, dVdx_norm, f2_transpose):
                    if f2_transpose is None: return None
                    # Need reward grad/hessian funcs defined above
                    if eval_compute_reward_grad_func is None or eval_compute_reward_hessian_func is None:
                        print("WARN: Eval reward grad/hessian func not available.")
                        return None
                    zero_actions = torch.zeros(s_norm_batch.shape[0], action_dim, device=device)
                    c1 = -vmap(eval_compute_reward_grad_func)(s_norm_batch, zero_actions)
                    c2 = -vmap(eval_compute_reward_hessian_func)(s_norm_batch, zero_actions)
                    c2_reg = c2 + torch.eye(action_dim, device=device) * args.hessian_reg

                    dVdx_col = dVdx_norm.unsqueeze(-1)
                    f2T_dVdx = torch.bmm(f2_transpose, dVdx_col).squeeze(-1)
                    term1 = c1 + f2T_dVdx
                    try: a_star = torch.linalg.solve(c2_reg, -term1.unsqueeze(-1)).squeeze(-1)
                    except Exception: a_star = torch.bmm(torch.linalg.pinv(c2_reg), -term1.unsqueeze(-1)).squeeze(-1)
                    return a_star # Return unclamped

             except Exception as e: print(f"Eval grad/jac/hess func setup failed: {e}")

        for seed in eval_seeds:
            eval_envs_base=DummyVecEnv([make_env(args.env_id,seed,False,f"{run_name}-eval-seed{seed}")]);eval_norm_envs=VecNormalize.load(norm_stats_path,eval_envs_base);eval_norm_envs.training=False;eval_norm_envs.norm_reward=False;
            obs_norm_np=eval_norm_envs.reset(seed=seed);done=False;episode_return_raw=0;num_steps=0;max_steps=1000
            while not done and num_steps<max_steps:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs_norm_np, dtype=torch.float32).to(device)
                    action = np.zeros((args.num_envs, action_dim)) # Default
                    if eval_compute_value_grad_func is not None and eval_get_f2_transpose is not None and eval_calculate_a_star_quad_approx is not None:
                        try:
                            dVdx = vmap(eval_compute_value_grad_func)(obs_tensor)
                            f2_T = eval_get_f2_transpose(obs_tensor)
                            if f2_T is not None:
                                # <<< Fix: Pass obs_tensor to eval helper >>>
                                action_star_unclamped = eval_calculate_a_star_quad_approx(obs_tensor, dVdx, f2_T)
                                if action_star_unclamped is not None:
                                    # <<< Fix: Clamp final action for eval >>>
                                    action_star_clamped = torch.max(torch.min(action_star_unclamped, action_space_high_t), action_space_low_t)
                                    action = action_star_clamped.cpu().numpy()
                                else: action = np.array([eval_norm_envs.action_space.sample() for _ in range(eval_norm_envs.num_envs)])
                            else: action = np.array([eval_norm_envs.action_space.sample() for _ in range(eval_norm_envs.num_envs)])
                        except Exception as e: print(f"Eval action calc failed: {e}"); action = np.array([eval_norm_envs.action_space.sample() for _ in range(eval_norm_envs.num_envs)])
                    else: action = np.array([eval_norm_envs.action_space.sample() for _ in range(eval_norm_envs.num_envs)])

                obs_norm_np,reward_raw_step,term,trunc,info=eval_norm_envs.step(action);done=term[0]or trunc[0];episode_return_raw+=reward_raw_step[0];num_steps+=1
            eval_returns_raw.append(episode_return_raw);print(f"  Eval Seed {seed}: Raw Episodic Return={episode_return_raw:.2f} ({num_steps} steps)");eval_envs_base.close()
        mean_eval_return_raw=np.mean(eval_returns_raw);std_eval_return_raw=np.std(eval_returns_raw);print(f"Evaluation complete. Avg Return: {mean_eval_return_raw:.2f} +/- {std_eval_return_raw:.2f}");
        for idx,r in enumerate(eval_returns_raw):writer.add_scalar("eval/raw_episodic_return",r,idx)
        if args.upload_model:print("Uploading models to Hugging Face Hub...");# ... (HF Upload logic) ...

    # --- Cleanup ---
    norm_envs.close(); writer.close(); print("\nTraining finished.")

