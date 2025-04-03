# Full HJB Actor-Critic Code with Stochastic Policy, TorchODE Dynamics Model,
# VecNormalize, Raw Buffer Storage, Validation, Early Stopping,
# HJB (vmap/grad), R2 Logging
# --- Uses Inferred Control Cost Weight, No Reward Model ---

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
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# Required imports
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecEnv, VecNormalize, DummyVecEnv

try:
    from torch.func import grad, vmap, jacrev

    TORCH_FUNC_AVAILABLE = True
    print("Imported grad, vmap, jacrev from torch.func")
except ImportError:
    try:
        from functorch import grad, vmap, jacrev

        TORCH_FUNC_AVAILABLE = True
        print("Imported grad, vmap, jacrev from functorch")
    except ImportError:
        print(
            "WARNING: torch.func / functorch not available. HJB calculations will be skipped."
        )
        TORCH_FUNC_AVAILABLE = False
try:
    import torchode as to

    TORCHODE_AVAILABLE = True
    print("Imported torchode.")
except ImportError:
    print("FATAL: torchode not found. Exiting.")
    TORCHODE_AVAILABLE = False
    exit()

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="stable_baselines3.common.buffers"
)


@dataclass
class Args:
    exp_name: str = (
        os.path.basename(__file__)[: -len(".py")] + "_hjb_vi_ode_inferred_cost"
    )  # VI=ValueIteration-like
    """the name of this experiment"""
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = True
    upload_model: bool = False
    hf_entity: str = ""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1000000
    learning_rate: float = 1e-3  # LR for Critic (V function)
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    batch_size: int = 256  # Agent update batch size
    learning_starts: int = 5000
    exploration_noise_std: float = 0.1  # Std dev for noise added to optimal action
    grad_norm_clip: Optional[float] = 1.0

    # Model Training Args
    model_train_freq: int = 250
    model_dataset_size: int = 50_000
    dynamics_learning_rate: float = 1e-3
    # Removed reward_learning_rate
    dynamic_train_threshold: float = 0.01  # Stricter threshold?
    # Removed reward_train_threshold
    model_val_ratio: float = 0.2
    model_val_patience: int = 10
    model_val_delta: float = 1e-5
    model_max_epochs: int = 50
    model_train_batch_size: int = 256
    model_validation_freq: int = 5
    model_updates_per_epoch: int = 1

    # Model Rollout Args (Currently unused)
    model_rollout_freq: int = 10000000
    model_rollout_length: int = 1
    num_model_rollout_starts: int = 0

    # HJB Residual Args
    hjb_coef: float = 1.0  # HJB is the main loss term now
    use_hjb_loss: bool = True
    terminal_coeff: float = 1.0
    ctrl_cost_weight: Optional[float] = None  # <<< Let's try to infer this >>>
    """Weight of the quadratic control cost (- reward = l_state + ctrl_cost_weight * ||a||^2). If None, try to infer from env."""

    # Env Args
    num_envs: int = 1
    env_dt: float = 0.02

    # Runtime filled
    minibatch_size: int = field(init=False)
    rho: float = field(init=False)


# --- Environment Creation ---
def make_env(env_id, seed, idx, capture_video, run_name):  # (Same)
    def thunk():
        render_mode = "rgb_array" if capture_video and idx == 0 else None
        try:
            env = gym.make(env_id, render_mode=render_mode)
        except Exception as e:
            print(
                f"Warning: Failed render_mode='{render_mode}'. Error: {e}. Defaulting."
            )
            env = gym.make(env_id)
        if capture_video and idx == 0 and env.render_mode == "rgb_array":
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}", episode_trigger=lambda x: x % 50 == 0
            )
        if isinstance(env.observation_space, gym.spaces.Dict):
            env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env.action_space.seed(seed + idx)
        return env

    return thunk


# --- Utilities ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):  # (Same)
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# --- Dynamics Model (Neural ODE using TorchODE) ---
class ODEFunc(nn.Module):  # (Same)
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        hidden_size = 256
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + action_dim, hidden_size)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden_size, obs_dim)),
        )
        print(f"ODEFunc: In={obs_dim+action_dim}, Out={obs_dim}")

    def forward(self, t, x_norm, a):
        return self.net(torch.cat([x_norm.float(), a.float()], dim=-1))


class DynamicModel(nn.Module):  # (Same)
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
        self.adjoint = to.AutoDiffAdjoint(
            step_method=self.step_method, step_size_controller=self.step_size_controller
        )
        print(f"DynamicModel: TorchODE (Solver: Euler, dt={self.dt})")

    def forward(self, initial_obs_norm, actions_norm):
        batch_size = initial_obs_norm.shape[0]
        dt0 = torch.full((batch_size,), self.dt / 5, device=self.device)
        t_span_tensor = torch.tensor([0.0, self.dt], device=self.device)
        t_eval = t_span_tensor.unsqueeze(0).repeat(batch_size, 1)
        problem = to.InitialValueProblem(
            y0=initial_obs_norm.float(),
            t_eval=t_eval,
        )
        t_eval_actual, sol_ys = to.odeint(
            self.ode_func,
            initial_obs_norm.float(),
            t_eval[0],
            solver=self.step_method,
            args=(actions_norm.float(),),
            dt0=dt0[0],
        )
        final_state_pred_norm = sol_ys[1]
        return final_state_pred_norm


# --- Reward Model ---
# <<< Removed RewardModel class >>>


# --- Agent Network Definitions ---
# <<< Removed Actor class >>>
class ValueNetwork(nn.Module):  # Renamed from HJBCritic
    def __init__(self, env: VecEnv):
        super().__init__()
        obs_dim = np.array(env.observation_space.shape).prod()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )
        print("Initialized ValueNetwork (Critic).")

    def forward(self, x_norm):
        return self.net(x_norm).squeeze(-1)


# --- Utility Functions ---
def calculate_metrics(preds, targets):  # (Same)
    mse = F.mse_loss(preds, targets).item()
    mae = F.l1_loss(preds, targets).item()
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = (1 - ss_res / ss_tot).item() if ss_tot > 1e-8 else -float("inf")
    return {"mse": mse, "mae": mae, "r2": r2}


def train_model_epoch(
    model,
    optimizer,
    train_loader,
    device,
    writer,
    epoch,
    global_step,
    model_name,
    is_dynamic_model,
):  # (Same)
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch_idx, batch_data in enumerate(train_loader):
        obs_norm, actions, targets_norm = [d.to(device) for d in batch_data]
    if is_dynamic_model:
        preds_norm = model(obs_norm, actions)
    else:
        raise ValueError(
            "train_model_epoch called without dynamic model flag"
        )  # No reward model
    loss = F.mse_loss(preds_norm, targets_norm)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    num_batches += 1
    if batch_idx % 50 == 0:
        writer.add_scalar(
            f"losses/{model_name}_batch_mse",
            loss.item(),
            global_step + epoch * len(train_loader) + batch_idx,
        )
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate_model(
    model, val_loader, device, writer, global_step, model_name, is_dynamic_model
):  # (Same)
    model.eval()
    all_preds_norm = []
    all_targets_norm = []
    with torch.no_grad():
        for batch_data in val_loader:
            obs_norm, actions, targets_norm = [d.to(device) for d in batch_data]
        if is_dynamic_model:
            preds_norm = model(obs_norm, actions)
        else:
            raise ValueError(
                "validate_model called without dynamic model flag"
            )  # No reward model
        all_preds_norm.append(preds_norm)
        all_targets_norm.append(targets_norm)
    if not all_preds_norm:
        return float("inf"), {
            "mse": float("inf"),
            "mae": float("inf"),
            "r2": -float("inf"),
        }
    all_preds_norm = torch.cat(all_preds_norm, dim=0)
    all_targets_norm = torch.cat(all_targets_norm, dim=0)
    val_metrics = calculate_metrics(all_preds_norm, all_targets_norm)
    val_loss = val_metrics["mse"]
    writer.add_scalar(f"losses/{model_name}_val_mse", val_metrics["mse"], global_step)
    writer.add_scalar(f"metrics/{model_name}_val_mae", val_metrics["mae"], global_step)
    writer.add_scalar(f"metrics/{model_name}_val_r2", val_metrics["r2"], global_step)
    return val_loss, val_metrics


# --- Main Execution ---
if __name__ == "__main__":
    if not TORCHODE_AVAILABLE:
        exit()
    if not TORCH_FUNC_AVAILABLE:
        print("FATAL: torch.func or functorch required for HJB gradients.")
        exit()
    args = tyro.cli(Args)

    # Calculate dependent args
    if args.model_rollout_length > 0:
        args.num_model_rollout_starts = args.num_model_rollout_starts
    else:
        args.num_model_rollout_starts = 0
        print("Warning: model_rollout_length <= 0.")
    args.rho = -math.log(args.gamma) if args.gamma > 0 and args.gamma < 1 else 0.0
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # --- Logging Setup ---
    if args.track:
        try:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
            print("WandB enabled.")
        except ImportError:
            print("WARNING: wandb not installed.")
            args.track = False
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    print(f"Run name: {run_name}")
    print(f"Arguments: {vars(args)}")

    # --- Seeding & Device ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # --- Environment Setup with VecNormalize ---
    print("Setting up environment...")
    envs = DummyVecEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    norm_envs = VecNormalize(
        envs,
        gamma=args.gamma,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    print("VecNormalize enabled.")
    try:
        env_dt = norm_envs.get_attr("dt")[0]
        print(f"Detected env dt: {env_dt}")
    except Exception:
        print(f"Warning: Could not detect env dt. Using default: {args.env_dt}")
        env_dt = args.env_dt
    args.env_dt = env_dt
    # <<< Change: Infer ctrl_cost_weight >>>
    try:
        ctrl_cost_weight = norm_envs.get_attr("_ctrl_cost_weight")[0]
        print(f"Detected env ctrl_cost_weight: {ctrl_cost_weight}")
        if args.ctrl_cost_weight is not None:
            print(
                f"Warning: Overriding inferred ctrl_cost_weight with args.ctrl_cost_weight={args.ctrl_cost_weight}"
            )
            ctrl_cost_weight = args.ctrl_cost_weight
    except Exception:
        if args.ctrl_cost_weight is None:
            # Default if not specified and not found in env
            args.ctrl_cost_weight = 0.1  # Default similar to HalfCheetah
            print(
                f"Warning: Could not detect env ctrl_cost_weight. Using default: {args.ctrl_cost_weight}"
            )
        ctrl_cost_weight = (
            args.ctrl_cost_weight
        )  # Use the one from args (might be default)
    # Ensure it's not zero for optimal action calculation
    if ctrl_cost_weight <= 0:
        raise ValueError("ctrl_cost_weight must be positive for this HJB formulation.")

    obs_space = norm_envs.observation_space
    action_space = norm_envs.action_space
    obs_dim = np.array(obs_space.shape).prod()
    action_dim = np.prod(action_space.shape)

    # --- Agent, Models, Optimizers ---
    # <<< Change: Removed Actor, RewardModel, policy_optimizer, reward_optimizer, alpha setup >>>
    critic = ValueNetwork(norm_envs).to(device)
    critic_optimizer = optim.AdamW(critic.parameters(), lr=args.learning_rate)
    # Use main LR
    dynamic_model = DynamicModel(obs_dim, action_dim, args.env_dt, device).to(device)
    dynamics_optimizer = optim.AdamW(
        dynamic_model.ode_func.parameters(), lr=args.dynamics_learning_rate
    )

    # --- Replay Buffer for RAW Data ---
    raw_obs_space = norm_envs.unwrapped.observation_space
    raw_action_space = norm_envs.unwrapped.action_space
    raw_obs_space.dtype = np.float32
    print(f"Replay buffer storing RAW data. Obs Shape: {raw_obs_space.shape}")
    sb3_buffer_device = "cpu"
    rb = ReplayBuffer(
        args.buffer_size,
        raw_obs_space,
        raw_action_space,
        device=sb3_buffer_device,
        n_envs=args.num_envs,
        handle_timeout_termination=True,
    )
    print("Replay buffer configured with handle_timeout_termination=True")

    # --- Continuous Discount Rate ---
    rho = -torch.log(torch.tensor(args.gamma, device=device))
    print(f"Continuous discount rate (rho): {rho.item():.4f}")

    # --- Training Start ---
    start_time = time.time()
    norm_envs.seed(args.seed)
    obs = norm_envs.reset()
    obs = obs.astype(np.float32)
    dynamic_model_accurate = False
    global_step = 0
    # <<< Removed reward_model_accurate >>>

    # --- vmap/grad setup ---
    compute_value_grad_func = None
    if args.use_hjb_loss and TORCH_FUNC_AVAILABLE:
        try:

            def compute_scalar_value_critic(single_obs_tensor):
                if single_obs_tensor.dim() == 1:
                    single_obs_tensor = single_obs_tensor.unsqueeze(0)
                return critic(single_obs_tensor).squeeze()

            compute_value_grad_func = grad(compute_scalar_value_critic)
            print("Value gradient function for HJB created.")
        except Exception as e:
            print(f"WARNING: Failed grad func creation: {e}. HJB disabled.")
            args.use_hjb_loss = False
    elif args.use_hjb_loss:
        print("WARNING: HJB requested but torch.func unavailable. HJB disabled.")
        args.use_hjb_loss = False

    # --- Define functions for f1, f2, a_star ---
    # Need access to dynamic_model and compute_value_grad_func
    def get_f1(s_norm_batch):
        zero_actions = torch.zeros(
            s_norm_batch.shape[0], action_dim, device=s_norm_batch.device
        )
        with torch.no_grad():
            f1_pred = dynamic_model.ode_func(0, s_norm_batch, zero_actions)
        return f1_pred

    # Define jacobian calculation function here to capture dynamic_model
    func_for_jac = dynamic_model.ode_func  # Get the function once
    compute_jac_f_a = None
    if TORCH_FUNC_AVAILABLE:
        try:
            compute_jac_f_a = jacrev(
                func_for_jac, argnums=2
            )  # Jacobian wrt action (arg 2)
        except Exception as e:
            print(f"WARN: jacrev failed: {e}. Cannot compute f2.")
            TORCH_FUNC_AVAILABLE = False  # Disable HJB if jacrev fails

    def get_f2_transpose(s_norm_batch):
        if compute_jac_f_a is None:
            return None
        try:
            # Need zero actions matching batch size
            zero_actions = torch.zeros(
                s_norm_batch.shape[0], action_dim, device=s_norm_batch.device
            )

            # Compute jacobian per sample using vmap
            def compute_jac_for_single_s(s_single, a_single):  # Pass action too
                s_batch = s_single.unsqueeze(0)
                a_batch = a_single.unsqueeze(0)
                jacobian_matrix = compute_jac_f_a(
                    torch.tensor(0.0), s_batch, a_batch
                )  # Pass t=0
                return jacobian_matrix.squeeze(0).squeeze(1) # Squeeze batch dim [0] and inner batch dim [1] -> [obs_dim, action_dim]

            # Vmap over state and action batches
            f2_matrices = vmap(compute_jac_for_single_s)(
                s_norm_batch, zero_actions
            )  # Shape should now be [batch, obs_dim, action_dim]
            f2_transpose = torch.permute(
                f2_matrices, (0, 2, 1)
            )  # Shape [batch, action_dim, obs_dim]
            return f2_transpose
        except Exception as e:
            print(f"ERROR computing f2 Jacobian: {e}")
            return None

    def calculate_a_star(dVdx_norm, f2_transpose):
        if f2_transpose is None:
            return None
        # a* = -1/(2*C) * f2^T * dVdx^T, where C = ctrl_cost_weight
        # dVdx_norm shape: [batch, obs_dim] -> needs [batch, obs_dim, 1]
        # f2_transpose shape: [batch, action_dim, obs_dim]
        dVdx_col = dVdx_norm.unsqueeze(-1)
        # Result = -1/(2C) * [batch, action_dim, obs_dim] @ [batch, obs_dim, 1] -> [batch, action_dim, 1]
        # Use the inferred ctrl_cost_weight
        a_star = (-1.0 / (2.0 * ctrl_cost_weight)) * torch.bmm(
            f2_transpose, dVdx_col
        ).squeeze(-1)
        return a_star

    # ========================================================================
    # <<< Main Training Loop >>>
    # ========================================================================
    print(f"Starting training loop for {args.total_timesteps} timesteps...")
    for global_step in range(args.total_timesteps):
        iter_start_time = time.time()
        # LR Annealing block removed

        # --- Action Selection & Environment Interaction ---
        if global_step < args.learning_starts:
            actions = np.array(
                [norm_envs.action_space.sample() for _ in range(norm_envs.num_envs)]
            )
        else:
            with torch.no_grad():
                # Calculate optimal action a* and add noise
                obs_tensor = torch.Tensor(obs).to(device)
                actions_star = torch.zeros(args.num_envs, action_dim).to(
                    device
                )  # Default if calculation fails
                if compute_value_grad_func is not None:
                    try:
                        dVdx = vmap(compute_value_grad_func)(obs_tensor)
                        f2_T = get_f2_transpose(obs_tensor)
                        if f2_T is not None:
                            actions_star_calc = calculate_a_star(dVdx, f2_T)
                            if actions_star_calc is not None:
                                actions_star = actions_star_calc
                        # else: print("WARN: f2 failed in action selection") # Avoid spamming logs
                    except Exception as e:
                        print(f"WARN: a* calc failed in action selection: {e}")
                # else: print("WARN: HJB/grad func disabled") # Avoid spamming logs

                noise = torch.normal(
                    0,
                    args.exploration_noise_std,
                    size=actions_star.shape,
                    device=device,
                )
                actions_noisy = actions_star + noise
                # Clip actions to environment bounds
                low = torch.tensor(action_space.low, device=device)
                high = torch.tensor(action_space.high, device=device)
                actions_clipped = torch.max(torch.min(actions_noisy, high), low)
                actions = actions_clipped.cpu().numpy()

        next_obs_norm, rewards_norm, dones_combined_np, infos = norm_envs.step(actions)
        next_obs_norm = next_obs_norm.astype(np.float32)
        rewards_norm = rewards_norm.astype(np.float32)
        terminations = np.array(
            [
                infos[i].get("TimeLimit.truncated", False) == False
                and dones_combined_np[i]
                for i in range(args.num_envs)
            ]
        )

        # Log real env returns
        if "final_info" in infos:
            final_infos = infos["final_info"]
        else:
            final_infos = [i for i in infos if i is not None]
        for info in final_infos:
            if info and "episode" in info:
                episode_info = info["episode"]
                return_val = float(episode_info["r"])
                length_val = int(episode_info["l"])
                print(
                    f"GStep={global_step}, EpReturn(Norm)={return_val:.2f}, EpLen={length_val}"
                )
                writer.add_scalar(
                    "charts/episodic_return_norm", return_val, global_step
                )
                writer.add_scalar("charts/episodic_length", length_val, global_step)
                break

        # Store RAW data in replay buffer
        raw_obs = norm_envs.get_original_obs()
        raw_reward = norm_envs.get_original_reward()
        real_next_obs_raw = norm_envs.unnormalize_obs(next_obs_norm)
        for idx, done in enumerate(dones_combined_np):
            try:
                is_truncated = infos[idx].get("TimeLimit.truncated", False)
                if (
                    done
                    and is_truncated
                    and infos[idx].get("final_observation") is not None
                ):
                    real_next_obs_raw[idx] = infos[idx]["final_observation"].astype(
                        np.float32
                    )
            except IndexError:
                pass
        rb.add(
            raw_obs, real_next_obs_raw, actions, raw_reward, dones_combined_np, infos
        )  # Pass combined done

        obs = next_obs_norm  # Update agent state

        # --- Model Training/Validation (Periodic) ---
        if (
            global_step > args.learning_starts
            and global_step % args.model_train_freq == 0
        ):
            print(f"\n--- GStep {global_step}: Checking/Training Dynamics Model ---")
            model_train_start_time = time.time()
            buffer_data_raw_tensors = rb.sample(args.model_dataset_size, env=None)
            obs_raw_np = buffer_data_raw_tensors.observations.cpu().numpy()
            next_obs_raw_np = buffer_data_raw_tensors.next_observations.cpu().numpy()
            actions_np = buffer_data_raw_tensors.actions.cpu().numpy()
            dones_term_only_np = buffer_data_raw_tensors.dones.cpu().numpy()
            obs_norm_np = norm_envs.normalize_obs(obs_raw_np)
            next_obs_norm_np = norm_envs.normalize_obs(next_obs_raw_np)
            obs_norm_t = torch.tensor(obs_norm_np, dtype=torch.float32).to(device)
            next_obs_norm_target_t = torch.tensor(
                next_obs_norm_np, dtype=torch.float32
            ).to(device)
            actions_t = torch.tensor(actions_np, dtype=torch.float32).to(device)
            dones_term_only_t = torch.tensor(
                dones_term_only_np, dtype=torch.float32
            ).to(device)
            # --- Dynamic Model ---
            non_terminal_mask_dyn = dones_term_only_t.squeeze(-1) == 0
            dyn_obs_t = obs_norm_t[non_terminal_mask_dyn]
            dyn_acts_t = actions_t[non_terminal_mask_dyn]
            dyn_targets_t = next_obs_norm_target_t[non_terminal_mask_dyn]
            if len(dyn_obs_t) < 2:
                print("Warn:Not enough samples for dyn model.")
                dynamic_model_accurate = False
            else:
                indices = torch.randperm(len(dyn_obs_t), device=device)
                split = int(len(dyn_obs_t) * (1 - args.model_val_ratio))
                train_idx, val_idx = indices[:split], indices[split:]
                train_dataset = TensorDataset(
                    dyn_obs_t[train_idx],
                    dyn_acts_t[train_idx],
                    dyn_targets_t[train_idx],
                )
                val_dataset = TensorDataset(
                    dyn_obs_t[val_idx], dyn_acts_t[val_idx], dyn_targets_t[val_idx]
                )
                train_loader = DataLoader(
                    train_dataset, batch_size=args.model_train_batch_size, shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=args.model_train_batch_size
                )
                print(f"DynModel:Tr={len(train_idx)},Vl={len(val_idx)}")
                best_val_loss = float("inf")
                patience_counter = 0
                dynamic_model_accurate = False
                best_dyn_state_dict = None
                final_model_epoch = 0
                dynamic_model.train()
                for epoch in range(args.model_max_epochs):
                    final_model_epoch = epoch
                    train_loss = 0
                    num_train_batches = 0
                    for _ in range(args.model_updates_per_epoch):
                        epoch_train_loss = train_model_epoch(
                            dynamic_model,
                            dynamics_optimizer,
                            train_loader,
                            device,
                            writer,
                            epoch,
                            global_step,
                            "dynamic",
                            True,
                        )
                        train_loss += epoch_train_loss
                        num_train_batches += 1
                    if (epoch + 1) % args.model_validation_freq == 0:
                        val_loss, val_metrics = validate_model(
                            dynamic_model,
                            val_loader,
                            device,
                            writer,
                            global_step + epoch + 1,
                            "dynamic",
                            True,
                        )
                        print(
                            f" DynEp {epoch+1}:TrLs={train_loss/num_train_batches if num_train_batches>0 else 0:.5f},VlLs={val_loss:.5f},VlR2={val_metrics['r2']:.3f}"
                        )
                        if val_loss < best_val_loss - args.model_val_delta:
                            best_val_loss = val_loss
                            patience_counter = 0
                            best_dyn_state_dict = copy.deepcopy(
                                dynamic_model.ode_func.state_dict()
                            )
                            # Save ODEFunc state
                        else:
                            patience_counter += args.model_validation_freq
                        if patience_counter >= args.model_val_patience:
                            print(f" Early stop dyn @ ep {epoch+1}.")
                            break
                if best_dyn_state_dict:
                    dynamic_model.ode_func.load_state_dict(best_dyn_state_dict)
                    print(f" Loaded best dyn model(VlLs:{best_val_loss:.5f})")
                    final_validation_loss_state = best_val_loss
                else:
                    print(" No improve dyn valid.")
                    dynamic_model.eval()
                    final_val_loss, _ = validate_model(
                        dynamic_model,
                        val_loader,
                        device,
                        writer,
                        global_step,
                        "dynamic_final_eval",
                        True,
                    )
                    dynamic_model.train()
                    final_validation_loss_state = final_val_loss
                dynamic_model.eval()
                _, final_val_metrics = validate_model(
                    dynamic_model,
                    val_loader,
                    device,
                    writer,
                    global_step,
                    "dynamic_final_metrics",
                    True,
                )
                dynamic_model.train()
                validation_r2_score = final_val_metrics["r2"]
                validation_loss_state = final_validation_loss_state
                dynamic_model_accurate = (
                    validation_loss_state <= args.dynamic_train_threshold
                )
                writer.add_scalar(
                    "losses/dynamics_model_validation_loss_final",
                    validation_loss_state,
                    global_step,
                )
                writer.add_scalar(
                    "losses/dynamics_model_R2_final", validation_r2_score, global_step
                )
                print(
                    f"DynModel Complete.FinalVlLs:{validation_loss_state:.5f}.FinR2:{validation_r2_score:.3f}.Acc:{dynamic_model_accurate}"
                )
                writer.add_scalar(
                    "charts/dynamic_model_accurate",
                    float(dynamic_model_accurate),
                    global_step,
                )
            # Removed Reward Model Training Block
            print(f"--- Model Check/Training Finished ---")
            model_train_time = time.time() - model_train_start_time
            writer.add_scalar("perf/model_train_time", model_train_time, global_step)

        # --- Phase 3: Model Rollout Generation ---
        # (Currently disabled by high freq/zero starts)
        # ...

        # --- Agent Training (Value Network Only) ---
        if global_step > args.learning_starts:
            # <<< Change: Only gate on dynamics model accuracy >>>
            proceed_with_update = True  # Default to training
            if not dynamic_model_accurate:
                if global_step % 1000 == 0:
                    print(
                        f"Info: Proceeding with agent update step {global_step}, but dynamics model INACCURATE"
                    )
                # Decide whether to skip or proceed with potentially bad HJB term
                # proceed_with_update = False # Option: Skip update entirely
            # else: # Optionally log if accurate
            # if global_step % 1000 == 0: print(f"Info: Proceeding with agent update step {global_step}, dynamics model ACCURATE")

            if proceed_with_update:
                data = rb.sample(
                    args.batch_size, env=None
                )  # Samples Tensors: raw data + term-only dones
                # Convert raw Tensors -> NumPy -> Normalize -> Normalized Tensors
                obs_raw_np = data.observations.cpu().numpy()
                actions_np = data.actions.cpu().numpy()
                rewards_raw_np = data.rewards.cpu().numpy()
                dones_term_only_np = data.dones.cpu().numpy()
                obs_norm_np = norm_envs.normalize_obs(obs_raw_np)
                rewards_norm_np = norm_envs.normalize_reward(
                    rewards_raw_np.reshape(-1)
                ).reshape(-1, 1)
                mb_obs = torch.tensor(obs_norm_np, dtype=torch.float32).to(device)
                mb_actions = torch.tensor(actions_np, dtype=torch.float32).to(
                    device
                )  # Action from buffer
                mb_rewards = (
                    torch.tensor(rewards_norm_np, dtype=torch.float32)
                    .to(device)
                    .squeeze(-1)
                )  # Normalized reward [batch]
                mb_dones = (
                    torch.tensor(dones_term_only_np, dtype=torch.float32)
                    .to(device)
                    .squeeze(-1)
                )  # Term-only dones [batch]

                # --- Critic Update ---
                terminations_mask = mb_dones.bool()
                non_terminations_mask = ~terminations_mask
                mb_obs_critic = mb_obs.clone().requires_grad_(
                    True
                )  # Need grad for dVdx
                all_current_v = critic(mb_obs_critic)

                # A. Terminal State Loss: V(s_term) = 0
                v_term = all_current_v[terminations_mask]
                terminal_critic_loss = torch.tensor(0.0, device=device)
                if v_term.numel() > 0:
                    terminal_critic_loss = F.mse_loss(v_term, torch.zeros_like(v_term))

                # B. Non-Terminal State Loss (HJB)
                hjb_loss_non_term = torch.tensor(0.0, device=device)
                if (
                    non_terminations_mask.any()
                    and args.use_hjb_loss
                    and compute_value_grad_func is not None
                ):
                    obs_non_term = mb_obs_critic[non_terminations_mask]
                    v_non_term = all_current_v[non_terminations_mask]
                    actions_buffer_non_term = mb_actions[
                        non_terminations_mask
                    ]  # Action actually taken
                    rewards_buffer_non_term = mb_rewards[
                        non_terminations_mask
                    ]  # Reward actually received (normalized)

                    # Calculate dV/dx, f1, f2, a*
                    try:
                        dVdx_non_term = vmap(compute_value_grad_func)(obs_non_term)
                        f1_non_term = get_f1(obs_non_term)
                        f2_T_non_term = get_f2_transpose(obs_non_term)
                        if f2_T_non_term is not None:
                            a_star_non_term = calculate_a_star(
                                dVdx_non_term, f2_T_non_term
                            )
                            # Calculate HJB residual: ( (-r_buffer - C*||a_buffer||^2) + <dV/dx, f1> - C*||a*||^2 ) - rho*V
                            # Note: r_buffer is normalized reward, C*||a||^2 is quadratic cost in raw action space? Needs consistent normalization.
                            # Let's use the HJB form: rho*V ≈ <dV/dx, f1> - l_state - 0.5 ||a*||^2_R
                            # Where r_state = -l_state. So rho*V ≈ <dV/dx, f1> + r_state - C ||a*||^2 (where C=ctrl_cost_weight)
                            # Estimate r_state ≈ r_buffer_norm + C * ||a_buffer||^2
                            r_state_approx = rewards_buffer_non_term + ctrl_cost_weight * torch.sum(
                                actions_buffer_non_term**2, dim=1
                            )  # Assumes a_buffer is raw action scale? No, should use normalized? Let's assume quadratic cost applies to raw actions. Need raw actions here.
                            # This is getting complex with normalization. Let's revert to simpler HJB residual form used before:
                            # residual = (r_buffer_norm + <dVdx, f>) - rho*V, where f = f(s, a_buffer)
                            # This ignores the analytical action/cost structure slightly but uses buffer data directly.
                            with torch.no_grad():
                                f_buffer_action = dynamic_model.ode_func(
                                    0, obs_non_term, actions_buffer_non_term
                                )
                            hjb_residual = (
                                rewards_buffer_non_term
                                + torch.einsum(
                                    "bi,bi->b", dVdx_non_term, f_buffer_action
                                )
                            ) - rho * v_non_term
                            hjb_loss_non_term = 0.5 * (hjb_residual**2).mean()

                        else:
                            print("WARN: HJB skipped due to f2 calculation failure.")
                    except Exception as e:
                        print(f"HJB Error:{e}")
                        hjb_loss_non_term = torch.tensor(0.0, device=device)

                # C. Total Critic Loss & Update
                critic_loss = (
                    args.hjb_coef * hjb_loss_non_term
                    + args.terminal_coeff * terminal_critic_loss
                )
                critic_optimizer.zero_grad()
                critic_loss.backward()
                if args.grad_norm_clip is not None:
                    nn.utils.clip_grad_norm_(critic.parameters(), args.grad_norm_clip)
                critic_optimizer.step()

                # Logging Critic losses
                writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
                writer.add_scalar(
                    "losses/critic_terminal", terminal_critic_loss.item(), global_step
                )
                writer.add_scalar(
                    "losses/critic_hjb_non_term", hjb_loss_non_term.item(), global_step
                )
                writer.add_scalar(
                    "metrics/critic_value_mean",
                    all_current_v.mean().item(),
                    global_step,
                )

        # Log SPS occasionally
        if global_step > 0 and global_step % 1000 == 0:
            sps = int(global_step / (time.time() - start_time))
            print(f"GStep: {global_step}, SPS: {sps}")
            writer.add_scalar("charts/SPS", sps, global_step)

    # --- End of Training Loop ---
    # --- Saving & Evaluation ---
    critic_final = critic  # Save final critic
    if args.save_model:
        run_folder = f"runs/{run_name}"
        os.makedirs(run_folder, exist_ok=True)
    critic_model_path = f"{run_folder}/{args.exp_name}_critic.cleanrl_model"
    torch.save(critic_final.state_dict(), critic_model_path)
    print(f"Critic saved: {critic_model_path}")
    dynamics_ode_path = f"{run_folder}/{args.exp_name}_dynamics_odefunc.cleanrl_model"
    torch.save(dynamic_model.ode_func.state_dict(), dynamics_ode_path)
    print(f"Dynamics ODEFunc saved: {dynamics_ode_path}")
    # Removed reward model saving
    norm_stats_path = f"{run_folder}/{args.exp_name}_vecnormalize.pkl"
    norm_envs.save(norm_stats_path)
    print(f"Normalization stats saved: {norm_stats_path}")
    if args.save_model:
        print("\nEvaluating agent performance...")
        eval_episodes = 10
        eval_seeds = range(args.seed + 100, args.seed + 100 + eval_episodes)
        eval_returns_raw = []
        # Evaluation needs policy derived from V and models
        eval_critic = ValueNetwork(norm_envs).to(device)
        eval_critic.load_state_dict(torch.load(critic_model_path, map_location=device))
        eval_critic.eval()
        eval_dynamic_model = DynamicModel(obs_dim, action_dim, args.env_dt, device).to(
            device
        )
        eval_dynamic_model.ode_func.load_state_dict(
            torch.load(dynamics_ode_path, map_location=device)
        )
        eval_dynamic_model.eval()
        # Need grad function for eval critic
        eval_compute_value_grad_func = None
        eval_get_f2_transpose = None
        eval_calculate_a_star = None
        if TORCH_FUNC_AVAILABLE:
            try:

                def eval_compute_scalar_value_critic(single_obs_tensor):
                    if single_obs_tensor.dim() == 1:
                        single_obs_tensor = single_obs_tensor.unsqueeze(0)
                    return eval_critic(single_obs_tensor).squeeze()

                eval_compute_value_grad_func = grad(eval_compute_scalar_value_critic)

                def eval_func_for_jac(t, s, a):
                    return eval_dynamic_model.ode_func(t, s, a)

                eval_compute_jac_f_a = jacrev(eval_func_for_jac, argnums=2)

                def eval_compute_jac_for_single_s(s_single):
                    s_batch = s_single.unsqueeze(0)
                    a_zeros_single = torch.zeros(1, action_dim, device=s_single.device)
                    jacobian_matrix = eval_compute_jac_f_a(
                        torch.tensor(0.0), s_batch, a_zeros_single
                    )
                    return jacobian_matrix.squeeze(0)

                def eval_get_f2_transpose(s_norm_batch):
                    return torch.permute(
                        vmap(eval_compute_jac_for_single_s)(s_norm_batch), (0, 2, 1)
                    )

                def eval_calculate_a_star(dVdx_norm, f2_transpose):
                    if f2_transpose is None:
                        return None
                    dVdx_col = dVdx_norm.unsqueeze(-1)
                    a_star = (-1.0 / (2.0 * ctrl_cost_weight)) * torch.bmm(
                        f2_transpose, dVdx_col
                    ).squeeze(-1)
                    return a_star

            except Exception as e:
                print(f"Eval grad/jac func failed: {e}")

        for seed in eval_seeds:
            eval_envs_base = DummyVecEnv(
                [make_env(args.env_id, seed, False, f"{run_name}-eval-seed{seed}")]
            )
            eval_norm_envs = VecNormalize.load(norm_stats_path, eval_envs_base)
            eval_norm_envs.training = False
            eval_norm_envs.norm_reward = False
            obs_norm_np = eval_norm_envs.reset(seed=seed)
            done = False
            episode_return_raw = 0
            num_steps = 0
            max_steps = 1000
            while not done and num_steps < max_steps:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs_norm_np, dtype=torch.float32).to(
                        device
                    )
                    action = np.zeros((args.num_envs, action_dim))  # Default action
                    if (
                        eval_compute_value_grad_func is not None
                        and eval_get_f2_transpose is not None
                        and eval_calculate_a_star is not None
                    ):
                        try:
                            obs_tensor_grad = obs_tensor.clone().requires_grad_(
                                True
                            )  # Need requires_grad? No, grad func handles it.
                            dVdx = vmap(eval_compute_value_grad_func)(obs_tensor)
                            f2_T = eval_get_f2_transpose(obs_tensor)
                            if f2_T is not None:
                                action_star = eval_calculate_a_star(dVdx, f2_T)
                                if action_star is not None:
                                    action = (
                                        action_star.cpu()
                                        .numpy()
                                        .clip(action_space.low, action_space.high)
                                    )
                        except Exception as e:
                            print(
                                f"Eval action calc failed: {e}"
                            )  # Use default random if fails
                    else:  # Use random action if grad funcs failed
                        action = np.array(
                            [
                                eval_norm_envs.action_space.sample()
                                for _ in range(eval_norm_envs.num_envs)
                            ]
                        )

                obs_norm_np, reward_raw_step, term, trunc, info = eval_norm_envs.step(
                    action
                )
                done = term[0] or trunc[0]
                episode_return_raw += reward_raw_step[0]
                num_steps += 1
            eval_returns_raw.append(episode_return_raw)
            print(
                f"  Eval Seed {seed}: Raw Episodic Return={episode_return_raw:.2f} ({num_steps} steps)"
            )
            eval_envs_base.close()
        mean_eval_return_raw = np.mean(eval_returns_raw)
        std_eval_return_raw = np.std(eval_returns_raw)
        print(
            f"Evaluation complete. Avg Return: {mean_eval_return_raw:.2f} +/- {std_eval_return_raw:.2f}"
        )
        for idx, r in enumerate(eval_returns_raw):
            writer.add_scalar("eval/raw_episodic_return", r, idx)
        if args.upload_model:
            print("Uploading models to Hugging Face Hub...")
            # ... (HF Upload logic) ...

    # --- Cleanup ---
    norm_envs.close()
    writer.close()
    print("\nTraining finished.")
