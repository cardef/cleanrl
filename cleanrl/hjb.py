# HJB Value Iteration with Learned Control-Affine Dynamics (Neural ODE)
# Assumes Quadratic Action Cost (Inferred C), No Reward Model, No Actor
# USES RAW (UNNORMALIZED) OBSERVATIONS AND REWARDS
# Critic Loss uses HJB residual with l(a*) inferred from buffer reward/action

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
from stable_baselines3.common.vec_env import (
    VecEnv,
    VecNormalize,
    DummyVecEnv,
)  # Still use VecNormalize, but disable norm

try:
    from torch.func import grad, vmap, jacrev

    print("Imported grad, vmap, jacrev from torch.func")
    TORCH_FUNC_AVAILABLE = True
except ImportError:
    try:
        from functorch import grad, vmap, jacrev

        print("Imported grad, vmap, jacrev from functorch")
        TORCH_FUNC_AVAILABLE = True
    except ImportError:
        print(
            "WARNING: torch.func / functorch required for HJB gradients/jacobians not available."
        )
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

warnings.filterwarnings(
    "ignore", category=UserWarning, module="stable_baselines3.common.buffers"
)


@dataclass
class Args:
    exp_name: str = (
        os.path.basename(__file__)[: -len(".py")] + "_hjb_vi_ode_raw_user_lstar"
    )  # Updated name
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
    env_id: str = "InvertedPendulum-v4"
    total_timesteps: int = 1000000
    learning_rate: float = 1e-4  # LR for Critic (V function)
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    batch_size: int = 256  # Agent update batch size
    learning_starts: int = 5000
    exploration_noise_std: float = 0.1
    grad_norm_clip: Optional[float] = 1.0

    # Model Training Args
    model_train_freq: int = 250
    model_dataset_size: int = 50_000
    dynamics_learning_rate: float = 1e-3
    dynamic_train_threshold: float = 0.01  # Threshold now applies to raw state MSE
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
    hjb_coef: float = 1.0
    use_hjb_loss: bool = True
    terminal_coeff: float = 1.0
    ctrl_cost_weight: Optional[float] = None  # Try to infer this
    """Weight C for the quadratic control cost term C*||a||^2 used ONLY for deriving a*."""

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
# Operates on RAW obs/action, predicts RAW next obs derivative
class ODEFunc(nn.Module):  # (Same)
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        shared_layer_size = 256
        f1_layer_size = 128
        f2_layer_size = 128
        self.shared_layers = nn.Sequential(
            layer_init(nn.Linear(obs_dim, shared_layer_size)),
            nn.SiLU(),
            layer_init(nn.Linear(shared_layer_size, shared_layer_size)),
            nn.SiLU(),
        )
        self.f1_head = nn.Sequential(
            layer_init(nn.Linear(shared_layer_size, f1_layer_size)),
            nn.SiLU(),
            layer_init(nn.Linear(f1_layer_size, obs_dim)),
        )
        self.f2_head = nn.Sequential(
            layer_init(nn.Linear(shared_layer_size, f2_layer_size)),
            nn.SiLU(),
            layer_init(nn.Linear(f2_layer_size, obs_dim * action_dim)),
        )
        print(
            f"Initialized Control-Affine ODEFunc (RAW): Shared={shared_layer_size}, f1=[{obs_dim}], f2=[{obs_dim}x{action_dim}]"
        )

    def get_f1_f2(self, x_raw):
        shared_features = self.shared_layers(x_raw.float())
        f1 = self.f1_head(shared_features)
        f2_flat = self.f2_head(shared_features)
        f2 = f2_flat.view(-1, self.obs_dim, self.action_dim)
        return f1, f2

    def forward(self, t, x_raw, a_raw):
        f1, f2 = self.get_f1_f2(x_raw)
        control_effect = torch.bmm(f2, a_raw.float().unsqueeze(-1)).squeeze(-1)
        dx_dt = f1 + control_effect
        return dx_dt


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
        # Adjoint is created but not used in the forward pass currently.
        self.adjoint = to.AutoDiffAdjoint(step_method=self.step_method, step_size_controller=self.step_size_controller)
        print(
            f"Initialized DynamicModel (RAW) using Control-Affine ODEFunc (Solver: Euler, dt={self.dt})"
        )

    def forward(self, initial_obs_raw, actions_raw):
        batch_size = initial_obs_raw.shape[0]
        # Use a small fraction of dt for dt0, ensuring it's a scalar tensor
        dt0 = torch.tensor(self.dt / 5.0, device=self.device)  # Scalar dt0
        t_span_tensor = torch.tensor([0.0, self.dt], device=self.device)
        t_eval = t_span_tensor.unsqueeze(0).repeat(batch_size, 1)
        # t_eval should be [t_start, t_end] for each batch element if needed, but odeint takes a single t_span
        problem = to.InitialValueProblem(y0=initial_obs_raw.float(), t_eval=t_eval) # t_eval not directly used by odeint here
        sol = self.adjoint.solve(problem, args=actions_raw.float(), dt0=dt0)
        final_state_pred_norm = sol.ys[:, 1, :]
        return final_state_pred_norm


# --- Agent Network Definitions ---
# Operates on RAW observations
class ValueNetwork(nn.Module):  # (Same)
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
        print("Initialized ValueNetwork (Critic) for RAW inputs.")

    def forward(self, x_raw):
        return self.net(x_raw.float()).squeeze(-1)  # Predicts raw value


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
):  # Operates on RAW data
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch_idx, batch_data in enumerate(train_loader):
        obs_raw, actions_raw, targets_raw = [d.to(device) for d in batch_data]
        if is_dynamic_model:
            preds_raw = model(obs_raw, actions_raw)
        else:
            raise ValueError("train_model_epoch called without dynamic model flag")
        loss = F.mse_loss(preds_raw, targets_raw)
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
):  # Operates on RAW data
    model.eval()
    all_preds_raw = []
    all_targets_raw = []
    with torch.no_grad():
        for batch_data in val_loader:
            obs_raw, actions_raw, targets_raw = [d.to(device) for d in batch_data]
            if is_dynamic_model:
                preds_raw = model(obs_raw, actions_raw)
            else:
                raise ValueError("validate_model called without dynamic model flag")
            all_preds_raw.append(preds_raw)
            all_targets_raw.append(targets_raw)
    if not all_preds_raw:
        return float("inf"), {
            "mse": float("inf"),
            "mae": float("inf"),
            "r2": -float("inf"),
        }
    all_preds_raw = torch.cat(all_preds_raw, dim=0)
    all_targets_raw = torch.cat(all_targets_raw, dim=0)
    val_metrics = calculate_metrics(all_preds_raw, all_targets_raw)
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
            args.track = False  # Disable tracking if wandb is not installed
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

    # --- Environment Setup without Normalization ---
    print("Setting up environment...")
    envs = DummyVecEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    # <<< Norm Change: Initialize VecNormalize but disable obs/reward norm >>>
    norm_envs = VecNormalize(envs, gamma=args.gamma, norm_obs=False, norm_reward=False)
    print("VecNormalize wrapper used, but Obs/Reward normalization DISABLED.")
    try:
        env_dt = norm_envs.get_attr("dt")[0]
        print(f"Detected env dt: {env_dt}")
    except Exception:
        print(f"Warning: Could not detect env dt. Using default: {args.env_dt}")
        env_dt = args.env_dt
    args.env_dt = env_dt
    # Infer ctrl_cost_weight
    try:
        ctrl_cost_weight = norm_envs.get_attr("_ctrl_cost_weight")[0]
        print(f"Detected env ctrl_cost_weight: {ctrl_cost_weight}")
    except Exception as e:
        ctrl_cost_weight = 0.1
        print(
            f"Warning: Could not detect env ctrl_cost_weight ({e}). Using default: {ctrl_cost_weight}"
        )
    if args.ctrl_cost_weight is not None:
        ctrl_cost_weight = args.ctrl_cost_weight
        print(f"Overriding ctrl_cost_weight with args: {ctrl_cost_weight}")
    if ctrl_cost_weight <= 0:
        raise ValueError("ctrl_cost_weight must be positive.")
    print(f"Using ctrl_cost_weight = {ctrl_cost_weight}")

    # <<< Norm Change: Use raw spaces for agent/model sizing >>>
    obs_space = norm_envs.observation_space
    action_space = norm_envs.action_space
    obs_dim = np.array(obs_space.shape).prod()
    action_dim = np.prod(action_space.shape)
    action_space_low_t = torch.tensor(
        action_space.low, dtype=torch.float32, device=device
    )
    action_space_high_t = torch.tensor(
        action_space.high, dtype=torch.float32, device=device
    )

    # --- Agent, Models, Optimizers ---
    # <<< Norm Change: Pass raw env spaces (from norm_envs as norm is off) >>>
    critic = ValueNetwork(norm_envs).to(device)
    critic_optimizer = optim.AdamW(critic.parameters(), lr=args.learning_rate)
    dynamic_model = DynamicModel(obs_dim, action_dim, args.env_dt, device).to(device)
    dynamics_optimizer = optim.AdamW(
        dynamic_model.ode_func.parameters(), lr=args.dynamics_learning_rate
    )

    # --- Replay Buffer for RAW Data ---
    raw_obs_space = (
        norm_envs.observation_space
    )  # Raw space is same as norm_envs space now
    raw_action_space = norm_envs.action_space
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
    obs = norm_envs.reset()  # Returns raw obs now
    obs = obs.astype(np.float32)
    dynamic_model_accurate = False
    global_step = 0

    # --- vmap/grad setup ---
    compute_value_grad_func = None
    if args.use_hjb_loss and TORCH_FUNC_AVAILABLE:
        try:
            # <<< Norm Change: Helper operates on raw obs >>>
            def compute_scalar_value_critic(single_obs_raw_tensor):
                if single_obs_raw_tensor.dim() == 1:
                    single_obs_raw_tensor = single_obs_raw_tensor.unsqueeze(0)
                return critic(single_obs_raw_tensor).squeeze()  # Pass raw obs to critic

            compute_value_grad_func = grad(compute_scalar_value_critic)
            print("Value gradient function for HJB created.")
        except Exception as e:
            print(f"WARNING: Failed grad func creation: {e}. HJB disabled.")
            args.use_hjb_loss = False
    elif args.use_hjb_loss:
        print("WARNING: HJB requested but torch.func unavailable. HJB disabled.")
        args.use_hjb_loss = False

    # --- Helper Functions for a_star ---
    def get_f1(s_raw_batch):  # Takes raw state
        with torch.no_grad():
            f1_pred, _ = dynamic_model.ode_func.get_f1_f2(s_raw_batch)  # Get raw f1
        return f1_pred

    def get_f2_transpose(s_raw_batch):  # Takes raw state
        with torch.no_grad():
            _, f2 = dynamic_model.ode_func.get_f1_f2(s_raw_batch)  # Raw f2 [b, o, a]
        f2_transpose = torch.permute(f2, (0, 2, 1))  # Shape [b, a, o]
        return f2_transpose

    def calculate_a_star(
        dVdx_raw, f2_transpose
    ):  # Takes raw dV/dx, raw f2_T -> returns raw a*
        if dVdx_raw is None or f2_transpose is None:
            return None
        dVdx_col = dVdx_raw.unsqueeze(-1)  # [b, o, 1]
        try:
            # a* = -1/(2*C) * f2^T * dVdx^T
            a_star_unclamped = (-1.0 / (2.0 * ctrl_cost_weight)) * torch.bmm(
                f2_transpose, dVdx_col
            ).squeeze(-1)
            return a_star_unclamped
        except Exception as e:
            print(f"ERROR calculating a_star: {e}")
            return None

    # ========================================================================
    # <<< Main Training Loop >>>
    # ========================================================================
    print(f"Starting training loop for {args.total_timesteps} timesteps...")
    for global_step in range(args.total_timesteps):
        iter_start_time = time.time()

        # --- Action Selection & Environment Interaction ---
        if global_step < args.learning_starts:
            actions = np.array(
                [action_space.sample() for _ in range(args.num_envs)]
            )  # Use raw action space
        else:
            with torch.no_grad():
                obs_tensor = torch.Tensor(obs).to(device)  # Raw obs tensor
                actions_star = torch.zeros(args.num_envs, action_dim).to(
                    device
                )  # Default
                if compute_value_grad_func is not None:
                    try:
                        dVdx = vmap(compute_value_grad_func)(
                            obs_tensor
                        )  # Grad of raw V w.r.t raw obs
                        f2_T = get_f2_transpose(obs_tensor)  # f2(s_raw)^T
                        if f2_T is not None:
                            actions_star_unclamped = calculate_a_star(
                                dVdx, f2_T
                            )  # Calculate raw a*
                            if actions_star_unclamped is not None:
                                actions_star = actions_star_unclamped
                    except Exception as e:
                        if global_step % 1000 == 0:
                            print(f"WARN: a* calc failed in action selection: {e}")
                        actions_star = (
                            torch.Tensor(action_space.sample()).to(device).unsqueeze(0)
                        )  # Sample raw action

                noise = torch.normal(
                    0,
                    args.exploration_noise_std,
                    size=actions_star.shape,
                    device=device,
                )
                actions_noisy = actions_star + noise
                actions_clipped = torch.max(
                    torch.min(actions_noisy, action_space_high_t), action_space_low_t
                )
                actions = actions_clipped.cpu().numpy()

        # <<< Norm Change: Step returns RAW obs/reward >>>
        next_obs, rewards_raw, dones_combined_np, infos = norm_envs.step(
            actions
        )  # Reward is RAW
        next_obs = next_obs.astype(np.float32)
        rewards_raw = rewards_raw.astype(np.float32)
        terminations = np.array(
            [
                infos[i].get("TimeLimit.truncated", False) == False
                and dones_combined_np[i]
                for i in range(args.num_envs)
            ]
        )

        # Log real env returns (raw)
        if "final_info" in infos:
            final_infos = infos["final_info"]
        else:
            final_infos = [i for i in infos if i is not None]
        for info in final_infos:
            if info and "episode" in info:
                episode_info = info["episode"]
                # Use .item() to extract scalar value for printing and logging
                print(
                    f"GStep={global_step}, EpReturn={episode_info['r'].item():.2f}, EpLen={episode_info['l'].item()}"
                )
                writer.add_scalar(
                    "charts/episodic_return", episode_info["r"].item(), global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", episode_info["l"].item(), global_step
                )
                break

        # Store RAW data in replay buffer
        real_next_obs_raw = next_obs.copy()
        # Start with the raw next_obs from step
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
            obs, real_next_obs_raw, actions, rewards_raw, dones_combined_np, infos
        )  # Store RAW obs, RAW reward

        obs = next_obs  # Update agent state to RAW next obs

        # --- Model Training/Validation (Periodic) ---
        if (
            global_step > args.learning_starts
            and global_step % args.model_train_freq == 0
        ):
            print(f"\n--- GStep {global_step}: Checking/Training Dynamics Model ---")
            model_train_start_time = time.time()
            buffer_data_raw_tensors = rb.sample(
                args.model_dataset_size, env=None
            )  # Samples RAW tensors
            # <<< Norm Change: Use RAW tensors directly for training/validation >>>
            obs_raw_t = buffer_data_raw_tensors.observations.to(device)
            next_obs_raw_t = buffer_data_raw_tensors.next_observations.to(device)
            actions_t = buffer_data_raw_tensors.actions.to(device)
            dones_term_only_t = buffer_data_raw_tensors.dones.to(
                device
            )  # Term-only dones

            # Train/Validate Dynamics Model only (using RAW data)
            non_terminal_mask_dyn = dones_term_only_t.squeeze(-1) == 0
            dyn_obs_t = obs_raw_t[non_terminal_mask_dyn]
            dyn_acts_t = actions_t[non_terminal_mask_dyn]
            dyn_targets_t = next_obs_raw_t[non_terminal_mask_dyn]
            # Target is raw next_obs
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
                # Threshold might need adjustment for raw MSE
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
            print(f"--- Model Check/Training Finished ---")
            model_train_time = time.time() - model_train_start_time
            writer.add_scalar("perf/model_train_time", model_train_time, global_step)

        # --- Phase 3: Model Rollout Generation ---
        # (Disabled)

        # --- Agent Training (Value Network Only) ---
        if global_step > args.learning_starts:
            proceed_with_update = True  # Removed gating
            if not dynamic_model_accurate:  # Only check dynamics accuracy now
                if global_step % 1000 == 0:
                    print(
                        f"Info: Proceeding with agent update step {global_step}, but dynamics model INACCURATE"
                    )

            if proceed_with_update:
                data = rb.sample(
                    args.batch_size, env=None
                )  # Samples Tensors: raw data + term-only dones
                # <<< Norm Change: Use RAW tensors directly >>>
                mb_obs = data.observations.to(device)
                mb_actions = data.actions.to(device)  # Raw actions from buffer
                mb_rewards_raw = data.rewards.to(device).squeeze(
                    -1
                )  # Use RAW reward [batch]
                # mb_next_obs = data.next_observations.to(device) # Raw next obs (needed for TD target if used)
                mb_dones = data.dones.to(device).squeeze(-1)  # Term-only dones [batch]

                # --- Critic Update ---
                terminations_mask = mb_dones.bool()
                non_terminations_mask = ~terminations_mask
                mb_obs_critic = mb_obs.clone().requires_grad_(True)
                all_current_v = critic(mb_obs_critic)  # V is value of raw state

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
                    actions_buffer_raw_non_term = mb_actions[
                        non_terminations_mask
                    ]  # Raw actions from buffer
                    rewards_buffer_raw_non_term = mb_rewards_raw[
                        non_terminations_mask
                    ]  # Raw rewards from buffer

                    # Calculate dV/dx (raw), f1 (raw), f2 (raw), a* (raw)
                    try:
                        dVdx_raw = vmap(compute_value_grad_func)(
                            obs_non_term
                        )  # Grad w.r.t. s_raw
                        with torch.no_grad():  # f1, f2 don't need grads for critic loss
                            f1_raw, f2_raw = dynamic_model.ode_func.get_f1_f2(
                                obs_non_term
                            )  # Operate on raw state
                            f2_T_raw = torch.permute(f2_raw, (0, 2, 1))

                        if f2_T_raw is not None:
                            # Calculate UNCLAMPED optimal action a*(s_raw)
                            a_star_non_term = calculate_a_star(
                                dVdx_raw, f2_T_raw
                            )  # Use helper

                            if a_star_non_term is not None:
                                # <<< Change: Calculate HJB residual using analytical form with raw values >>>
                                # residual = ( <dV/dx, f1> - C*||a*||^2 - l_state ) - rho*V
                                # Estimate raw l_state = -r_state ≈ -(r_raw + C*||a_buffer||^2)
                                l_state_raw_approx = (
                                    -rewards_buffer_raw_non_term
                                    - ctrl_cost_weight
                                    * torch.sum(actions_buffer_raw_non_term**2, dim=1)
                                )  # Uses raw r and raw a

                                # Calculate quadratic cost term C*||a*||^2 (using raw a*)
                                a_star_cost_term = ctrl_cost_weight * torch.sum(
                                    a_star_non_term**2, dim=1
                                )

                                # Calculate <dVdx, f1> term (all raw)
                                dvdx_f1_term = torch.einsum(
                                    "bi,bi->b", dVdx_raw, f1_raw
                                )

                                # HJB residual based on l_state estimate (all raw)
                                # Note the sign: HJB is rho*V = l_state + <dV/dx, f1> - C*||a*||^2
                                # Residual = ( l_state_raw + dvdx_f1_term - a_star_cost_term ) - rho * v_non_term
                                hjb_residual = (
                                    l_state_raw_approx + dvdx_f1_term - a_star_cost_term
                                ) - rho * v_non_term

                                hjb_loss_non_term = 0.5 * (hjb_residual**2).mean()
                            else:
                                print(
                                    "WARN: HJB skipped due to a* calculation failure."
                                )
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
    critic_final = critic
    if args.save_model:
        run_folder = f"runs/{run_name}"
        os.makedirs(run_folder, exist_ok=True)
    critic_model_path = f"{run_folder}/{args.exp_name}_critic.cleanrl_model"
    torch.save(critic_final.state_dict(), critic_model_path)
    print(f"Critic saved: {critic_model_path}")
    dynamics_ode_path = f"{run_folder}/{args.exp_name}_dynamics_odefunc.cleanrl_model"
    torch.save(dynamic_model.ode_func.state_dict(), dynamics_ode_path)
    print(f"Dynamics ODEFunc saved: {dynamics_ode_path}")
    # <<< Norm Change: No VecNormalize stats to save >>>
    # norm_stats_path = f"{run_folder}/{args.exp_name}_vecnormalize.pkl"; norm_envs.save(norm_stats_path); print(f"Normalization stats saved: {norm_stats_path}");
    if args.save_model:
        print("\nEvaluating agent performance...")
        eval_episodes = 10
        eval_seeds = range(args.seed + 100, args.seed + 100 + eval_episodes)
        eval_returns_raw = []
        # <<< Norm Change: Evaluation uses raw env >>>
        eval_critic = ValueNetwork(envs).to(device)
        # Pass raw env space info
        eval_critic.load_state_dict(torch.load(critic_model_path, map_location=device))
        eval_critic.eval()
        eval_dynamic_model = DynamicModel(obs_dim, action_dim, args.env_dt, device).to(
            device
        )
        eval_dynamic_model.ode_func.load_state_dict(
            torch.load(dynamics_ode_path, map_location=device)
        )
        eval_dynamic_model.eval()
        # Need grad function for eval critic (operating on raw data)
        eval_compute_value_grad_func = None
        eval_get_f2_transpose = None
        eval_calculate_a_star = None
        if TORCH_FUNC_AVAILABLE:
            try:

                def eval_compute_scalar_value_critic(s):  # Takes raw state
                    if s.dim() == 1:
                        s = s.unsqueeze(0)
                        return eval_critic(s).squeeze()

                eval_compute_value_grad_func = grad(eval_compute_scalar_value_critic)

                def eval_get_f2_transpose(s_raw_batch):  # Takes raw state
                    with torch.no_grad():
                        _, f2_eval = eval_dynamic_model.ode_func.get_f1_f2(s_raw_batch)
                    return torch.permute(f2_eval, (0, 2, 1))

                def eval_calculate_a_star(dVdx_raw, f2_transpose):  # Takes raw gradient
                    if f2_transpose is None:
                        return None
                    dVdx_col = dVdx_raw.unsqueeze(-1)
                    a_star_unclamped = (-1.0 / (2.0 * ctrl_cost_weight)) * torch.bmm(
                        f2_transpose, dVdx_col
                    ).squeeze(-1)
                    return a_star_unclamped  # Return unclamped

            except Exception as e:
                print(f"Eval grad/jac func setup failed: {e}")

        for seed in eval_seeds:
            # <<< Norm Change: Use raw env for eval >>>
            eval_env = DummyVecEnv(
                [make_env(args.env_id, seed, False, f"{run_name}-eval-seed{seed}")]
            )
            obs_raw_np, _ = eval_env.reset(seed=seed)
            # Get raw obs
            obs_raw_np = obs_raw_np.astype(np.float32)
            done = False
            episode_return_raw = 0
            num_steps = 0
            max_steps = 1000
            while not done and num_steps < max_steps:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs_raw_np, dtype=torch.float32).to(
                        device
                    )  # Use raw obs tensor
                    action = np.zeros((args.num_envs, action_dim))  # Default
                    if (
                        eval_compute_value_grad_func is not None
                        and eval_get_f2_transpose is not None
                        and eval_calculate_a_star is not None
                    ):
                        try:
                            dVdx = vmap(eval_compute_value_grad_func)(
                                obs_tensor
                            )  # Grad w.r.t raw obs
                            f2_T = eval_get_f2_transpose(obs_tensor)  # f2(s_raw)^T
                            if f2_T is not None:
                                action_star_unclamped = eval_calculate_a_star(
                                    dVdx, f2_T
                                )
                                if action_star_unclamped is not None:
                                    # Clamp final action for environment step
                                    action_star_clamped = torch.max(
                                        torch.min(
                                            action_star_unclamped, action_space_high_t
                                        ),
                                        action_space_low_t,
                                    )
                                    action = action_star_clamped.cpu().numpy()
                                else:
                                    action = np.array(
                                        [
                                            eval_env.action_space.sample()
                                            for _ in range(eval_env.num_envs)
                                        ]
                                    )
                            else:
                                action = np.array(
                                    [
                                        eval_env.action_space.sample()
                                        for _ in range(eval_env.num_envs)
                                    ]
                                )
                        except Exception as e:
                            print(f"Eval action calc failed: {e}")
                            action = np.array(
                                [
                                    eval_env.action_space.sample()
                                    for _ in range(eval_env.num_envs)
                                ]
                            )
                    else:
                        action = np.array(
                            [
                                eval_env.action_space.sample()
                                for _ in range(eval_env.num_envs)
                            ]
                        )

                # <<< Norm Change: Step raw env >>>
                obs_raw_np, reward_raw_step, term, trunc, info = eval_env.step(
                    action
                )  # Get raw obs, raw reward
                obs_raw_np = obs_raw_np.astype(np.float32)
                done = term[0] or trunc[0]
                episode_return_raw += reward_raw_step[0]
                num_steps += 1
            eval_returns_raw.append(episode_return_raw)
            print(
                f"  Eval Seed {seed}: Raw Episodic Return={episode_return_raw:.2f} ({num_steps} steps)"
            )
            eval_env.close()
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
