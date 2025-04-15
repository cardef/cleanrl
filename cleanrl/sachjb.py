# Full HJB Actor-Critic Code with Stochastic Policy, TorchODE, Models,
# Validation, Early Stopping, HJB (vmap/grad), R2 Logging, VecNormalize
# --- Final Version with ALL Syntax Fixes v4 (Logging Format Fix) ---

import os
import random
import time
import math
import copy  # For deepcopy in early stopping
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
    from torch.func import grad, vmap

    print("Imported grad, vmap from torch.func")
    TORCH_FUNC_AVAILABLE = True
except ImportError:
    try:  # Fallback for older PyTorch with functorch
        from functorch import grad, vmap

        print("Imported grad, vmap from functorch")
        TORCH_FUNC_AVAILABLE = True
    except ImportError:
        print(
            "WARNING: torch.func / functorch not available. HJB residual calculation will be skipped."
        )
        TORCH_FUNC_AVAILABLE = False
try:
    import torchode as to

    print("Imported torchode.")
    TORCHODE_AVAILABLE = True
except ImportError:
    print(
        "FATAL: torchode not found (`pip install torchode`). Neural ODE Dynamics Model cannot be used."
    )
    TORCHODE_AVAILABLE = False
    exit()  # Exit if torchode is required but not found

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="stable_baselines3.common.buffers"
)


@dataclass
class Args:
    # Args definition remains the same...
    exp_name: str = os.path.basename(__file__)[: -len(".py")] + "_hjb_ac_ode_norm_final"
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
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    batch_size: int = 256
    learning_starts: int = 5000
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    grad_norm_clip: Optional[float] = 1.0
    alpha: float = 0.01
    autotune: bool = True
    model_train_freq: int = 1000
    model_dataset_size: int = 50_000
    dynamics_learning_rate: float = 1e-3
    reward_learning_rate: float = 1e-3
    dynamic_train_threshold: float = 0.05
    reward_train_threshold: float = 0.05
    model_val_ratio: float = 0.2
    model_val_patience: int = 10
    model_val_delta: float = 1e-5
    model_max_epochs: int = 50
    model_train_batch_size: int = 256
    model_validation_freq: int = 5
    model_updates_per_epoch: int = 1
    model_rollout_freq: int = 250
    model_rollout_length: int = 1
    num_model_rollout_starts: int = 4096
    hjb_coef: float = 0.1
    use_hjb_loss: bool = True
    viscosity_coeff: float = 0.0
    terminal_coeff: float = 1.0
    num_envs: int = 1
    env_dt: float = 0.05
    minibatch_size: int = field(init=False)
    rho: float = field(init=False)


# --- Environment Creation ---
def make_env(env_id, seed, idx, capture_video, run_name):
    """Creates a base environment instance."""

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
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initializes weights orthogonally and biases to a constant."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# --- Dynamics Model (Neural ODE using TorchODE) ---
class ODEFunc(nn.Module):
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
        print(f"Initialized ODEFunc: Input {obs_dim + action_dim}, Output {obs_dim}")

    def forward(self, t, x_norm, a):
        return self.net(torch.cat([x_norm.float(), a.float()], dim=-1))


class DynamicModel(nn.Module):
    def __init__(self, obs_dim, action_dim, dt: float, device: torch.device):
        super().__init__()
        if not TORCHODE_AVAILABLE:
            raise ImportError("torchode not found.")
        self.ode_func = ODEFunc(obs_dim, action_dim)
        self.dt = dt
        self.device = device
        self.term = to.ODETerm(self.ode_func, with_args=True)
        self.step_method = to.Euler(term=self.term)
        self.step_size_controller = to.FixedStepController()
        self.adjoint = to.AutoDiffAdjoint(
            step_method=self.step_method, step_size_controller=self.step_size_controller
        )
        print(f"Initialized DynamicModel using TorchODE (Solver: Euler, dt={self.dt})")

    def forward(self, initial_obs_norm, actions_norm):
        batch_size = initial_obs_norm.shape[0]
        dt0 = torch.full((batch_size,), self.dt / 5, device=self.device)
        t_span_tensor = torch.tensor([0.0, self.dt], device=self.device)
        t_eval = t_span_tensor.unsqueeze(0).repeat(batch_size, 1)
        problem = to.InitialValueProblem(
            y0=initial_obs_norm.float(),
            t_eval=t_eval,
        )
        sol = self.adjoint.solve(problem, args=actions_norm.float(), dt0=dt0)
        final_state_pred_norm = sol.ys[:, 1, :]
        return final_state_pred_norm


# --- Reward Model ---
class RewardModel(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        hidden_size = 128
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + action_dim, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1)),
        )
        print(f"Initialized RewardModel: Input {obs_dim+action_dim}, Output 1")

    def forward(self, obs_norm, action):
        return self.net(torch.cat([obs_norm.float(), action.float()], dim=1)).squeeze(
            -1
        )


# --- Agent Network Definitions ---
class HJBCritic(nn.Module):  # Single Critic
    def __init__(self, env: VecEnv):
        super().__init__()
        obs_dim = np.array(env.observation_space.shape).prod()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )
        print("Initialized Single HJBCritic Network.")

    def forward(self, x_norm):
        return self.critic(x_norm).squeeze(-1)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):  # Stochastic Actor
    def __init__(self, env: VecEnv):
        super().__init__()
        obs_dim = np.array(env.observation_space.shape).prod()
        action_dim = np.prod(env.action_space.shape)
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        action_low = torch.tensor(env.action_space.low, dtype=torch.float32)
        action_high = torch.tensor(env.action_space.high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)
        print("Initialized SAC-style Stochastic Actor.")

    def forward(self, x_norm):
        x = F.relu(self.fc1(x_norm))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x_norm, deterministic=False):
        mean, log_std = self(x_norm)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        if deterministic:
            x_t = mean
        else:
            x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


# --- Utility Functions ---
def calculate_metrics(preds, targets):
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
):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch_idx, batch_data in enumerate(train_loader):
        obs_norm, actions, targets_norm = [d.to(device) for d in batch_data]
        if is_dynamic_model:
            preds_norm = model(obs_norm, actions)
            # <<< Debug Prints for Dynamics Model >>>
            if batch_idx == 0 and epoch % args.model_validation_freq == 0: # Print only occasionally
                print(f"\n--- Debug Dyn Train Batch (Epoch {epoch}, Batch {batch_idx}) ---")
                print(f"Input Obs Norm sample (first 5): {obs_norm[0, :5].detach().cpu().numpy()}")
                print(f"Target Next Obs Norm sample (first 5): {targets_norm[0, :5].detach().cpu().numpy()}")
                print(f"Predicted Next Obs Norm sample (first 5): {preds_norm[0, :5].detach().cpu().numpy()}")
                diff = torch.abs(preds_norm - targets_norm)
                print(f"Abs Diff Stats: Min={diff.min():.4e}, Max={diff.max():.4e}, Mean={diff.mean():.4e}")
                # Check ODEFunc output directly if possible (might require modifying DynamicModel)
                # with torch.no_grad():
                #    f_pred = model.ode_func(0, obs_norm, actions)
                #    print(f"ODEFunc Output 'f' Stats: Min={f_pred.min():.4e}, Max={f_pred.max():.4e}, Mean={f_pred.mean():.4e}")
                print("--- End Debug Dyn Train Batch ---")
        else:
            preds_norm = model(obs_norm, actions)
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
):
    model.eval()
    all_preds_norm = []
    all_targets_norm = []
    with torch.no_grad():
        for batch_data in val_loader:
            obs_norm, actions, targets_norm = [d.to(device) for d in batch_data]
            if is_dynamic_model:
                preds_norm = model(obs_norm, actions)
            else:
                preds_norm = model(obs_norm, actions)
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
    args = tyro.cli(Args)

    # Calculate dependent args
    if args.model_rollout_length > 0:
        args.num_model_rollout_starts = args.num_model_rollout_starts
        print(
            f"Generating {args.num_model_rollout_starts} rollouts of length {args.model_rollout_length}"
        )
    else:
        args.num_model_rollout_starts = 0
        print(
            "Warning: model_rollout_length <= 0. No model rollouts will be generated."
        )
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
        norm_obs=False,
        norm_reward=True,
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
    obs_space = norm_envs.observation_space
    action_space = norm_envs.action_space
    obs_dim = np.array(obs_space.shape).prod()
    action_dim = np.prod(action_space.shape)

    # --- Agent, Models, Optimizers ---
    actor = Actor(norm_envs).to(device)
    critic = HJBCritic(norm_envs).to(device)
    policy_optimizer = optim.AdamW(actor.parameters(), lr=args.policy_lr)
    critic_optimizer = optim.AdamW(critic.parameters(), lr=args.q_lr)
    dynamic_model = DynamicModel(obs_dim, action_dim, args.env_dt, device).to(device)
    reward_model = RewardModel(obs_dim, action_dim).to(device)
    dynamics_optimizer = optim.AdamW(
        dynamic_model.ode_func.parameters(), lr=args.dynamics_learning_rate
    )
    reward_optimizer = optim.AdamW(
        reward_model.parameters(), lr=args.reward_learning_rate
    )

    # --- Alpha Setup ---
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.AdamW([log_alpha], lr=args.q_lr)
        print(f"Autotune alpha. Target Entropy: {target_entropy:.2f}")
    else:
        alpha = args.alpha
        a_optimizer = None
        log_alpha = None
        target_entropy = None
        print(f"Fixed alpha: {alpha}")

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
    norm_envs.seed(args.seed)  # Seed first
    obs = norm_envs.reset()  # Then reset
    obs = obs.astype(np.float32)
    dynamic_model_accurate = False
    reward_model_accurate = False
    global_step = 0

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

    # ========================================================================
    # <<< Main Training Loop >>>
    # ========================================================================
    print(f"Starting training loop for {args.total_timesteps} timesteps...")
    for global_step in range(args.total_timesteps):
        iter_start_time = time.time()
        # LR Annealing (Removed)

        # --- Action Selection & Environment Interaction (Phase 1 Equivalent) ---
        if global_step < args.learning_starts:
            actions = np.array(
                [norm_envs.action_space.sample() for _ in range(norm_envs.num_envs)]
            )
        else:
            with torch.no_grad():
                actions_tensor, _, _ = actor.get_action(
                    torch.Tensor(obs).to(device), deterministic=False
                )
                actions = actions_tensor.cpu().numpy()

        # <<< Fix: Unpack 4 values from SB3 VecEnv step >>>
        next_obs_norm, rewards_norm, dones_combined_np, infos = norm_envs.step(actions)
        next_obs_norm = next_obs_norm.astype(np.float32)
        rewards_norm = rewards_norm.astype(np.float32)  # Normalized reward from wrapper

        # Calculate termination flags for buffer (needed if handle_timeout=True)
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
            final_infos = [i for i in infos if i is not None]  # Compatibility
        for info in final_infos:
            if info and "episode" in info:
                episode_info = info["episode"]
                # <<< Fix: Cast info values before formatting >>>
                return_val = float(episode_info["r"])
                length_val = int(episode_info["l"])
                print(
                    f"GStep={global_step}, EpReturn(Norm)={return_val:.2f}, EpLen={length_val}"
                )
                writer.add_scalar(
                    "charts/episodic_return_norm", return_val, global_step
                )
                writer.add_scalar("charts/episodic_length", length_val, global_step)
                break  # Log only first finished env info if multiple finish

        # Store RAW data in replay buffer
        raw_obs = norm_envs.unnormalize_obs(obs)
        raw_reward = norm_envs.get_original_reward()
        real_next_obs_raw = norm_envs.unnormalize_obs(next_obs_norm)
        for idx, done in enumerate(dones_combined_np):  # Check combined done
            # Use try-except for robustness accessing infos[idx]
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
                pass  # Ignore if infos doesn't have entry for this index
        # Pass combined done flag to buffer add

        # <<< Debug Check: Compare raw obs vs raw next_obs BEFORE adding to buffer >>>
        try:
            diff_add_check = np.mean(np.abs(raw_obs - real_next_obs_raw))
            if global_step % 100 == 0: # Print occasionally
                print(f"DEBUG Add Check (GStep {global_step}): Mean Abs Diff Raw Obs vs Raw Next Obs: {diff_add_check:.4e}")
            assert diff_add_check > 1e-6 or np.all(dones_combined_np), "Raw obs and next_obs are too similar BEFORE adding to buffer!"
        except AssertionError as e:
            print(f"ASSERTION FAILED: {e}") # Print assertion message if it fails

        rb.add(
            raw_obs, real_next_obs_raw, actions, raw_reward, dones_combined_np, infos
        )

        obs = next_obs_norm  # Update agent state to NORMALIZED next obs

        # --- Model Training/Validation (Phase 2 Equivalent - Periodic) ---
        if (
            global_step > args.learning_starts
            and global_step % args.model_train_freq == 0
        ):
            print(f"\n--- GStep {global_step}: Checking/Training Models ---")
            model_train_start_time = time.time()
            buffer_data_raw_tensors = rb.sample(args.model_dataset_size, env=None)
            obs_raw_np = buffer_data_raw_tensors.observations.cpu().numpy()
            next_obs_raw_np = buffer_data_raw_tensors.next_observations.cpu().numpy()
            actions_np = buffer_data_raw_tensors.actions.cpu().numpy()
            rewards_raw_np = buffer_data_raw_tensors.rewards.cpu().numpy()
            dones_term_only_np = buffer_data_raw_tensors.dones.cpu().numpy()

            # <<< Debug Check: Compare Raw Obs vs Raw Next Obs >>>
            print("--- Debug Sampled Raw Tensors ---")
            obs_raw_sample = buffer_data_raw_tensors.observations
            next_obs_raw_sample = buffer_data_raw_tensors.next_observations
            print(f"Sampled Obs Raw Shape: {obs_raw_sample.shape}")
            print(f"Sampled Next Obs Raw Shape: {next_obs_raw_sample.shape}")
            # Calculate difference ON THE SAME DEVICE as the tensors
            diff_raw = torch.abs(obs_raw_sample - next_obs_raw_sample).mean()
            print(f"Mean Abs Diff between Raw Obs and Raw Next Obs in Sample: {diff_raw.item():.4e}")
            # Assert they are different enough (tune tolerance)
            assert diff_raw.item() > 1e-6, "Sampled raw obs and next_obs are too similar! Check buffer storage/sampling."
            print("--- End Debug Sampled Raw Tensors ---")

            obs_norm_np = norm_envs.normalize_obs(obs_raw_np)
            next_obs_norm_np = norm_envs.normalize_obs(next_obs_raw_np)
            rewards_norm_np = norm_envs.normalize_reward(
                rewards_raw_np.reshape(-1)
            ).reshape(-1, 1)
            obs_norm_t = torch.tensor(obs_norm_np, dtype=torch.float32).to(device)
            next_obs_norm_target_t = torch.tensor(
                next_obs_norm_np, dtype=torch.float32
            ).to(device)
            actions_t = torch.tensor(actions_np, dtype=torch.float32).to(device)
            rewards_norm_target_t = torch.tensor(
                rewards_norm_np, dtype=torch.float32
            ).to(device)
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
                    train_loss = train_model_epoch(
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
                    # <<< Syntax Fix: Validation Check Block >>>
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
                            f"  Dyn Epoch {epoch+1}: TrLs={train_loss}, ValLs={val_loss}, ValR2={val_metrics['r2']}"
                        )
                        if val_loss < best_val_loss - args.model_val_delta:
                            best_val_loss = val_loss
                            patience_counter = 0
                            best_dyn_state_dict = copy.deepcopy(
                                dynamic_model.state_dict()
                            )  # Save full model state
                        else:
                            patience_counter += args.model_validation_freq
                        if patience_counter >= args.model_val_patience:
                            print(f"  Early stopping dyn model at epoch {epoch+1}.")
                            break  # Exit inner epoch loop
                # Load best state and run final validation
                if best_dyn_state_dict:
                    dynamic_model.load_state_dict(best_dyn_state_dict)
                    print(f"  Loaded best dyn model (VlLs:{best_val_loss:.5f})")
                    final_validation_loss_state = (
                        best_val_loss  # Use the best loss found
                    )
                else:
                    print(
                        "  No improvement in dyn validation, using final model state."
                    )
                    # Need to calculate final val loss if no improvement was ever found
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

                # Perform final R2 calculation on the model state being used
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
                validation_loss_state = (
                    final_validation_loss_state  # Use the determined final/best loss
                )

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
                    f"DynModel Complete. Final Val Loss:{validation_loss_state:.5f}. Final R2:{validation_r2_score:.3f}. Acc:{dynamic_model_accurate}"
                )
                writer.add_scalar(
                    "charts/dynamic_model_accurate",
                    float(dynamic_model_accurate),
                    global_step,
                )

            # --- Reward Model ---
            rew_obs_t = obs_norm_t
            rew_acts_t = actions_t
            rew_targets_t = rewards_norm_target_t.squeeze(-1)
            if len(rew_obs_t) < 2:
                print("Warn:Not enough samples for rew model.")
                reward_model_accurate = False
            else:
                indices = torch.randperm(len(rew_obs_t), device=device)
                split = int(len(rew_obs_t) * (1 - args.model_val_ratio))
                train_idx, val_idx = indices[:split], indices[split:]
                train_dataset = TensorDataset(
                    rew_obs_t[train_idx],
                    rew_acts_t[train_idx],
                    rew_targets_t[train_idx],
                )
                val_dataset = TensorDataset(
                    rew_obs_t[val_idx], rew_acts_t[val_idx], rew_targets_t[val_idx]
                )
                train_loader = DataLoader(
                    train_dataset, batch_size=args.model_train_batch_size, shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=args.model_train_batch_size
                )
                print(f"RewModel:Tr={len(train_idx)},Vl={len(val_idx)}")
                best_val_loss = float("inf")
                patience_counter = 0
                reward_model_accurate = False
                best_rew_state_dict = None
                final_model_epoch = 0
                reward_model.train()
                for epoch in range(args.model_max_epochs):
                    final_model_epoch = epoch
                    train_loss = train_model_epoch(
                        reward_model,
                        reward_optimizer,
                        train_loader,
                        device,
                        writer,
                        epoch,
                        global_step,
                        "reward",
                        False,
                    )
                    # <<< Syntax Fix: Validation Check Block >>>
                    if (epoch + 1) % args.model_validation_freq == 0:
                        val_loss, val_metrics = validate_model(
                            reward_model,
                            val_loader,
                            device,
                            writer,
                            global_step + epoch + 1,
                            "reward",
                            False,
                        )
                        print(
                            f"  Rew Epoch {epoch+1}: TrLs={train_loss:.5f}, ValLs={val_loss:.5f}, ValR2={val_metrics['r2']:.3f}"
                        )
                        if val_loss < best_val_loss - args.model_val_delta:
                            best_val_loss = val_loss
                            patience_counter = 0
                            best_rew_state_dict = copy.deepcopy(
                                reward_model.state_dict()
                            )
                        else:
                            patience_counter += args.model_validation_freq
                        if patience_counter >= args.model_val_patience:
                            print(f"  Early stopping rew model at epoch {epoch+1}.")
                            break  # Exit inner epoch loop
                # Load best state and run final validation
                if best_rew_state_dict:
                    reward_model.load_state_dict(best_rew_state_dict)
                    print(f"  Loaded best rew model (VlLs:{best_val_loss:.5f})")
                    final_validation_loss_reward = best_val_loss
                else:
                    print("  No improvement in rew validation.")
                    reward_model.eval()
                    final_val_loss, _ = validate_model(
                        reward_model,
                        val_loader,
                        device,
                        writer,
                        global_step,
                        "reward_final_eval",
                        False,
                    )
                    reward_model.train()
                    final_validation_loss_reward = final_val_loss

                reward_model.eval()
                _, final_val_metrics = validate_model(
                    reward_model,
                    val_loader,
                    device,
                    writer,
                    global_step,
                    "reward_final_metrics",
                    False,
                )
                reward_model.train()
                validation_loss_reward = final_validation_loss_reward
                validation_r2_score_reward = final_val_metrics[
                    "r2"
                ]  # Get R2 for reward model too
                reward_model_accurate = (
                    validation_loss_reward <= args.reward_train_threshold
                )
                writer.add_scalar(
                    "losses/reward_model_validation_loss_final",
                    validation_loss_reward,
                    global_step,
                )
                writer.add_scalar(
                    "losses/reward_model_R2_final",
                    validation_r2_score_reward,
                    global_step,
                )  # Log reward R2
                print(
                    f"RewModel Complete. Final Val Loss:{validation_loss_reward:.5f}. Final R2:{validation_r2_score_reward:.3f}. Acc:{reward_model_accurate}"
                )
                writer.add_scalar(
                    "charts/reward_model_accurate",
                    float(reward_model_accurate),
                    global_step,
                )
            print(f"--- Model Check/Training Finished ---")
            model_train_time = time.time() - model_train_start_time
            writer.add_scalar("perf/model_train_time", model_train_time, global_step)

        # --- Phase 3: Model Rollout Generation ---
        model_rollout_gen_time = 0
        can_rollout = (
            global_step > args.learning_starts
            and global_step % args.model_rollout_freq == 0
            and dynamic_model_accurate
            and reward_model_accurate
            and rb.size() >= args.num_model_rollout_starts
            and 1==0
        )
        if can_rollout:
            rollout_start_time = time.time()
            print(
                f"--- GStep {global_step}: Generating Model Rollouts ({args.num_model_rollout_starts} starts, length {args.model_rollout_length}) ---"
            )
            dynamic_model.eval()
            reward_model.eval()
            actor.eval()
            start_states_samples = rb.sample(args.num_model_rollout_starts, env=None)
            current_obs_raw_np = start_states_samples.observations.cpu().numpy()
            num_added = 0
            with torch.no_grad():
                for h in range(args.model_rollout_length):
                    current_obs_norm_np = norm_envs.normalize_obs(current_obs_raw_np)
                    current_obs_norm_t = torch.tensor(
                        current_obs_norm_np, dtype=torch.float32
                    ).to(device)
                    action_tensor, _, _ = actor.get_action(
                        current_obs_norm_t, deterministic=False
                    )
                    action_np = action_tensor.cpu().numpy()
                    next_obs_norm_pred_t = dynamic_model(
                        current_obs_norm_t, action_tensor
                    )
                    reward_norm_pred_t = reward_model(current_obs_norm_t, action_tensor)
                    next_obs_raw_pred_np = norm_envs.unnormalize_obs(
                        next_obs_norm_pred_t.cpu().numpy()
                    )
                    reward_raw_pred_np = norm_envs.unnormalize_reward(
                        reward_norm_pred_t.cpu().numpy()
                    )
                    dones_model = np.zeros(args.num_model_rollout_starts, dtype=bool)
                    terminations_model = np.zeros(
                        args.num_model_rollout_starts, dtype=bool
                    )
                    infos_model = [{} for _ in range(args.num_model_rollout_starts)]
                    if args.num_envs == 1:
                        for i in range(args.num_model_rollout_starts):
                            rb.add(
                                current_obs_raw_np[i : i + 1],
                                next_obs_raw_pred_np[i : i + 1],
                                action_np[i : i + 1],
                                reward_raw_pred_np[i : i + 1],
                                dones_model[i : i + 1],
                                [{}],
                            )  # Pass combined done (False)
                            num_added += 1
                    else:
                        rb.add(
                            current_obs_raw_np,
                            next_obs_raw_pred_np,
                            action_np,
                            reward_raw_pred_np,
                            dones_model,
                            infos_model,
                        )
                        num_added += args.num_model_rollout_starts
                    current_obs_raw_np = next_obs_raw_pred_np
            model_rollout_gen_time = time.time() - rollout_start_time
            print(
                f"  Model rollout generation complete ({num_added} samples added). Time: {model_rollout_gen_time:.2f}s"
            )
            writer.add_scalar(
                "perf/model_rollout_gen_time", model_rollout_gen_time, global_step
            )
            actor.train()
        elif (
            global_step > args.learning_starts
            and global_step % args.model_rollout_freq == 0
        ):
            print(f"GStep {global_step}: Skipping model rollouts.")

        # --- Agent Training ---
        if global_step > args.learning_starts:
            # Agent update proceeds even if models are inaccurate
            if not dynamic_model_accurate or not reward_model_accurate:
                if global_step % 1000 == 0:
                    print(
                        f"Info: Proceeding with agent update step {global_step}, but models INACCURATE (DynAcc={dynamic_model_accurate}, RewAcc={reward_model_accurate})"
                    )

            data = rb.sample(
                args.batch_size, env=None
            )  # Samples Tensors: raw data + term-only dones
            # Convert raw Tensors -> NumPy -> Normalize -> Normalized Tensors
            obs_raw_np = data.observations.cpu().numpy()
            actions_np = data.actions.cpu().numpy()
            rewards_raw_np = data.rewards.cpu().numpy()
            next_obs_raw_np = data.next_observations.cpu().numpy()
            dones_term_only_np = data.dones.cpu().numpy()
            obs_norm_np = norm_envs.normalize_obs(obs_raw_np)
            rewards_norm_np = norm_envs.normalize_reward(
                rewards_raw_np.reshape(-1)
            ).reshape(-1, 1)
            mb_obs = torch.tensor(obs_norm_np, dtype=torch.float32).to(device)
            mb_actions = torch.tensor(actions_np, dtype=torch.float32).to(device)
            mb_rewards = (
                torch.tensor(rewards_norm_np, dtype=torch.float32)
                .to(device)
                .squeeze(-1)
            )
            mb_dones = (
                torch.tensor(dones_term_only_np, dtype=torch.float32)
                .to(device)
                .squeeze(-1)
            )  # Term-only dones [batch]

            # --- Critic Update ---
            terminations_mask = mb_dones.bool()
            non_terminations_mask = ~terminations_mask
            all_current_v = critic(mb_obs)
            v_term = all_current_v[terminations_mask]
            terminal_critic_loss = torch.tensor(0.0, device=device)
            if v_term.numel() > 0:
                terminal_critic_loss = F.mse_loss(v_term, torch.zeros_like(v_term))
            hjb_loss_non_term = torch.tensor(0.0, device=device)
            if (
                non_terminations_mask.any()
                and args.use_hjb_loss
                and compute_value_grad_func is not None
            ):
                obs_non_term = mb_obs[non_terminations_mask]
                obs_non_term.requires_grad_(True)
                with torch.no_grad():
                    actions_pi_non_term, _, _ = actor.get_action(
                        obs_non_term, deterministic=False
                    )
                v_non_term = all_current_v[non_terminations_mask]
                with torch.no_grad():
                    f_non_term = dynamic_model.ode_func(
                        0, obs_non_term, actions_pi_non_term
                    )
                    r_non_term = reward_model(obs_non_term, actions_pi_non_term)
                try:
                    dVdx_non_term = vmap(compute_value_grad_func)(obs_non_term)
                except Exception as e:
                    print(f"HJB dVdx Error:{e}")
                    dVdx_non_term = torch.zeros_like(obs_non_term)
                obs_non_term.requires_grad_(False)
                r_hjb = mb_rewards[
                    non_terminations_mask
                ]  # Use normalized reward from sample
                hjb_residual = (
                    r_hjb + torch.einsum("bi,bi->b", dVdx_non_term, f_non_term)
                ) - rho * v_non_term
                hjb_loss_non_term = 0.5 * (hjb_residual**2).mean()
            critic_loss = hjb_loss_non_term + args.terminal_coeff * terminal_critic_loss
            critic_optimizer.zero_grad()
            critic_loss.backward()
            if args.grad_norm_clip is not None:
                nn.utils.clip_grad_norm_(critic.parameters(), args.grad_norm_clip)
            critic_optimizer.step()

            # --- Actor Update (Delayed & SAC Style) ---
            if global_step % args.policy_frequency == 0:
                for p in critic.parameters():
                    p.requires_grad = False
                obs_for_actor = mb_obs[non_terminations_mask]
                if obs_for_actor.numel() > 0:
                    pi_actions, log_pi, _ = actor.get_action(
                        obs_for_actor, deterministic=False
                    )
                    obs_for_actor.requires_grad_(True)
                    try:
                        dVdx_actor = vmap(compute_value_grad_func)(
                            obs_for_actor
                        ).detach()
                    except Exception as e:
                        print(f"Actor dVdx Error:{e}")
                        dVdx_actor = torch.zeros_like(obs_for_actor)
                    obs_for_actor.requires_grad_(False)
                    with torch.no_grad():
                        f_actor = dynamic_model.ode_func(0, obs_for_actor, pi_actions)
                        r_actor = reward_model(obs_for_actor, pi_actions)
                    hamiltonian_actor = -r_actor + torch.einsum(
                        "bi,bi->b", dVdx_actor, f_actor
                    )
                    actor_loss = (alpha * log_pi.squeeze(-1) + hamiltonian_actor).mean()
                    policy_optimizer.zero_grad()
                    actor_loss.backward()
                    if args.grad_norm_clip is not None:
                        nn.utils.clip_grad_norm_(
                            actor.parameters(), args.grad_norm_clip
                        )
                    policy_optimizer.step()
                    writer.add_scalar(
                        "losses/actor_loss", actor_loss.item(), global_step
                    )
                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi_alpha, _ = actor.get_action(
                                obs_for_actor, deterministic=False
                            )
                        alpha_loss = (
                            -log_alpha.exp() * (log_pi_alpha.detach() + target_entropy)
                        ).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()
                        writer.add_scalar(
                            "losses/alpha_loss", alpha_loss.item(), global_step
                        )
                else:
                    actor_loss = torch.tensor(0.0)
                    print(
                        f"Skipping actor update step {global_step}: no non-terminal states in batch"
                    )
                for p in critic.parameters():
                    p.requires_grad = True

            # Logging Critic losses
            writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
            writer.add_scalar(
                "losses/critic_terminal", terminal_critic_loss.item(), global_step
            )
            writer.add_scalar(
                "losses/critic_hjb_non_term", hjb_loss_non_term.item(), global_step
            )
            writer.add_scalar(
                "metrics/critic_value_mean", all_current_v.mean().item(), global_step
            )
            writer.add_scalar("losses/alpha", alpha, global_step)

        # Log SPS occasionally
        if global_step > 0 and global_step % 1000 == 0:
            sps = int(global_step / (time.time() - start_time))
            print(f"GStep: {global_step}, SPS: {sps}")
            writer.add_scalar("charts/SPS", sps, global_step)

    # --- End of Training Loop ---
    # --- Saving & Evaluation ---
    actor_final = actor
    critic_final = critic
    if args.save_model:
        run_folder = f"runs/{run_name}"
        os.makedirs(run_folder, exist_ok=True)
        actor_model_path = f"{run_folder}/{args.exp_name}_actor.cleanrl_model"
        torch.save(actor_final.state_dict(), actor_model_path)
        print(f"Actor saved: {actor_model_path}")
        critic_model_path = f"{run_folder}/{args.exp_name}_critic.cleanrl_model"
        torch.save(critic_final.state_dict(), critic_model_path)
        print(f"Critic saved: {critic_model_path}")
        dynamics_ode_path = (
            f"{run_folder}/{args.exp_name}_dynamics_odefunc.cleanrl_model"
        )
        torch.save(dynamic_model.ode_func.state_dict(), dynamics_ode_path)
        print(f"Dynamics ODEFunc saved: {dynamics_ode_path}")
        reward_model_path = f"{run_folder}/{args.exp_name}_reward_model.cleanrl_model"
        torch.save(reward_model.state_dict(), reward_model_path)
        print(f"Reward model saved: {reward_model_path}")
        norm_stats_path = f"{run_folder}/{args.exp_name}_vecnormalize.pkl"
        norm_envs.save(norm_stats_path)
        print(f"Normalization stats saved: {norm_stats_path}")
    if args.save_model:
        print("\nEvaluating agent performance...")
        eval_episodes = 10
        eval_seeds = range(args.seed + 100, args.seed + 100 + eval_episodes)
        eval_returns_raw = []
        for seed in eval_seeds:
            eval_envs_base = DummyVecEnv(
                [make_env(args.env_id, seed, False, f"{run_name}-eval-seed{seed}")]
            )
            eval_norm_envs = VecNormalize.load(norm_stats_path, eval_envs_base)
            eval_norm_envs.training = False
            eval_norm_envs.norm_reward = False
            eval_actor = Actor(eval_norm_envs).to(device)
            eval_actor.load_state_dict(
                torch.load(actor_model_path, map_location=device)
            )
            eval_actor.eval()
            obs_norm_np = eval_norm_envs.reset(seed=seed)
            done = False
            episode_return_raw = 0
            num_steps = 0
            max_steps = 1000
            while not done and num_steps < max_steps:
                with torch.no_grad():
                    action, _, _ = eval_actor.get_action(
                        torch.Tensor(obs_norm_np).to(device), deterministic=True
                    )
                    action = action.cpu().numpy()
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
