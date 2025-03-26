# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import vmap
from torch.func import grad
import torchode as to
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
# Note: ReplayBuffer is imported from stable_baselines3
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the Atari game"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6) # Reduced buffer size
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    batch_size: int = 256 # Reduced batch size for agent updates
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2 # Standard DDPG/TD3 delayed policy update frequency
    """the frequency of training policy (delayed)"""
    ema_decay: float = 0.995 # EMA decay rate for target networks
    """EMA decay rate (typically 0.999-0.9999)"""
    # Removed noise_clip as it's TD3 specific, not canonical DDPG or this HJB variant

    # Model Training specific arguments
    model_train_freq: int = 1000 # Frequency to check and potentially retrain models
    """Frequency (in global steps) to check model accuracy and retrain if needed"""
    model_dataset_size: int = 50000 # Size of dataset sampled for model training/validation
    """Number of samples drawn from the buffer for model training/validation"""
    dynamic_train_threshold: float = 0.01
    """validation loss threshold to consider dynamic model accurate enough"""
    reward_train_threshold: float = 0.001
    """validation loss threshold to consider reward model accurate enough"""
    model_val_ratio: float = 0.2 # Unified validation ratio
    """ratio of validation data for model training"""
    model_val_patience: int = 10 # Unified patience epochs
    """patience epochs for model early stopping"""
    model_val_delta: float = 1e-4 # Unified minimum improvement delta
    """minimum improvement delta for model early stopping"""
    model_max_epochs: int = 50 # Unified maximum training epochs
    """maximum training epochs for models"""
    model_train_batch_size: int = 1024 # Mini-batch size for model training epochs
    """batch size for training models"""
    grad_norm_clip: Optional[float] = 1.0 # Gradient clipping for actor/critic
    """gradient norm clipping threshold (None for no clipping)"""


def make_env(env_id, seed, idx, capture_video, run_name):
    """Creates a Gymnasium environment with specified wrappers."""
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # Apply observation normalization
        env = gym.wrappers.NormalizeObservation(env)
        env.action_space.seed(seed)
        # It's essential to get the dt after all wrappers that might affect it
        try:
            # For MuJoCo environments
            dt = env.unwrapped.model.opt.timestep * env.unwrapped.frame_skip
        except AttributeError:
            # Fallback for other environments, may need adjustment
            print("Warning: Could not automatically determine environment dt. Using default 1/50.")
            dt = 0.02 # A common default, but verify for your specific env_id!

        return env, dt

    return thunk


# --- Model Definitions ---

class ODEFunc(nn.Module):
    """The function f(t, x, a) defining the ODE dx/dt = f(t, x, a)."""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.SiLU(), # Changed activation
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, obs_dim),
        )

    def forward(self, t, x, a):
        # t is unused in this autonomous system, but required by torchode
        return self.net(torch.cat([x, a], dim=-1))

class DynamicModel(nn.Module):
    """Predicts the next state using a neural ODE."""
    def __init__(self, obs_dim, action_dim, dt: float, device: torch.device):
        super().__init__()
        self.ode_func = ODEFunc(obs_dim, action_dim)
        self.dt = dt
        self.device = device

        # TorchODE components
        self.term = to.ODETerm(self.ode_func, with_args=True)
        # Using a fixed-step method might be more stable if dt is constant
        # self.step_method = to.Euler(term=self.term)
        self.step_method = to.Tsit5(term=self.term) # Adaptive step size
        self.step_size_controller = to.FixedStepController() # Use fixed dt
        # self.step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-6, term=self.term) # Adaptive dt control
        self.adjoint = to.AutoDiffAdjoint(
            step_method=self.step_method,
            step_size_controller=self.step_size_controller,
        )

    def forward(self, initial_obs, actions):
        batch_size = initial_obs.shape[0]
        # Use the environment's dt for the step size controller
        dt0 = torch.full((batch_size,), self.dt, device=self.device)
        # Evaluate from t=0 to t=dt
        t_eval = torch.tensor([0.0, self.dt], device=self.device).repeat((batch_size, 1))

        problem = to.InitialValueProblem(
            y0=initial_obs,
            t_eval=t_eval,
        )
        # Solve the ODE
        sol = self.adjoint.solve(problem, args=actions, dt0=dt0)

        # Return only the final state prediction at t=dt
        return sol.ys[:, 1, :]  # Shape: (batch_size, obs_dim)

class RewardModel(nn.Module):
    """Predicts the reward for a given state-action pair."""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        return self.net(x).squeeze(-1) # Output shape: (batch_size,)

# --- Agent Network Definitions ---

class HJBCritic(nn.Module):
    """Value function V(x)."""
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.observation_space.shape).prod()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = self.fc3(x).squeeze(-1) # Output shape: (batch_size,)
        return x

class HJBActor(nn.Module):
    """Policy function pi(x)."""
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.observation_space.shape).prod()
        action_dim = np.prod(env.action_space.shape)
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        # Action rescaling buffers
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        # Output [-1, 1] range actions
        x = torch.tanh(self.fc_mu(x))
        # Rescale to environment action space
        return x * self.action_scale + self.action_bias

# --- Utility Functions ---

def calculate_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Calculates MSE, MAE, and R-squared metrics."""
    mse = F.mse_loss(preds, targets).item()
    mae = F.l1_loss(preds, targets).item()

    # Calculate R-squared
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = (1 - ss_res / ss_tot).item() if ss_tot != 0 else 1.0 # Handle zero variance case

    return {"mse": mse, "mae": mae, "r2": r2}

def train_model_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    global_step: int,
    model_name: str,
    is_dynamic_model: bool,
) -> float:
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0.0
    for batch_idx, batch_data in enumerate(train_loader):
        obs, actions, targets = [d.to(device) for d in batch_data]

        if is_dynamic_model:
            preds = model(obs, actions) # Dynamic model predicts next_obs
        else:
            preds = model(obs, actions) # Reward model predicts reward

        loss = F.mse_loss(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Log batch loss occasionally (e.g., every 10 batches)
        if batch_idx % 10 == 0:
             step = global_step + epoch * len(train_loader) + batch_idx # More granular step
             writer.add_scalar(f"losses/{model_name}_batch_mse", loss.item(), step)

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    global_step: int,
    model_name: str,
    is_dynamic_model: bool,
) -> Tuple[float, Dict[str, float]]:
    """Validates the model."""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_data in val_loader:
            obs, actions, targets = [d.to(device) for d in batch_data]

            if is_dynamic_model:
                preds = model(obs, actions)
            else:
                preds = model(obs, actions)

            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    val_metrics = calculate_metrics(all_preds, all_targets)
    val_loss = val_metrics["mse"] # Use MSE for early stopping check

    # Log validation metrics
    writer.add_scalar(f"losses/{model_name}_val_mse", val_metrics["mse"], global_step)
    writer.add_scalar(f"metrics/{model_name}_val_mae", val_metrics["mae"], global_step)
    writer.add_scalar(f"metrics/{model_name}_val_r2", val_metrics["r2"], global_step)

    return val_loss, val_metrics


# --- Main Execution ---

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Environment setup
    env_fn = make_env(args.env_id, args.seed, 0, args.capture_video, run_name)
    envs, env_dt = env_fn() # Get env and dt
    print(f"Environment dt: {env_dt}")
    assert isinstance(envs.action_space, gym.spaces.Box), "only continuous action space is supported"
    # Ensure observation space is float32 for normalization wrapper
    envs.observation_space.dtype = np.float32

    # Agent setup
    actor = HJBActor(envs).to(device)
    critic = HJBCritic(envs).to(device)
    actor_optimizer = optim.AdamW(actor.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic.parameters(), lr=args.learning_rate)

    # EMA models for stability
    ema_actor = torch.optim.swa_utils.AveragedModel(
        actor,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema_decay)
    )
    ema_critic = torch.optim.swa_utils.AveragedModel(
        critic,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema_decay)
    )

    # Dynamic and Reward Model setup
    obs_dim = np.array(envs.observation_space.shape).prod()
    action_dim = np.prod(envs.action_space.shape)
    dynamic_model = DynamicModel(obs_dim, action_dim, env_dt, device).to(device)
    reward_model = RewardModel(obs_dim, action_dim).to(device)
    dynamic_optimizer = optim.AdamW(dynamic_model.parameters(), lr=args.learning_rate)
    reward_optimizer = optim.AdamW(reward_model.parameters(), lr=args.learning_rate)

    # Replay buffer
    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        handle_timeout_termination=False, # Important for model learning
    )

    # Continuous-time discount rate (rho) based on discrete gamma and dt
    # gamma = exp(-rho * dt) => rho = -log(gamma) / dt
    rho = -torch.log(torch.tensor(args.gamma, device=device)) / env_dt
    print(f"Continuous discount rate (rho): {rho.item()}")

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    dynamic_model_accurate = False
    reward_model_accurate = False

    # --- Training Loop ---
    for global_step in range(args.total_timesteps):
        # Action selection
        if global_step < args.learning_starts:
            # Sample single action for the single environment
            actions = envs.action_space.sample()
        else:
            with torch.no_grad():
                # Use EMA actor for exploration
                # Add batch dimension for the single observation
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
                actions_tensor = ema_actor(obs_tensor)
                # Add exploration noise scaled by action range
                noise = torch.normal(0, actor.action_scale * args.exploration_noise, device=device)
                actions_tensor += noise
                # Remove batch dimension and clip action
                actions = actions_tensor.squeeze(0).cpu().numpy().clip(envs.action_space.low, envs.action_space.high)

        # Environment step
        # Unpack correctly for single env: obs, reward, terminated, truncated, info
        next_obs, reward, terminated, truncated, info = envs.step(actions)
        # Convert reward to float32 for consistency
        reward = np.float32(reward)

        # Logging
        # info dict contains the final_info if episode ended
        if terminated or truncated:
            final_info = info.get("final_info")
            if final_info and "episode" in final_info:
                print(f"global_step={global_step}, episodic_return={final_info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", final_info['episode']['r'], global_step)
                writer.add_scalar("charts/episodic_length", final_info['episode']['l'], global_step)
                # Log normalized env stats if available (assuming NormalizeObservation wrapper)
                # Note: Accessing _running_mean/_running_var might be fragile if wrappers change
                if hasattr(envs, "_running_mean"):
                    writer.add_scalar("charts/obs_mean", envs._running_mean.mean().item(), global_step)
                    writer.add_scalar("charts/obs_std", np.sqrt(envs._running_var).mean().item(), global_step)


        # Store transition in replay buffer
        # Use `real_next_obs` for buffer storage if available (handles truncation)
        real_next_obs = next_obs.copy()
        # Handle truncation for single environment
        if truncated:
            # Check if 'final_observation' is available and not None
            final_obs = info.get("final_observation")
            if final_obs is not None:
                 real_next_obs = final_obs
            # else: use next_obs as is, though it might be from a truncated episode start

        # Add batch dimension before adding to buffer
        # Wrap single values in arrays/lists as expected by ReplayBuffer
        rb.add(np.expand_dims(obs, 0), np.expand_dims(real_next_obs, 0), np.expand_dims(actions, 0), np.array([reward]), np.array([terminated]), [info])


        # Update current observation
        obs = next_obs
            for info in infos["final_info"]:
                # Skip gymnasium internal final_info keys
                if "episode" not in info: continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                # Log normalized env stats if available
                if "running_mean" in info:
                    writer.add_scalar("charts/obs_mean", info["running_mean"].mean().item(), global_step)
                    writer.add_scalar("charts/obs_std", np.sqrt(info["running_var"]).mean().item(), global_step)


        # Store transition in replay buffer
        # Use `real_next_obs` for buffer storage if available (handles truncation)
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                # Check if 'final_observation' is available and not None
                if "final_observation" in infos and infos["final_observation"] is not None and infos["final_observation"][idx] is not None:
                     real_next_obs[idx] = infos["final_observation"][idx]
                # else: use next_obs as is, though it might be from a truncated episode start

        # Add batch dimension before adding to buffer
        rb.add(np.expand_dims(obs, 0), np.expand_dims(real_next_obs, 0), np.expand_dims(actions, 0), np.array([rewards]), np.array([terminations]), [infos])


        # Update current observation
        obs = next_obs

        # --- Model Training (Periodic) ---
        if global_step > args.learning_starts and global_step % args.model_train_freq == 0:
            print(f"\n--- Global Step {global_step}: Checking/Training Models ---")

            # --- Dynamic Model Training ---
            print("Sampling data for Dynamic Model...")
            dyn_data = rb.sample(args.model_dataset_size, env=None) # Sample large dataset
            # Filter non-terminal transitions for dynamics prediction
            non_terminal_mask = dyn_data.dones.squeeze(-1) == 0
            dyn_obs = dyn_data.observations[non_terminal_mask]
            dyn_acts = dyn_data.actions[non_terminal_mask]
            dyn_targets = dyn_data.next_observations[non_terminal_mask]

            if len(dyn_obs) < 2: # Need at least 2 samples for train/val split
                print("Warning: Not enough non-terminal samples for dynamic model training.")
                dynamic_model_accurate = False # Assume inaccurate if cannot train
            else:
                # Split data
                indices = torch.randperm(len(dyn_obs), device=device)
                split = int(len(dyn_obs) * (1 - args.model_val_ratio))
                train_idx, val_idx = indices[:split], indices[split:]

                train_dataset = TensorDataset(dyn_obs[train_idx], dyn_acts[train_idx], dyn_targets[train_idx])
                val_dataset = TensorDataset(dyn_obs[val_idx], dyn_acts[val_idx], dyn_targets[val_idx])
                train_loader = DataLoader(train_dataset, batch_size=args.model_train_batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=args.model_train_batch_size)

                print(f"Dynamic Model: Train size={len(train_idx)}, Val size={len(val_idx)}")
                best_val_loss = float('inf')
                patience_counter = 0
                dynamic_model_accurate = False # Reset accuracy flag

                for epoch in range(args.model_max_epochs):
                    train_loss = train_model_epoch(dynamic_model, dynamic_optimizer, train_loader, device, writer, epoch, global_step, "dynamic", True)
                    val_loss, val_metrics = validate_model(dynamic_model, val_loader, device, writer, global_step + epoch + 1, "dynamic", True) # Offset step slightly

                    print(f"  Epoch {epoch+1}/{args.model_max_epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

                    # Early stopping check
                    if val_loss < best_val_loss - args.model_val_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Check if the best loss meets the threshold
                        if best_val_loss <= args.dynamic_train_threshold:
                            dynamic_model_accurate = True
                            print(f"  Dynamic model reached accuracy threshold ({best_val_loss:.6f} <= {args.dynamic_train_threshold}).")
                            # Optionally break early if accuracy is met and patience allows
                            # break
                    else:
                        patience_counter += 1
                        if patience_counter >= args.model_val_patience:
                            print(f"  Early stopping triggered at epoch {epoch+1}.")
                            break

                # Final check on accuracy based on the best validation loss achieved
                dynamic_model_accurate = (best_val_loss <= args.dynamic_train_threshold)
                print(f"Dynamic Model Training Complete. Best Val Loss: {best_val_loss:.6f}. Accurate: {dynamic_model_accurate}")
                writer.add_scalar("charts/dynamic_model_accurate", float(dynamic_model_accurate), global_step)


            # --- Reward Model Training ---
            print("Sampling data for Reward Model...")
            rew_data = rb.sample(args.model_dataset_size, env=None) # Sample fresh large dataset
            rew_obs = rew_data.observations
            rew_acts = rew_data.actions
            rew_targets = rew_data.rewards.squeeze(-1) # Target is the scalar reward

            # Split data
            indices = torch.randperm(len(rew_obs), device=device)
            split = int(len(rew_obs) * (1 - args.model_val_ratio))
            train_idx, val_idx = indices[:split], indices[split:]

            train_dataset = TensorDataset(rew_obs[train_idx], rew_acts[train_idx], rew_targets[train_idx])
            val_dataset = TensorDataset(rew_obs[val_idx], rew_acts[val_idx], rew_targets[val_idx])
            train_loader = DataLoader(train_dataset, batch_size=args.model_train_batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.model_train_batch_size)

            print(f"Reward Model: Train size={len(train_idx)}, Val size={len(val_idx)}")
            best_val_loss = float('inf')
            patience_counter = 0
            reward_model_accurate = False # Reset accuracy flag

            for epoch in range(args.model_max_epochs):
                train_loss = train_model_epoch(reward_model, reward_optimizer, train_loader, device, writer, epoch, global_step, "reward", False)
                val_loss, val_metrics = validate_model(reward_model, val_loader, device, writer, global_step + epoch + 1, "reward", False)

                print(f"  Epoch {epoch+1}/{args.model_max_epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

                # Early stopping check
                if val_loss < best_val_loss - args.model_val_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if best_val_loss <= args.reward_train_threshold:
                        reward_model_accurate = True
                        print(f"  Reward model reached accuracy threshold ({best_val_loss:.6f} <= {args.reward_train_threshold}).")
                        # Optionally break early
                        # break
                else:
                    patience_counter += 1
                    if patience_counter >= args.model_val_patience:
                        print(f"  Early stopping triggered at epoch {epoch+1}.")
                        break

            reward_model_accurate = (best_val_loss <= args.reward_train_threshold)
            print(f"Reward Model Training Complete. Best Val Loss: {best_val_loss:.6f}. Accurate: {reward_model_accurate}")
            writer.add_scalar("charts/reward_model_accurate", float(reward_model_accurate), global_step)
            print(f"--- Model Check/Training Finished ---")


        # --- Agent Training ---
        if global_step > args.learning_starts:
            # Skip agent update if models are not deemed accurate
            if not dynamic_model_accurate or not reward_model_accurate:
                if global_step % 100 == 0: # Log occasionally
                     print(f"Skipping agent update at step {global_step} (Dyn accurate: {dynamic_model_accurate}, Rew accurate: {reward_model_accurate})")
                continue

            # Sample batch for agent update
            data = rb.sample(args.batch_size, env=None) # Use args.batch_size here
            mb_obs = data.observations
            mb_obs.requires_grad_(True) # Enable gradient tracking for observations

            # --- Critic Update ---
            # Calculate dV/dx using the EMA critic for stability
            # Need to wrap the EMA model call for grad
            ema_critic_eval = lambda x: ema_critic(x)
            compute_value_grad = grad(ema_critic_eval)
            # Use vmap for batch processing
            dVdx = vmap(compute_value_grad)(mb_obs) # Shape: (batch_size, obs_dim)

            # Get actions from the current (non-EMA) actor for the actor loss calculation later
            # but use EMA actor actions for HJB residual calculation for consistency with dVdx source
            with torch.no_grad():
                 current_actions_ema = ema_actor(mb_obs)

            # Predict dynamics f(x, pi(x)) using the learned dynamic model's ODE function
            # Note: We need the *function* f, not the integrated next state
            f = dynamic_model.ode_func(
                torch.tensor(0.0, device=device), # t=0
                mb_obs,
                current_actions_ema # Use EMA actions consistent with dVdx source
            ) # Shape: (batch_size, obs_dim)

            # Predict reward r(x, pi(x)) using the learned reward model
            r = reward_model(mb_obs, current_actions_ema) # Shape: (batch_size,)

            # Calculate the Hamiltonian H(x, pi(x), dV/dx) = r(x, pi(x)) + dV/dx^T * f(x, pi(x))
            # Use torch.einsum for robust batch dot product
            hamiltonian = r + torch.einsum("bi,bi->b", dVdx, f) # Shape: (batch_size,)

            # Calculate the HJB residual: H - rho * V(x)
            # Use the *current* critic (not EMA) for the V(x) term, as this is what we are optimizing
            current_v = critic(mb_obs) # Shape: (batch_size,)
            hjb_residual = hamiltonian - rho * current_v # Shape: (batch_size,)

            # Critic loss: 0.5 * MSE of the HJB residual
            critic_loss = 0.5 * (hjb_residual ** 2).mean()

            # Optimize the critic
            critic_optimizer.zero_grad()
            critic_loss.backward()
            if args.grad_norm_clip is not None:
                nn.utils.clip_grad_norm_(critic.parameters(), args.grad_norm_clip)
            critic_optimizer.step()

            # Update EMA Critic
            ema_critic.update([critic]) # Pass as list


            # --- Actor Update (Delayed) ---
            if global_step % args.policy_frequency == 0:
                # Actor loss aims to maximize the Hamiltonian H(x, pi(x), dV/dx)
                # We need gradients through the actor's actions pi(x) used in H.
                # Recalculate H with actions from the *current* actor.
                # dVdx was calculated using EMA critic, treat it as constant for actor update.

                # Detach dVdx as we don't want gradients flowing back to the critic from the actor loss
                dVdx_detached = dVdx.detach()

                # Get actions from the *current* actor
                current_actions_actor = actor(mb_obs)

                # Recalculate f and r with current actor's actions
                f_actor = dynamic_model.ode_func(
                    torch.tensor(0.0, device=device),
                    mb_obs, # mb_obs still requires grad here
                    current_actions_actor
                )
                r_actor = reward_model(mb_obs, current_actions_actor)

                # Recalculate Hamiltonian using current actor's actions
                hamiltonian_actor = r_actor + torch.einsum("bi,bi->b", dVdx_detached, f_actor)

                # Actor loss is the negative mean Hamiltonian (maximization)
                actor_loss = (-hamiltonian_actor).mean()

                # Optimize the actor
                actor_optimizer.zero_grad()
                # Need retain_graph=True if mb_obs.grad is needed elsewhere, but likely not here.
                # If critic backward pass already used mb_obs grad, need to be careful.
                # Let's clear grad first.
                if mb_obs.grad is not None:
                    mb_obs.grad.zero_()
                actor_loss.backward()
                if args.grad_norm_clip is not None:
                    nn.utils.clip_grad_norm_(actor.parameters(), args.grad_norm_clip)
                actor_optimizer.step()

                # Update EMA Actor
                ema_actor.update([actor]) # Pass as list

                # Logging Actor/Critic Updates
                writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/critic_values", current_v.mean().item(), global_step)
                writer.add_scalar("metrics/hamiltonian", hamiltonian.mean().item(), global_step) # Log Hamiltonian from critic step
                writer.add_scalar("metrics/hjb_residual", hjb_residual.mean().item(), global_step)


            # Log SPS
            if global_step % 100 == 0: # Log less frequently
                sps = int(global_step / (time.time() - start_time))
                print(f"SPS: {sps}")
                writer.add_scalar("charts/SPS", sps, global_step)


    # --- End of Training ---

    # Use EMA models for final saving and evaluation
    # Update the original models with EMA weights
    # Note: AveragedModel doesn't have a direct 'load_state_dict' equivalent for the base model.
    # We need to manually copy parameters or use the internal module.
    # Using deepcopy might be safer if EMA model is used elsewhere.
    import copy
    actor_final = copy.deepcopy(ema_actor.module).to('cpu')
    critic_final = copy.deepcopy(ema_critic.module).to('cpu')


    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        # Save the EMA model weights
        torch.save((actor_final.state_dict(), critic_final.state_dict()), model_path)
        print(f"EMA model saved to {model_path}")

        # Evaluation using the saved EMA model
        from cleanrl_utils.evals.ddpg_eval import evaluate # Assuming this eval script works

        # Need to modify evaluate to load the correct model types (HJBActor, HJBCritic)
        # and potentially handle normalization if the eval env needs it.
        print("Starting evaluation...")
        # Create a non-normalized env for evaluation usually
        def make_eval_env(env_id, seed, idx, capture_video, run_name):
            def thunk():
                 env = gym.make(env_id)
                 env = gym.wrappers.RecordEpisodeStatistics(env)
                 # NO normalization for eval usually, unless model expects it
                 env.action_space.seed(seed)
                 return env, 0.0 # dt not needed for standard eval
            return thunk

        episodic_returns = evaluate(
            model_path,
            make_eval_env, # Use eval env maker
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(HJBActor, HJBCritic), # Pass correct model classes
            device="cpu", # Evaluate on CPU
            exploration_noise=0, # No exploration noise during evaluation
            norm_obs=args.capture_video # Hack: Use capture_video flag to signal if obs were normalized during training
                                        # The evaluate script needs modification to handle this
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            # Assuming push_to_hub can handle the model tuple format
            push_to_hub(args, episodic_returns, repo_id, "HJB", f"runs/{run_name}", f"videos/{run_name}-eval") # Algo name "HJB"

    envs.close()
    writer.close()
    print("Training finished.")

    # Note: The following lines were part of the original __main__ block but are now handled earlier.
    # args = tyro.cli(Args) # Handled earlier
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}" # Handled earlier
    # if args.track: # Handled earlier
    #     import wandb # Handled earlier
    #     wandb.init(...) # Handled earlier
    # writer = SummaryWriter(f"runs/{run_name}") # Handled earlier
    # writer.add_text(...) # Handled earlier

    # TRY NOT TO MODIFY: seeding (This block seems duplicated, already handled earlier)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    # TRY NOT TO MODIFY: seeding (This block seems duplicated, already handled earlier)
    # random.seed(args.seed) # Handled earlier
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
