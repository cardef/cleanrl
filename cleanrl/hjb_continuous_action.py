# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy

"""
Experiment Script: Model-Based Actor-Critic with Continuous-Time HJB Objective

This script implements a Reinforcement Learning agent for continuous control tasks
based on an Actor-Critic framework. It deviates significantly from standard DDPG/TD3
by incorporating:

1.  **Learned World Models:**
    * A `DynamicModel` using a Neural Ordinary Differential Equation (Neural ODE)
        via the `torchode` library to learn the continuous-time system dynamics:
        dx/dt = f(x, a; theta_f).
    * A `RewardModel` using a standard MLP to learn the reward function:
        r = r(x, a; theta_r).
2.  **Hamilton-Jacobi-Bellman (HJB) Inspired Critic Loss:**
    * The critic (Value function V(x)) is trained to satisfy the HJB equation
        in continuous time: rho * V(x) = max_a [ r(x, a) + dV/dx * f(x, a) ].
    * The loss minimizes the residual of this equation, potentially including
        a viscosity term (Laplacian of V) for regularization.
    * Terminal states are handled separately, enforcing V(x_terminal) = 0.
3.  **Model-Based Actor Update:**
    * The actor (Policy pi(x)) is trained to maximize the Hamiltonian:
        H(x, pi(x), dV/dx) = r(x, pi(x)) + dV/dx * f(x, pi(x)).
4.  **Model Accuracy Gating:**
    * Agent training steps (actor and critic updates) are only performed when
        the learned dynamic and reward models demonstrate sufficient accuracy on
        a validation set, measured by MSE against thresholds.
5.  **EMA Target Networks:**
    * Exponential Moving Average (EMA) models (`torch.optim.swa_utils.AveragedModel`)
        are used to provide stable targets for both actor and critic updates.
6.  **Noise Annealing:**
    * Exploration noise added to the actor's output is linearly annealed over
        a portion of the training steps.

This approach aims to potentially improve sample efficiency and stability by
leveraging learned models of the environment within a continuous-time control framework.
"""
import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import vmap
from torch.func import grad, hessian
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
    viscosity_coeff: float = 0.0
    """Coefficient for the viscosity regularization term (Laplacian of V) in the critic loss."""
    env_id: str = "InvertedPendulum-v4"
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
    policy_frequency: int = 20 # Standard DDPG/TD3 delayed policy update frequency
    """Frequency (in agent updates) to update the policy (actor) network. Critic updated more often."""
    ema_decay: float = 0.0 # EMA decay rate for target networks
    """Exponential Moving Average (EMA) decay rate for target networks"""

    # Exploration noise annealing parameters
    exploration_noise_start: float = 0.1
    """Initial scale of the Gaussian exploration noise (relative to action range)."""
    exploration_noise_end: float = 0.1
    """Final scale of the Gaussian exploration noise after annealing."""
    exploration_noise_anneal_fraction: float = 0.8
    """Fraction of `total_timesteps` over which to linearly anneal the exploration noise scale."""

    # Model Training specific arguments
    model_train_freq: int = 1000
    """Frequency (in environment steps) to check model accuracy and potentially retrain dynamics and reward models."""
    model_dataset_size: int = 50_000
    """Number of samples drawn from the replay buffer for training/validating models."""
    dynamic_train_threshold: float = 0.001
    """MSE validation loss threshold below which the dynamic model is considered 'accurate'."""
    reward_train_threshold: float = 0.001
    """MSE validation loss threshold below which the reward model is considered 'accurate'."""
    model_val_ratio: float = 0.2
    """Fraction of the `model_dataset_size` used for validation during model training."""
    model_val_patience: int = 10
    """Number of epochs without improvement on validation loss before early stopping model training."""
    model_val_delta: float = 1e-4
    """Minimum improvement in validation loss required to reset the early stopping patience counter."""
    model_max_epochs: int = 50
    """Maximum number of epochs to train each model per training cycle."""
    model_train_batch_size: int = 256
    """Minibatch size used during the epochs of model training."""
    grad_norm_clip: Optional[float] = 0.5 # Gradient clipping for actor/critic
    """gradient norm clipping threshold (None for no clipping)"""
    terminal_coeff: float = 1e0
    """Weighting coefficient for the loss ensuring V(terminal state) = 0."""


def make_env(env_id, seed, idx, capture_video, run_name):
    """Creates a Gymnasium environment with specified wrappers."""
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        

        return env

    return thunk


# --- Model Definitions ---

class ODEFunc(nn.Module):
    """
    Defines the function f(t, x, a) for the Neural ODE: dx/dt = f(t, x, a).
    This network predicts the instantaneous rate of change of the state.

    Args:
        obs_dim: Dimensionality of the observation space.
        action_dim: Dimensionality of the action space.
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.SiLU(), # Changed activation
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, obs_dim),
        )

    def forward(self, t, x, a):
        """
        Computes the state derivative dx/dt.

        Args:
            t: Current time (scalar or batched). Unused in autonomous systems.
            x: Current state(s) (batch_size, obs_dim).
            a: Current action(s) (batch_size, action_dim).

        Returns:
            The predicted state derivative(s) (batch_size, obs_dim).
        """
        x_float = x.float()
        a_float = a.float()
        return self.net(torch.cat([x_float, a_float], dim=-1))

class DynamicModel(nn.Module):
    """
    Predicts the next state by integrating the learned ODE dynamics over one time step dt.

    Args:
        obs_dim: Dimensionality of the observation space.
        action_dim: Dimensionality of the action space.
        dt: The time step duration of the environment.
        device: The torch device (e.g., 'cpu', 'cuda').
    """
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
        """
        Predicts the state after one time step dt using the Neural ODE.

        Args:
            initial_obs: The starting observation(s) (batch_size, obs_dim).
            actions: The action(s) applied during the time step (batch_size, action_dim).
                       Assumed constant over the interval [0, dt].

        Returns:
            The predicted next observation(s) (batch_size, obs_dim).
        """
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
            nn.Linear(256, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        return self.net(x).squeeze(-1) # Output shape: (batch_size,)

# --- Agent Network Definitions ---

class HJBCritic(nn.Module):
    """
    The Value Function V(x) approximator (Critic).
    Uses a clipped double-Q approach by maintaining two separate critic networks
    to mitigate overestimation bias, although here they approximate V(x) directly.

    Args:
        env: The Gymnasium environment instance (used to get observation dim).
    """
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.observation_space.shape).prod()
        self.critic1 = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )
        self.critic2 = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, critic_net=1):
        """Forward pass through specified critic network (1 or 2)."""
        if critic_net == 1:
            return self.critic1(x).squeeze()
        return self.critic2(x).squeeze()

class HJBActor(nn.Module):
    """
    The Policy Function pi(x) approximator (Actor).
    Outputs a deterministic action for a given state.

    Args:
        env: The Gymnasium environment instance (used to get observation/action dims and bounds).
    """
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
    """
    Trains a model (Dynamic or Reward) for one epoch using MSE loss.

    Args:
        model: The model (DynamicModel or RewardModel) to train.
        optimizer: The optimizer for the model.
        train_loader: DataLoader providing training batches (obs, actions, targets).
        device: The torch device.
        writer: TensorBoard SummaryWriter for logging.
        epoch: The current training epoch number.
        global_step: The current total environment steps (for logging).
        model_name: String identifier for logging ('dynamic' or 'reward').
        is_dynamic_model: Boolean indicating if it's the DynamicModel (affects forward pass signature).
        grad_norm_clip_model: Optional max gradient norm for model training.

    Returns:
        The average training loss for the epoch.
    """
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
    """
    Validates a model (Dynamic or Reward) on the validation dataset.

    Args:
        model: The model (DynamicModel or RewardModel) to validate.
        val_loader: DataLoader providing validation batches (obs, actions, targets).
        device: The torch device.
        writer: TensorBoard SummaryWriter for logging.
        global_step: The current total environment steps (for logging).
        model_name: String identifier for logging ('dynamic' or 'reward').
        is_dynamic_model: Boolean indicating if it's the DynamicModel.

    Returns:
        A tuple containing:
        - val_loss (float): The average validation MSE loss.
        - val_metrics (Dict[str, float]): Dictionary of validation metrics (MSE, MAE, R2).
    """
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
    import stable_baselines3 as sb3
    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
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
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    # It's essential to get the dt after all wrappers that might affect it
    try:
        # For MuJoCo environments
        env_dt = envs.get_attr("dt")[0]
    except AttributeError:
        # Fallback for other environments, may need adjustment
        print("Warning: Could not automatically determine environment dt. Using default 0.05.")
        env_dt = 0.05 # A common default, but verify for your specific env_id!
    # Agent setup
    actor = HJBActor(envs).to(device)
    critic = HJBCritic(envs).to(device)
    actor_optimizer = optim.AdamW(actor.parameters(), lr=args.learning_rate)
    critic1_optimizer = optim.AdamW(critic.critic1.parameters(), lr=args.learning_rate)
    critic2_optimizer = optim.AdamW(critic.critic2.parameters(), lr=args.learning_rate)

    # EMA models for stability
    ema_actor = torch.optim.swa_utils.AveragedModel(
        actor,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema_decay)
    )
    ema_critic1 = torch.optim.swa_utils.AveragedModel(
        critic.critic1,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema_decay)
    )
    ema_critic2 = torch.optim.swa_utils.AveragedModel(
        critic.critic2,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema_decay)
    )

    # Dynamic and Reward Model setup
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    action_dim = np.prod(envs.single_action_space.shape)
    dynamic_model = DynamicModel(obs_dim, action_dim, env_dt, device).to(device)
    reward_model = RewardModel(obs_dim, action_dim).to(device)
    dynamic_optimizer = optim.AdamW(dynamic_model.parameters(), lr=args.learning_rate)
    reward_optimizer = optim.AdamW(reward_model.parameters(), lr=args.learning_rate)
    envs.single_observation_space.dtype = np.float32
    # Replay buffer
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False, # Important for model learning
    )

    # Continuous-time discount rate (rho) based on discrete gamma and dt
    # gamma = exp(-rho * dt) => rho = -log(gamma) / dt
    rho = -torch.log(torch.tensor(args.gamma, device=device))
    print(f"Continuous discount rate (rho): {rho.item()}")

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    # Ensure initial obs is float32
    obs = obs.astype(np.float32)
    dynamic_model_accurate = False
    reward_model_accurate = False

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                # Calculate annealed noise scale
                if args.exploration_noise_anneal_fraction > 0:
                    anneal_steps = int(args.total_timesteps * args.exploration_noise_anneal_fraction)
                    current_noise_scale = args.exploration_noise_end + (
                        args.exploration_noise_start - args.exploration_noise_end
                    ) * (1 - min(global_step / anneal_steps, 1))
                else:
                    current_noise_scale = args.exploration_noise_start
                    
                actions = ema_actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * current_noise_scale)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

            # Log the current noise scale
            writer.add_scalar("charts/exploration_noise_scale", current_noise_scale, global_step)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
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
                            break
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
                        break
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
            with torch.no_grad():
                current_actions_ema = ema_actor(mb_obs)

            # 1. Create Masks for terminal/non-terminal states
            terminations_mask = data.dones.squeeze(-1).bool()
            non_terminations_mask = ~terminations_mask

            # Calculate critic values for all states
            all_current_v1 = critic(mb_obs, critic_net=1) 
            all_current_v2 = critic(mb_obs, critic_net=2)

            # --- A. Terminal State Loss ---
            v1_at_termination = all_current_v1[terminations_mask]
            v2_at_termination = all_current_v2[terminations_mask]
            terminal_critic_loss = torch.tensor(0.0, device=device)
            if v1_at_termination.numel() > 0:
                terminal_loss1 = F.mse_loss(v1_at_termination, torch.zeros_like(v1_at_termination))
                terminal_loss2 = F.mse_loss(v2_at_termination, torch.zeros_like(v2_at_termination))
                terminal_critic_loss = terminal_loss1 + terminal_loss2

            # --- B. Non-Terminal State Loss --- 
            hjb_loss1_non_term = torch.tensor(0.0, device=device)
            hjb_loss2_non_term = torch.tensor(0.0, device=device)

            if non_terminations_mask.any():
                obs_non_term = mb_obs[non_terminations_mask]
                actions_ema_non_term = current_actions_ema[non_terminations_mask]
                
                # Get minimum V for non-terminal states
                v1_non_term = all_current_v1[non_terminations_mask]
                v2_non_term = all_current_v2[non_terminations_mask]
                min_v_non_term = torch.min(v1_non_term, v2_non_term)
                
                # Calculate dynamics and reward
                f_non_term = dynamic_model.ode_func(
                    torch.tensor(0.0, device=device), 
                    obs_non_term,
                    actions_ema_non_term
                )
                r_non_term = reward_model(obs_non_term, actions_ema_non_term)
                
                # Compute gradients for non-terminal states
                compute_value_grad1 = grad(lambda x: critic(x, critic_net=1).sum())
                compute_value_grad2 = grad(lambda x: critic(x, critic_net=2).sum())
                dVdx1_non_term = vmap(compute_value_grad1)(obs_non_term)
                dVdx2_non_term = vmap(compute_value_grad2)(obs_non_term)

                # Viscosity regularization
                if args.viscosity_coeff > 0:
                    hessians1 = vmap(hessian(lambda x: critic(x, critic_net=1)))(obs_non_term)
                    laplacians1 = torch.einsum('bii->b', hessians1)
                    hessians2 = vmap(hessian(lambda x: critic(x, critic_net=2)))(obs_non_term)
                    laplacians2 = torch.einsum('bii->b', hessians2)
                else:
                    # Create zero tensors with appropriate shape to avoid None errors
                    batch_size_non_term = obs_non_term.shape[0]
                    laplacians1 = torch.zeros(batch_size_non_term, device=device)
                    laplacians2 = torch.zeros(batch_size_non_term, device=device)

                # HJB residuals
                hjb_residual1 = (r_non_term + torch.einsum("bi,bi->b", dVdx1_non_term, f_non_term)) - rho * min_v_non_term - args.viscosity_coeff * laplacians1
                hjb_residual2 = (r_non_term + torch.einsum("bi,bi->b", dVdx2_non_term, f_non_term)) - rho * min_v_non_term - args.viscosity_coeff * laplacians2
                
                
                
                hjb_loss1_non_term = 0.5 * (hjb_residual1**2).mean()
                hjb_loss2_non_term = 0.5 * (hjb_residual2**2).mean()

            # --- C. Total Critic Loss ---
            critic_loss = hjb_loss1_non_term + hjb_loss2_non_term + args.terminal_coeff * terminal_critic_loss

            # Optimize both critics
            critic1_optimizer.zero_grad()
            critic2_optimizer.zero_grad()
            critic_loss.backward()
            if args.grad_norm_clip is not None:
                nn.utils.clip_grad_norm_(critic.critic1.parameters(), args.grad_norm_clip)
                nn.utils.clip_grad_norm_(critic.critic2.parameters(), args.grad_norm_clip)
            critic1_optimizer.step()
            critic2_optimizer.step()

            # Update EMA critics
            ema_critic1.update_parameters(critic.critic1)
            ema_critic2.update_parameters(critic.critic2)


            # --- Actor Update (Delayed) ---
            if global_step % args.policy_frequency == 0:
                # Actor loss aims to maximize the Hamiltonian H(x, pi(x), dV/dx)
                # We need gradients through the actor's actions pi(x) used in H.
                # Recalculate H with actions from the *current* actor.
                # dVdx was calculated using EMA critic, treat it as constant for actor update.

                # Use critic1's gradients for actor update
                with torch.no_grad():
                    compute_value_grad1 = grad(lambda x: ema_critic1(x).squeeze())
                    dVdx1 = vmap(compute_value_grad1)(mb_obs)

                current_actions_actor = actor(mb_obs)
                f_actor = dynamic_model.ode_func(torch.tensor(0.0, device=device), mb_obs, current_actions_actor)
                r_actor = reward_model(mb_obs, current_actions_actor)

                hamiltonian_actor = r_actor + torch.einsum("bi,bi->b", dVdx1, f_actor)

                # Create mask for transitions where next state is NON-terminal (same as critic)
                non_terminal_mask = ~data.dones.squeeze(-1).bool()
                hamiltonian_non_term = hamiltonian_actor[non_terminal_mask]

                # Skip update if no non-terminal transitions in batch
                if hamiltonian_non_term.numel() == 0:
                    print(f"Skipping actor update at step {global_step} - no non-terminal transitions")
                    continue

                # Actor loss is the negative mean Hamiltonian (maximization)
                actor_loss = (-hamiltonian_non_term).mean()

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
                ema_actor.update_parameters(actor)

                # Logging Actor/Critic Updates
                writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
                writer.add_scalar("losses/critic_terminal_loss", terminal_critic_loss.item(), global_step)
                writer.add_scalar("losses/critic1_hjb_non_term", hjb_loss1_non_term.item(), global_step)
                writer.add_scalar("losses/critic2_hjb_non_term", hjb_loss2_non_term.item(), global_step)
                writer.add_scalar("losses/actor_non_term", actor_loss.item(), global_step)
                writer.add_scalar("metrics/critic1_value", all_current_v1.mean().item(), global_step)
                writer.add_scalar("metrics/critic2_value", all_current_v2.mean().item(), global_step)
                writer.add_scalar("metrics/hamiltonian", hamiltonian_actor.mean().item(), global_step)


            # Log SPS and exploration noise
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
    # Create final critic by combining both EMA critics
    critic_final = HJBCritic(envs).to('cpu')
    critic_final.critic1.load_state_dict(ema_critic1.module.state_dict())
    critic_final.critic2.load_state_dict(ema_critic2.module.state_dict())


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
            norm_obs=False # No observation normalization
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
