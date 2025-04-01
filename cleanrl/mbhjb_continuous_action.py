# Full PPO+MBPO Code with TorchODE Dynamics Model, Reward Model,
# VecNormalize, Raw Buffer Storage, Validation, Early Stopping,
# Modified GAE, Simplified HJB (vmap/grad), R2 Logging
# Formatted to avoid multiple statements per line separated by semicolons.

import os
import random
import time
import math
import copy  # For deepcopy in early stopping
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# Required imports
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecEnv, VecNormalize, DummyVecEnv

try:
    from torch.func import grad, vmap

    print("Imported grad, vmap from torch.func")
    TORCH_FUNC_AVAILABLE = True
except ImportError:
    print(
        "WARNING: torch.func not available (requires PyTorch >= 1.13/2.0 with functorch). HJB residual calculation will be skipped."
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
    exp_name: str = (
        os.path.basename(__file__)[: -len(".py")] + "_mbpo_torchode_full_fmt"
    )  # Changed name slightly
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
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4  # Policy LR
    """the learning rate of the policy optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048  # Real steps per iteration
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32  # PPO minibatches
    """the number of mini-batches"""
    update_epochs: int = 10  # PPO update epochs
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # MBPO Dynamics/Reward Model and Training Args
    dynamics_learning_rate: float = 1e-3
    reward_learning_rate: float = 1e-3
    replay_buffer_size: int = 1_000_000
    model_train_batch_size: int = 256
    model_train_freq: int = 250
    model_train_epochs: int = 1000  # Max epochs per training phase
    model_rollout_freq: int = 1000
    model_rollout_length: int = 5
    model_rollout_batch_size: int = 4096  # Target size
    num_model_rollout_starts: int = field(init=False)
    use_model_for_updates: bool = True

    # Model Validation Args
    model_validation_batch_size: int = 1024
    model_state_accuracy_threshold: float = 1.5
    model_early_stopping_patience: int = 50
    model_validation_freq: int = 1

    # HJB Residual Args
    hjb_coef: float = 0.5
    use_hjb_loss: bool = True

    # Environment dt
    env_dt: float = 0.05

    # to be filled in runtime
    minibatch_size: int = 0
    num_iterations: int = 0
    rho: float = field(init=False)


# --- Environment Creation ---
def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        render_mode = "rgb_array" if capture_video and idx == 0 else None
        try:
            env = gym.make(env_id, render_mode=render_mode)
        except Exception as e:
            print(
                f"Warning: Failed to set render_mode='{render_mode}'. Error: {e}. Trying default."
            )
            env = gym.make(env_id)

        if capture_video and idx == 0 and env.render_mode == "rgb_array":
            print(f"Capturing video for env {idx} to videos/{run_name}")
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}", episode_trigger=lambda x: x % 50 == 0
            )

        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env

    return thunk


# --- Utilities ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
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
            layer_init(nn.Linear(hidden_size, obs_dim)),
        )
        print(
            f"Initialized ODEFunc: Input {obs_dim + action_dim}, Output {obs_dim} (state derivative)"
        )

    def forward(self, t, x, a):
        return self.net(torch.cat([x.float(), a.float()], dim=-1))


class DynamicModel(nn.Module):
    def __init__(self, obs_dim, action_dim, dt: float, device: torch.device):
        super().__init__()
        if not TORCHODE_AVAILABLE:
            raise ImportError("torchode not found.")
        self.ode_func = ODEFunc(obs_dim, action_dim)
        self.dt = dt
        self.device = device
        self.term = to.ODETerm(self.ode_func, with_args=True)
        self.step_method = to.Tsit5(term=self.term)
        self.step_size_controller = to.FixedStepController()
        self.adjoint = to.AutoDiffAdjoint(
            step_method=self.step_method, step_size_controller=self.step_size_controller
        )
        print(f"Initialized DynamicModel using TorchODE (dt={self.dt})")

    def forward(self, initial_obs_norm, actions_norm):
        batch_size = initial_obs_norm.shape[0]
        dt0 = torch.full((batch_size,), self.dt, device=self.device)
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
        print(f"Initialized RewardModel: Input {obs_dim + action_dim}, Output 1")

    def forward(self, obs_norm, action):
        return self.net(torch.cat([obs_norm.float(), action.float()], dim=-1))


# --- Agent (Policy and Value Function) ---
class Agent(nn.Module):
    def __init__(self, envs: VecEnv):
        super().__init__()
        obs_dim = np.array(envs.observation_space.shape).prod()
        action_dim = np.prod(envs.action_space.shape)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x_norm):
        return self.critic(x_norm)

    def get_action_and_value(self, x_norm, action=None):
        action_mean = self.actor_mean(x_norm)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.critic(x_norm)
        return action, log_prob, entropy, value


# --- GAE Calculation (Simplified for Rollouts) ---
def compute_gae_for_rollout(rewards, values, dones, gamma, gae_lambda, next_value):
    """Computes GAE for a rollout where dones are typically 0."""
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - dones[t] # Should be 1.0 for model data
            nextvalues = next_value.squeeze() # Use the provided final next value
        else:
            nextnonterminal = 1.0 - dones[t+1] # Should be 1.0
            nextvalues = values[t+1]

        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values
    return advantages, returns


# --- GAE Calculation (Real Data - handles truncation) ---
def compute_gae_real_data(
    real_rewards_storage_norm,
    real_values_storage_norm,
    real_dones_storage,
    next_obs_norm,
    next_done,
    real_infos_storage,
    agent,
    gamma,
    gae_lambda,
    norm_envs,
    device,
):
    num_steps = real_rewards_storage_norm.shape[0]
    num_envs = real_rewards_storage_norm.shape[1]
    advantages = torch.zeros_like(real_rewards_storage_norm).to(device)
    lastgaelam = torch.zeros(num_envs).to(device)
    with torch.no_grad():
        next_value_norm = agent.get_value(next_obs_norm).flatten()
        last_infos = real_infos_storage[-1]  # Infos from the last step

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal_t = torch.zeros_like(next_done).float()
            nextvalues_t = torch.zeros_like(next_value_norm).float()
            for i in range(num_envs):
                is_done = next_done[i].item() > 0.5
                info = last_infos[i] if i < len(last_infos) else {}
                is_truncated = info.get("TimeLimit.truncated", False) if info else False
                if is_done and is_truncated:
                    final_obs_raw = info.get("terminal_observation")
                    if final_obs_raw is not None:
                        final_obs_norm_np = norm_envs.normalize_obs(final_obs_raw)
                        final_obs_tensor = (
                            torch.tensor(final_obs_norm_np, dtype=torch.float32)
                            .unsqueeze(0)
                            .to(device)
                        )
                        final_value = agent.get_value(final_obs_tensor).item()
                        nextnonterminal_t[i] = 1.0
                        nextvalues_t[i] = final_value
                    else:
                        nextnonterminal_t[i] = 0.0
                        nextvalues_t[i] = 0.0
                elif is_done and not is_truncated:
                    nextnonterminal_t[i] = 0.0
                    nextvalues_t[i] = 0.0
                else:
                    nextnonterminal_t[i] = 1.0
                    nextvalues_t[i] = next_value_norm[i]
            nextnonterminal = nextnonterminal_t
            nextvalues = nextvalues_t
        else:
            nextnonterminal = 1.0 - real_dones_storage[t + 1]
            nextvalues = real_values_storage[t + 1]

        delta = (
            real_rewards_storage_norm[t]
            + gamma * nextvalues * nextnonterminal
            - real_values_storage_norm[t]
        )
        advantages[t] = lastgaelam = (
            delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        )

    returns_norm = advantages + real_values_storage_norm
    return advantages, returns_norm


# --- GAE Calculation (Model Data - simpler) ---
def compute_gae_model_data(
    model_rewards_storage_norm,
    model_values_storage_norm,
    gamma,
    gae_lambda,
    model_final_next_values_storage,
):
    dones = torch.zeros_like(model_rewards_storage_norm).to(
        model_rewards_storage_norm.device
    )
    advantages, returns_norm = compute_gae_for_rollout(
        model_rewards_storage_norm,
        model_values_storage_norm,
        dones,
        gamma,
        gae_lambda,
        model_final_next_values_storage,
    )
    return advantages, returns_norm


# --- R2 Score Calculation ---
def calculate_r2_score(y_true, y_pred, epsilon=1e-8):
    ssr = torch.sum((y_true - y_pred) ** 2)
    mean_true = torch.mean(y_true, dim=0, keepdim=True)
    sst = torch.sum((y_true - mean_true) ** 2)
    r2 = 1.0 - (ssr / (sst + epsilon))
    return r2


# --- Main Execution ---
if __name__ == "__main__":
    if not TORCHODE_AVAILABLE:
        exit()
    args = tyro.cli(Args)

    # Calculate dependent args
    args.num_model_rollout_starts = (
        args.model_rollout_batch_size // args.model_rollout_length
    )
    args.num_iterations = args.total_timesteps // (args.num_envs * args.num_steps)
    args.rho = (
        -math.log(args.gamma) if args.gamma > 0 and args.gamma < 1 else 0.0
    )  # Avoid log(1) or log(0)

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
            print("WandB tracking enabled.")
        except ImportError:
            print("WARNING: wandb not installed. Tracking disabled.")
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

    # --- Environment Setup ---
    print("Setting up environment...")
    envs = DummyVecEnv(
        [
            make_env(args.env_id, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ]
    )
    norm_envs = VecNormalize(
        envs,
        gamma=args.gamma,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    print("VecNormalize enabled.")
    assert isinstance(
        norm_envs.action_space, gym.spaces.Box
    ), "only continuous action space is supported"
    obs_space = norm_envs.observation_space
    action_space = norm_envs.action_space
    obs_shape = obs_space.shape
    action_shape = action_space.shape

    # --- Agent, Models, Optimizers ---
    agent = Agent(norm_envs).to(device)
    dynamic_model = DynamicModel(obs_shape[0], action_shape[0], args.env_dt, device).to(
        device
    )
    reward_model = RewardModel(obs_shape[0], action_shape[0]).to(device)
    policy_optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    dynamics_optimizer = optim.Adam(
        dynamic_model.ode_func.parameters(), lr=args.dynamics_learning_rate
    )
    reward_optimizer = optim.Adam(
        reward_model.parameters(), lr=args.reward_learning_rate
    )

    # --- Replay Buffer for RAW Data ---
    raw_obs_space = envs.observation_space  # Use .observation_space for DummyVecEnv
    raw_action_space = envs.action_space
    print(
        f"Replay buffer using space definition from DummyVecEnv attributes. Shape: {raw_obs_space.shape}, Dtype: {raw_obs_space.dtype}"
    )
    sb3_buffer_device = "cpu"
    real_buffer = ReplayBuffer(
        args.replay_buffer_size,
        raw_obs_space,
        raw_action_space,
        device=sb3_buffer_device,
        n_envs=args.num_envs,
        handle_timeout_termination=True,
    )
    print("Replay buffer configured with handle_timeout_termination=True")

    # --- Temporary Storage (Per Iteration, using NORMALIZED shapes) ---
    real_obs_storage = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(
        device
    )
    real_actions_storage = torch.zeros(
        (args.num_steps, args.num_envs) + action_shape
    ).to(device)
    real_logprobs_storage = torch.zeros((args.num_steps, args.num_envs)).to(device)
    real_rewards_storage = torch.zeros((args.num_steps, args.num_envs)).to(
        device
    )  # For GAE (normalized)
    real_dones_storage = torch.zeros((args.num_steps, args.num_envs)).to(
        device
    )  # Done = term or trunc
    real_values_storage = torch.zeros((args.num_steps, args.num_envs)).to(device)
    real_single_rewards_storage = torch.zeros((args.num_steps, args.num_envs)).to(
        device
    )  # For HJB (normalized)
    real_infos_storage = [
        {} for _ in range(args.num_steps)
    ]  # Store infos list per step

    model_obs_storage = torch.zeros(
        (args.model_rollout_length, args.num_model_rollout_starts) + obs_shape
    ).to(device)
    model_actions_storage = torch.zeros(
        (args.model_rollout_length, args.num_model_rollout_starts) + action_shape
    ).to(device)
    model_rewards_storage = torch.zeros(
        (args.model_rollout_length, args.num_model_rollout_starts)
    ).to(
        device
    )  # For GAE (predicted norm reward)
    model_single_rewards_storage = torch.zeros(
        (args.model_rollout_length, args.num_model_rollout_starts, 1)
    ).to(
        device
    )  # For HJB (predicted norm reward)
    model_logprobs_storage = torch.zeros(
        (args.model_rollout_length, args.num_model_rollout_starts)
    ).to(device)
    model_values_storage = torch.zeros(
        (args.model_rollout_length, args.num_model_rollout_starts)
    ).to(device)
    model_final_next_values_storage = torch.zeros(
        (1, args.num_model_rollout_starts)
    ).to(device)

    # --- Runtime State ---
    global_step = 0
    start_time = time.time()
    model_is_accurate = False
    model_data_generated_this_iter = False
    norm_envs.seed(args.seed)
    current_obs_norm_np = norm_envs.reset()
    next_obs = torch.Tensor(current_obs_norm_np).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # --- Define Helper for vmap/grad ---
    def compute_scalar_value(single_obs_norm_tensor):
        # Ensure agent model is accessible in this scope if defined outside __main__
        return agent.get_value(single_obs_norm_tensor.unsqueeze(0)).squeeze()

    compute_value_grad_func = None
    if args.use_hjb_loss and TORCH_FUNC_AVAILABLE:
        try:
            compute_value_grad_func = grad(compute_scalar_value)
            print("Value gradient function for HJB created.")
        except Exception as e:
            print(f"WARNING: Failed grad function creation: {e}. HJB disabled.")
            args.use_hjb_loss = False
    elif args.use_hjb_loss:
        print("WARNING: HJB requested but torch.func unavailable. HJB disabled.")
        args.use_hjb_loss = False

    # ========================================================================
    # <<< Main Training Loop >>>
    # ========================================================================
    print(f"Starting training for {args.num_iterations} iterations...")
    for iteration in range(1, args.num_iterations + 1):
        iter_start_time = time.time()
        # LR Annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            policy_optimizer.param_groups[0]["lr"] = lrnow

        # ========================================================================
        # <<< Phase 1: Collect Real Experience & Store RAW Data >>>
        # ========================================================================
        agent.eval()
        last_original_obs_np = norm_envs.unnormalize_obs(next_obs.cpu().numpy())
        # Clear infos storage for this iteration
        real_infos_storage = [{} for _ in range(args.num_steps)]

        for step in range(0, args.num_steps):
            step_global_step = global_step + step * args.num_envs
            real_obs_storage[step] = next_obs
            real_dones_storage[step] = next_done  # Store done from previous step

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                real_values_storage[step] = value.flatten()

            real_actions_storage[step] = action
            real_logprobs_storage[step] = logprob
            action_np = action.cpu().numpy()

            next_obs_norm_cpu, reward_norm_cpu, dones_cpu, infos = norm_envs.step(
                action_np
            )
            next_done_cpu = dones_cpu  # Combined done flag for this step

            real_rewards_storage[step] = (
                torch.tensor(reward_norm_cpu).to(device).view(-1)
            )  # Norm reward for GAE
            real_single_rewards_storage[step] = real_rewards_storage[
                step
            ]  # Also for HJB

            next_obs_tensor = torch.Tensor(next_obs_norm_cpu).to(device)
            next_done_tensor = torch.Tensor(dones_cpu).to(
                device
            )  # Use combined done flag

            original_obs_np = last_original_obs_np
            original_reward_np = norm_envs.get_original_reward()

            # Determine CORRECT raw next_obs based on termination/truncation
            original_next_obs_np = np.zeros_like(original_obs_np)
            for i in range(args.num_envs):
                if dones_cpu[i]:
                    is_truncated = infos[i].get("TimeLimit.truncated", False)
                    if is_truncated and "terminal_observation" in infos[i]:
                        # Unnormalize the terminal observation before storing it as raw data
                        original_next_obs_np[i] = norm_envs.unnormalize_obs(infos[i]["terminal_observation"])
                    else:  # True termination or missing terminal_obs
                        original_next_obs_np[i] = norm_envs.unnormalize_obs(
                            next_obs_norm_cpu[i]
                        )
                else:  # Not done
                    original_next_obs_np[i] = norm_envs.unnormalize_obs(
                        next_obs_norm_cpu[i]
                    )

            # Add RAW data to buffer
            real_buffer.add(
                original_obs_np,
                original_next_obs_np,
                action_np,
                original_reward_np,
                dones_cpu,
                infos,
            )
            last_original_obs_np = (
                original_next_obs_np  # Update raw obs for next iteration
            )

            # Store infos list for this step (needed for GAE bootstrap)
            real_infos_storage[step] = infos

            # Update states for next loop iteration
            next_obs = next_obs_tensor
            next_done = next_done_tensor  # Use combined done flag

            # Log episodic returns
            done_indices = np.where(dones_cpu)[0]
            for i in done_indices:
                try:
                    info = infos[i]
                    if info and "episode" in info:
                        episode_info = info["episode"]
                        log_step = step_global_step + i
                        return_val = float(episode_info["r"])
                        length_val = int(episode_info["l"])
                        writer.add_scalar(
                            "charts/episodic_return_norm", return_val, log_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", length_val, log_step
                        )
                        print(
                            f"global_step={log_step}, env_id={i}, episodic_return_norm={return_val:.2f}, length={length_val}"
                        )
                except Exception as e:
                    print(f"Warning: Error processing info dict for env {i}: {e}")

        global_step += args.num_envs * args.num_steps
        agent.train()

        # ========================================================================
        # <<< Phase 2: Train Dynamics & Reward Models & Validate (with Early Stopping) >>>
        # ========================================================================
        model_train_time = 0
        validation_loss_state = np.nan
        validation_loss_reward = np.nan
        validation_r2_score = np.nan
        can_train_model = (
            global_step >= args.model_train_freq
            and global_step % args.model_train_freq < (args.num_envs * args.num_steps)
            and real_buffer.size()
            >= (args.model_train_batch_size + args.model_validation_batch_size)
        )

        if can_train_model:
            model_train_start_time = time.time()
            dynamic_model.train()
            reward_model.train()
            print(
                f"Iter {iteration}/{args.num_iterations}, GStep {global_step}: Training models..."
            )
            # Early Stopping Initialization
            best_val_loss_state = float("inf")
            best_state_dict_dynamics = None
            epochs_without_state_improvement = 0
            best_val_loss_reward = float("inf")
            best_state_dict_reward = None
            epochs_without_reward_improvement = 0
            final_model_epoch = 0
            total_dynamics_loss_epoch_avg = 0
            total_reward_loss_epoch_avg = 0

            for model_epoch in range(args.model_train_epochs):
                final_model_epoch = model_epoch
                epoch_dynamics_loss = 0
                epoch_reward_loss = 0
                num_batches = 0
                updates_per_epoch = 1  # Fixed updates per epoch

                # Training Step
                for _ in range(updates_per_epoch):
                    train_samples_tensors = real_buffer.sample(
                        args.model_train_batch_size
                    )
                    obs_raw_tensor = train_samples_tensors.observations
                    actions_tensor = train_samples_tensors.actions
                    rewards_raw_tensor = train_samples_tensors.rewards
                    next_obs_raw_tensor = train_samples_tensors.next_observations
                    dones_tensor = (
                        train_samples_tensors.dones
                    )  # Dones here are term ONLY
                    obs_raw_np_train = obs_raw_tensor.cpu().numpy()
                    next_obs_raw_np_train = next_obs_raw_tensor.cpu().numpy()
                    rewards_raw_np_train = rewards_raw_tensor.cpu().numpy()
                    obs_norm_np_train = norm_envs.normalize_obs(obs_raw_np_train)
                    next_obs_norm_np_train = norm_envs.normalize_obs(
                        next_obs_raw_np_train
                    )
                    rewards_norm_np_train = norm_envs.normalize_reward(
                        rewards_raw_np_train.reshape(-1)
                    ).reshape(-1, 1)
                    obs_batch_train = torch.tensor(
                        obs_norm_np_train, dtype=torch.float32
                    ).to(device)
                    action_batch_train = actions_tensor.to(device)
                    reward_batch_train_target = torch.tensor(
                        rewards_norm_np_train, dtype=torch.float32
                    ).to(device)
                    next_obs_batch_train_target = torch.tensor(
                        next_obs_norm_np_train, dtype=torch.float32
                    ).to(device)
                    dones_batch_train = dones_tensor.to(device).float()
                    # Train Dynamics (masked by term-only dones)
                    next_obs_norm_pred_train = dynamic_model(
                        obs_batch_train, action_batch_train
                    )
                    state_mse_all = nn.functional.mse_loss(
                        next_obs_norm_pred_train,
                        next_obs_batch_train_target,
                        reduction="none",
                    )
                    mask = 1.0 - dones_batch_train
                    if len(state_mse_all.shape) > len(mask.shape):
                        mask = mask.view(
                            mask.shape[0],
                            *([1] * (len(state_mse_all.shape) - len(mask.shape))),
                        )
                    masked_state_mse = state_mse_all * mask
                    loss_dynamics = masked_state_mse.sum() / mask.sum().clamp(min=1.0)
                    dynamics_optimizer.zero_grad()
                    loss_dynamics.backward()
                    dynamics_optimizer.step()
                    epoch_dynamics_loss += loss_dynamics.item()
                    # Train Reward (unmasked)
                    reward_norm_pred_train = reward_model(
                        obs_batch_train, action_batch_train
                    )
                    loss_reward = nn.functional.mse_loss(
                        reward_norm_pred_train, reward_batch_train_target
                    )
                    reward_optimizer.zero_grad()
                    loss_reward.backward()
                    reward_optimizer.step()
                    epoch_reward_loss += loss_reward.item()
                    num_batches += 1
                # Accumulate epoch averages
                if num_batches > 0:
                    total_dynamics_loss_epoch_avg += epoch_dynamics_loss / num_batches
                    total_reward_loss_epoch_avg += epoch_reward_loss / num_batches

                # Early Stopping Validation Check
                if (model_epoch + 1) % args.model_validation_freq == 0:
                    dynamic_model.eval()
                    reward_model.eval()


                    
                    val_samples_tensors = real_buffer.sample(
                        args.model_validation_batch_size
                    )
                    # Convert/Normalize Val Data...
                    obs_raw_tensor_val = val_samples_tensors.observations
                    actions_tensor_val = val_samples_tensors.actions
                    rewards_raw_tensor_val = val_samples_tensors.rewards
                    next_obs_raw_tensor_val = val_samples_tensors.next_observations
                    dones_tensor_val = val_samples_tensors.dones  # Term-only dones
                    obs_raw_np_val = obs_raw_tensor_val.cpu().numpy()
                    next_obs_raw_np_val = next_obs_raw_tensor_val.cpu().numpy()
                    rewards_raw_np_val = rewards_raw_tensor_val.cpu().numpy()
                    obs_norm_np_val = norm_envs.normalize_obs(obs_raw_np_val)
                    next_obs_norm_np_val = norm_envs.normalize_obs(next_obs_raw_np_val)
                    rewards_norm_np_val = norm_envs.normalize_reward(
                        rewards_raw_np_val.reshape(-1)
                    ).reshape(-1, 1)
                    obs_val_batch = torch.tensor(
                        obs_norm_np_val, dtype=torch.float32
                    ).to(device)
                    action_val_batch = actions_tensor_val.to(device)
                    reward_val_batch_target = torch.tensor(
                        rewards_norm_np_val, dtype=torch.float32
                    ).to(device)
                    next_obs_val_batch_target = torch.tensor(
                        next_obs_norm_np_val, dtype=torch.float32
                    ).to(device)
                    dones_val_batch = dones_tensor_val.to(device).float()
                    with torch.no_grad():
                        next_obs_norm_pred_val = dynamic_model(
                            obs_val_batch, action_val_batch
                        )
                        reward_norm_pred_val = reward_model(
                            obs_val_batch, action_val_batch
                        )
                    # Calculate Masked State Val Loss
                    val_state_mse_all = nn.functional.mse_loss(
                        next_obs_norm_pred_val,
                        next_obs_val_batch_target,
                        reduction="none",
                    )
                    val_mask = 1.0 - dones_val_batch
                    if len(val_state_mse_all.shape) > len(val_mask.shape):
                        val_mask = val_mask.view(
                            val_mask.shape[0],
                            *(
                                [1]
                                * (len(val_state_mse_all.shape) - len(val_mask.shape))
                            ),
                        )
                    val_masked_state_mse = val_state_mse_all * val_mask
                    val_loss_state_tensor = (
                        val_masked_state_mse.sum() / val_mask.sum().clamp(min=1.0)
                    )
                    current_val_loss_state = val_loss_state_tensor.item()
                    # Calculate Reward Val Loss
                    val_loss_reward_tensor = nn.functional.mse_loss(
                        reward_norm_pred_val, reward_val_batch_target
                    )
                    current_val_loss_reward = val_loss_reward_tensor.item()
                    print(
                        f"    Epoch {model_epoch+1}/{args.model_train_epochs}: Val State Loss={current_val_loss_state:.4f}, Val Reward Loss={current_val_loss_reward:.4f}"
                    )
                    # Check Dynamics Improvement
                    if current_val_loss_state < best_val_loss_state:
                        best_val_loss_state = current_val_loss_state
                        best_state_dict_dynamics = copy.deepcopy(
                            dynamic_model.ode_func.state_dict()
                        )
                        epochs_without_state_improvement = 0
                        print(
                            f"      New best dynamics validation loss: {best_val_loss_state:.4f}"
                        )
                    else:
                        epochs_without_state_improvement += args.model_validation_freq
                    # Check Reward Improvement
                    if current_val_loss_reward < best_val_loss_reward:
                        best_val_loss_reward = current_val_loss_reward
                        best_state_dict_reward = copy.deepcopy(
                            reward_model.state_dict()
                        )
                        epochs_without_reward_improvement = 0
                        print(
                            f"      New best reward validation loss: {best_val_loss_reward:.4f}"
                        )
                    else:
                        epochs_without_reward_improvement += args.model_validation_freq
                    dynamic_model.train()
                    reward_model.train()  # Back to train mode
                    # Check early stopping condition
                    if (
                        epochs_without_state_improvement
                        >= args.model_early_stopping_patience
                        and epochs_without_reward_improvement
                        >= args.model_early_stopping_patience
                    ):
                        print(
                            f"    EARLY STOPPING model training at epoch {model_epoch+1}"
                        )
                        break
            # --- End Model Training Epoch Loop ---

            # Load best model states if found
            if best_state_dict_dynamics is not None:
                dynamic_model.ode_func.load_state_dict(best_state_dict_dynamics)
                print(
                    f"  Loaded best dynamics model state dict (Val Loss: {best_val_loss_state:.4f})"
                )
                final_validation_loss_state = best_val_loss_state
            
            if best_state_dict_reward is not None:
                reward_model.load_state_dict(best_state_dict_reward)
                print(
                    f"  Loaded best reward model state dict (Val Loss: {best_val_loss_reward:.4f})"
                )
                final_validation_loss_reward = best_val_loss_reward
            

            # <<< Early Stopping Change: Perform final validation calculation AFTER potentially loading best models >>>
            # This block runs regardless of whether early stopping occurred or best models were loaded.
            # It calculates the validation metrics based on the model state that will actually be used.
            print("--- Final Validation Debug ---") # Add identifier
            # 1. Check Normalization Stats BEFORE validation sampling
            try:
                obs_var = norm_envs.obs_rms.var
                ret_var = norm_envs.ret_rms.var
                print(f"DEBUG: Final Val - Obs RMS Var (Min/Max/Mean): {obs_var.min():.4e} / {obs_var.max():.4e} / {obs_var.mean():.4e}")
                print(f"DEBUG: Final Val - Ret RMS Var: {ret_var.item():.4e}")
                # Add asserts if you want to halt on bad variance (adjust epsilon)
                # assert np.all(obs_var > 1e-8), "Obs variance too low!"
                # assert ret_var > 1e-8, "Reward variance too low!"
            except Exception as e:
                print(f"DEBUG: Error checking norm stats: {e}")

            dynamic_model.eval(); reward_model.eval()
            val_samples_tensors = real_buffer.sample(args.model_validation_batch_size)
            # Convert/Normalize validation data...
            # 2. Check data immediately AFTER sampling
            assert not torch.isnan(val_samples_tensors.observations).any(), "NaN in sampled raw obs"
            assert not torch.isnan(val_samples_tensors.next_observations).any(), "NaN in sampled raw next_obs"
            assert not torch.isnan(val_samples_tensors.rewards).any(), "NaN in sampled raw rewards"
            assert not torch.isnan(val_samples_tensors.dones).any(), "NaN in sampled raw dones"
            assert not torch.isnan(val_samples_tensors.actions).any(), "NaN in sampled raw actions"

            obs_raw_tensor_val=val_samples_tensors.observations;actions_tensor_val=val_samples_tensors.actions;rewards_raw_tensor_val=val_samples_tensors.rewards;next_obs_raw_tensor_val=val_samples_tensors.next_observations;dones_tensor_val=val_samples_tensors.dones
            obs_raw_np_val=obs_raw_tensor_val.cpu().numpy();next_obs_raw_np_val=next_obs_raw_tensor_val.cpu().numpy();rewards_raw_np_val=rewards_raw_tensor_val.cpu().numpy()
            obs_norm_np_val=norm_envs.normalize_obs(obs_raw_np_val);next_obs_norm_np_val=norm_envs.normalize_obs(next_obs_raw_np_val);rewards_norm_np_val=norm_envs.normalize_reward(rewards_raw_np_val.reshape(-1)).reshape(-1,1)
            # 3. Check data AFTER normalization (NumPy)
            assert not np.isnan(obs_norm_np_val).any(), "NaN detected AFTER normalize_obs (NumPy)"
            assert not np.isnan(next_obs_norm_np_val).any(), "NaN detected AFTER normalize_obs (NumPy) for next_obs"
            assert not np.isnan(rewards_norm_np_val).any(), "NaN detected AFTER normalize_reward (NumPy)"

            obs_val_batch=torch.tensor(obs_norm_np_val,dtype=torch.float32).to(device);action_val_batch=actions_tensor_val.to(device);reward_val_batch_target=torch.tensor(rewards_norm_np_val,dtype=torch.float32).to(device);next_obs_val_batch_target=torch.tensor(next_obs_norm_np_val,dtype=torch.float32).to(device);dones_val_batch=dones_tensor_val.to(device).float() # Need dones tensor here too
            # 4. Check data AFTER converting back to Tensor
            assert not torch.isnan(obs_val_batch).any(), "NaN detected AFTER converting obs_norm to Tensor"
            assert not torch.isnan(action_val_batch).any(), "NaN detected AFTER converting action to Tensor"
            assert not torch.isnan(next_obs_val_batch_target).any(), "NaN detected AFTER converting next_obs_norm to Tensor"
            assert not torch.isnan(reward_val_batch_target).any(), "NaN detected AFTER converting reward_norm to Tensor"

            next_obs_norm_pred_val = torch.zeros_like(next_obs_val_batch_target) # Initialize dummy prediction
            reward_norm_pred_val = torch.zeros_like(reward_val_batch_target) # Initialize dummy prediction
            with torch.no_grad():
                try:
                    next_obs_norm_pred_val = dynamic_model(obs_val_batch, action_val_batch)
                    reward_norm_pred_val = reward_model(obs_val_batch, action_val_batch)
                    # 5. Check data AFTER model prediction
                    assert not torch.isnan(next_obs_norm_pred_val).any(), "NaN detected AFTER dynamics model prediction"
                    assert not torch.isinf(next_obs_norm_pred_val).any(), "Inf detected AFTER dynamics model prediction"
                    assert not torch.isnan(reward_norm_pred_val).any(), "NaN detected AFTER reward model prediction"
                    assert not torch.isinf(reward_norm_pred_val).any(), "Inf detected AFTER reward model prediction"
                except Exception as e:
                    print(f"DEBUG: Error during final validation model prediction: {e}")

            # Calculate final masked State Val Loss and Reward Val Loss
            val_loss_state = torch.tensor(float('nan')).to(device) # Default to NaN
            val_loss_reward = torch.tensor(float('nan')).to(device) # Default to NaN
            try:
                val_state_mse_all = nn.functional.mse_loss(next_obs_norm_pred_val, next_obs_val_batch_target, reduction='none')
                val_mask = 1.0 - dones_val_batch
                if len(val_state_mse_all.shape) > len(val_mask.shape): val_mask = val_mask.view(val_mask.shape[0], *([1]*(len(val_state_mse_all.shape)-len(val_mask.shape))))
                val_masked_state_mse = val_state_mse_all * val_mask
                val_mask_sum = val_mask.sum().clamp(min=1.0)
                # 6. Check intermediate loss values
                assert not torch.isnan(val_state_mse_all).any(), "NaN detected in element-wise state mse"
                assert not torch.isinf(val_state_mse_all).any(), "Inf detected in element-wise state mse"
                assert not torch.isnan(val_masked_state_mse).any(), "NaN detected in masked state mse"
                assert not torch.isinf(val_masked_state_mse).any(), "Inf detected in masked state mse"
                assert not torch.isnan(val_mask_sum).any(), "NaN detected in mask sum"

                val_loss_state = val_masked_state_mse.sum() / val_mask_sum
                val_loss_reward = nn.functional.mse_loss(reward_norm_pred_val, reward_val_batch_target) # Final reward loss

                # 7. Check final loss values
                assert not torch.isnan(val_loss_state).any(), "NaN detected in final state loss calculation"
                assert not torch.isinf(val_loss_state).any(), "Inf detected in final state loss calculation"
                assert not torch.isnan(val_loss_reward).any(), "NaN detected in final reward loss calculation"
                assert not torch.isinf(val_loss_reward).any(), "Inf detected in final reward loss calculation"
            except Exception as e:
                 print(f"DEBUG: Error during final validation loss calculation: {e}")

            # These are the definitive final validation losses for this training phase
            validation_loss_state = val_loss_state.item()
            validation_loss_reward = val_loss_reward.item() # Store the final reward loss as well

            # Calculate final masked R2 score (using the same predictions and targets)
            # Define non_terminal_indices based on the val_mask used for state loss
            non_terminal_indices = val_mask.squeeze().nonzero(as_tuple=False).squeeze(-1)
            if non_terminal_indices.numel() > 0:
                filtered_pred = next_obs_norm_pred_val[non_terminal_indices]
                filtered_target = next_obs_val_batch_target[non_terminal_indices]
                # Add checks before R2 calculation
                assert not torch.isnan(filtered_pred).any(), "NaN in filtered predictions for R2"
                assert not torch.isinf(filtered_pred).any(), "Inf in filtered predictions for R2"
                assert not torch.isnan(filtered_target).any(), "NaN in filtered targets for R2"
                assert not torch.isinf(filtered_target).any(), "Inf in filtered targets for R2"
                try:
                    validation_r2_score = calculate_r2_score(
                        filtered_target, filtered_pred
                    ).item()
                    assert not np.isnan(validation_r2_score), "NaN R2 score calculated"
                    assert not np.isinf(validation_r2_score), "Inf R2 score calculated"
                except Exception as e:
                    print(f"DEBUG: Error calculating R2 score: {e}")
                    validation_r2_score = np.nan
            else:
                validation_r2_score = np.nan

            # Update accuracy flag based on final calculated state validation loss
            model_is_accurate = validation_loss_state < args.model_state_accuracy_threshold

            writer.add_scalar(
                "losses/dynamics_model_validation_loss_state", # Log the final state loss
                validation_loss_state,
                global_step,
            )
            writer.add_scalar(
                "losses/reward_model_validation_loss", # Log the final reward loss
                validation_loss_reward,
                global_step,
            )
            writer.add_scalar(
                "losses/dynamics_model_R2_validation", validation_r2_score, global_step
            )
            print(
                f"  Final Model Validation -> State Loss: {validation_loss_state:.4f}, Reward Loss: {validation_loss_reward:.4f}, State R2: {validation_r2_score:.3f}, Accurate (State): {model_is_accurate}"
            )
            dynamic_model.train()
            reward_model.train()
            model_train_time = time.time() - model_train_start_time
        elif iteration % args.model_train_freq == 0:
            model_is_accurate = False
            print(
                f"Iter {iteration}/{args.num_iterations}, GStep {global_step}: Skipping model training (buffer too small)."
            )
        writer.add_scalar("charts/model_is_accurate", model_is_accurate, global_step)

        # ========================================================================
        # <<< Phase 3: Generate Model Rollouts (Conditionally) >>>
        # ========================================================================
        model_rollout_gen_time = 0
        model_data_generated_this_iter = False
        can_rollout = (
            global_step >= args.model_rollout_freq
            and global_step % args.model_rollout_freq < (args.num_envs * args.num_steps)
            and real_buffer.size() >= args.num_model_rollout_starts
            and model_is_accurate
        )
        if can_rollout:
            model_rollout_start_time = time.time()
            print(
                f"Iter {iteration}/{args.num_iterations}, GStep {global_step}: Generating model rollouts..."
            )
            model_data_generated_this_iter = True
            start_states_samples_tensors = real_buffer.sample(
                args.num_model_rollout_starts
            )
            current_obs_raw_np = start_states_samples_tensors.observations.cpu().numpy()
            dynamic_model.eval()
            reward_model.eval()
            agent.eval()
            (
                rollout_obs_list,
                rollout_action_list,
                rollout_reward_list,
                rollout_single_reward_list,
            ) = ([], [], [], [])
            rollout_logprob_list, rollout_value_list = [], []
            with torch.no_grad():
                for h in range(args.model_rollout_length):
                    current_obs_norm_np = norm_envs.normalize_obs(current_obs_raw_np)
                    current_obs_norm = torch.tensor(current_obs_norm_np).to(device)
                    action_model, logprob_model, _, value_model = (
                        agent.get_action_and_value(current_obs_norm)
                    )
                    next_obs_norm_pred = dynamic_model(current_obs_norm, action_model)
                    reward_norm_pred = reward_model(current_obs_norm, action_model)
                    rollout_obs_list.append(current_obs_norm)
                    rollout_action_list.append(action_model)
                    rollout_reward_list.append(reward_norm_pred.squeeze(-1))
                    rollout_single_reward_list.append(reward_norm_pred)
                    rollout_logprob_list.append(logprob_model)
                    rollout_value_list.append(value_model.squeeze(-1))
                    current_obs_raw_np = norm_envs.unnormalize_obs(
                        next_obs_norm_pred.cpu().numpy()
                    )
                final_next_obs_norm_np = norm_envs.normalize_obs(current_obs_raw_np)
                final_next_obs_norm = torch.tensor(final_next_obs_norm_np).to(device)
                final_next_value_model = agent.get_value(final_next_obs_norm).reshape(
                    1, -1
                )
            model_obs_storage = torch.stack(rollout_obs_list, dim=0)
            model_actions_storage = torch.stack(rollout_action_list, dim=0)
            model_rewards_storage = torch.stack(rollout_reward_list, dim=0)
            model_single_rewards_storage = torch.stack(
                rollout_single_reward_list, dim=0
            )
            model_logprobs_storage = torch.stack(rollout_logprob_list, dim=0)
            model_values_storage = torch.stack(rollout_value_list, dim=0)
            model_final_next_values_storage = final_next_value_model
            model_rollout_gen_time = time.time() - model_rollout_start_time
            print(f"  Rollout gen complete.")
            agent.train()
        elif iteration % args.model_rollout_freq == 0:
            print(
                f"Iter {iteration}/{args.num_iterations}, GStep {global_step}: Skipping model rollouts."
            )

        # ========================================================================
        # <<< Phase 4: Prepare Data and Perform PPO Update (with GAE Mod + HJB) >>>
        # ========================================================================
        ppo_update_start_time = time.time()
        agent.train()
        # --- Select Data Source & Prepare Batches ---
        if args.use_model_for_updates and model_data_generated_this_iter:
            print(f"  Using MODEL data for PPO update.")
            advantages, returns_norm = compute_gae_model_data(
                model_rewards_storage,
                model_values_storage,
                args.gamma,
                args.gae_lambda,
                model_final_next_values_storage,
            )
            b_obs = model_obs_storage.reshape((-1,) + obs_shape)
            b_logprobs = model_logprobs_storage.reshape(-1)
            b_actions = model_actions_storage.reshape((-1,) + action_shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns_norm.reshape(-1)
            b_values = model_values_storage.reshape(-1)
            b_single_rewards = model_single_rewards_storage.reshape(-1, 1)
            b_dones = (
                torch.zeros_like(b_returns).to(device).float()
            )  # Dones are False for model rollouts
            current_batch_size = b_obs.shape[0]
            current_minibatch_size = max(1, current_batch_size // args.num_minibatches)
        else:  # Use REAL data
            if args.use_model_for_updates and not model_data_generated_this_iter:
                print(
                    f"  Using REAL data for PPO update (model data not generated/accurate)."
                )
            elif not args.use_model_for_updates:
                print(
                    f"  Using REAL data for PPO update (use_model_for_updates=False)."
                )
            advantages, returns_norm = compute_gae_real_data(
                real_rewards_storage,
                real_values_storage,
                real_dones_storage,
                next_obs,
                next_done,
                real_infos_storage,
                agent,
                args.gamma,
                args.gae_lambda,
                norm_envs,
                device,
            )
            b_obs = real_obs_storage.reshape((-1,) + obs_shape)
            b_logprobs = real_logprobs_storage.reshape(-1)
            b_actions = real_actions_storage.reshape((-1,) + action_shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns_norm.reshape(-1)
            b_values = real_values_storage.reshape(-1)
            b_single_rewards = real_single_rewards_storage.reshape(-1, 1)
            # Dones for HJB mask (term or trunc, from real_dones_storage)
            b_dones = real_dones_storage.reshape(-1).float()
            current_batch_size = b_obs.shape[0]
            current_minibatch_size = max(1, current_batch_size // args.num_minibatches)

        # --- PPO Optimization Loop ---
        if current_batch_size >= current_minibatch_size and current_minibatch_size > 0:
            b_inds = np.arange(current_batch_size)
            clipfracs = []
            approx_kls = []
            old_approx_kls = []
            final_epoch_losses = {}
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, current_batch_size, current_minibatch_size):
                    end = start + current_minibatch_size
                    mb_inds = b_inds[start:end]
                    mb_obs = b_obs[mb_inds]
                    mb_actions = b_actions[mb_inds]
                    mb_logprobs_old = b_logprobs[mb_inds]
                    mb_advantages = b_advantages[mb_inds]
                    mb_returns = b_returns[mb_inds]
                    mb_values_old = b_values[mb_inds]
                    mb_rewards_norm = b_single_rewards[mb_inds]
                    mb_dones = b_dones[mb_inds].unsqueeze(
                        -1
                    )  # Dones for HJB mask [mb, 1]
                    # Standard PPO Losses
                    action_new, log_prob_new, entropy, value_norm_new_raw = (
                        agent.get_action_and_value(mb_obs, mb_actions)
                    )
                    value_norm_new = value_norm_new_raw.view(-1)
                    logratio = log_prob_new - mb_logprobs_old
                    ratio = logratio.exp()
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]
                    approx_kls.append(approx_kl.item())
                    old_approx_kls.append(old_approx_kl.item())
                    mb_adv_normalized = mb_advantages
                    if args.norm_adv:
                        mb_adv_normalized = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )
                    pg_loss1 = -mb_adv_normalized * ratio
                    pg_loss2 = -mb_adv_normalized * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    if args.clip_vloss:
                        v_loss_unclipped = nn.functional.mse_loss(
                            value_norm_new, mb_returns, reduction="none"
                        )
                        v_clipped = mb_values_old + torch.clamp(
                            value_norm_new - mb_values_old,
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = nn.functional.mse_loss(
                            v_clipped, mb_returns, reduction="none"
                        )
                        v_loss = (
                            0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                        )
                    else:
                        v_loss = 0.5 * nn.functional.mse_loss(
                            value_norm_new, mb_returns
                        )
                    entropy_loss = entropy.mean()
                    # HJB Calculation (Masked)
                    L_HJB = torch.tensor(0.0).to(device)
                    if args.use_hjb_loss and compute_value_grad_func is not None:
                        try:
                            action_from_batch = mb_actions
                            dVdx_norm = vmap(compute_value_grad_func)(mb_obs)
                            with torch.no_grad():
                                f_norm = dynamic_model.ode_func(
                                    0, mb_obs, action_from_batch
                                )
                            r_norm_actual = mb_rewards_norm
                            hamiltonian = r_norm_actual.squeeze(-1) + torch.einsum(
                                "bi,bi->b", dVdx_norm, f_norm
                            )
                            hjb_residual = hamiltonian - args.rho * value_norm_new
                            hjb_residual_sq = hjb_residual.pow(2)
                            # Mask HJB loss using mb_dones
                            # If using real data, mb_dones=term_or_trunc. If using model data, mb_dones=0.
                            # Mask should ideally only remove true terminations. This requires dones from buffer sample w/ handle_timeout=True.
                            # TODO: Pass dones from buffer sample into PPO update phase for more accurate HJB masking.
                            mask = 1.0 - mb_dones.squeeze(-1)
                            masked_hjb_residual_sq = hjb_residual_sq * mask
                            L_HJB = masked_hjb_residual_sq.sum() / mask.sum().clamp(
                                min=1.0
                            )
                        except Exception as e:
                            L_HJB = torch.tensor(0.0).to(device)
                            print(f"HJB Error: {e}")
                    # Total Loss
                    L_Total = (
                        pg_loss
                        - args.ent_coef * entropy_loss
                        + args.vf_coef * v_loss
                        + args.hjb_coef * L_HJB
                    )
                    # Optimization
                    policy_optimizer.zero_grad()
                    L_Total.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    policy_optimizer.step()
                    # Store losses
                    if start == 0 and epoch == args.update_epochs - 1:
                        final_epoch_losses = {
                            "v_loss": v_loss.item(),
                            "pg_loss": pg_loss.item(),
                            "entropy_loss": entropy_loss.item(),
                            "hjb_loss": L_HJB.item(),
                            "old_approx_kl": old_approx_kl.item(),
                            "approx_kl": approx_kl.item(),
                        }
                if (
                    args.target_kl is not None
                    and np.mean(approx_kls[-args.num_minibatches :]) > args.target_kl
                ):
                    print(f"  Epoch {epoch+1}: Early stopping PPO update.")
                    break
        else:
            print(
                f"Iter {iteration}/{args.num_iterations}: Skipping PPO update - Invalid batch/minibatch size."
            )
            final_epoch_losses = {
                "v_loss": 0,
                "pg_loss": 0,
                "entropy_loss": 0,
                "hjb_loss": 0,
                "old_approx_kl": 0,
                "approx_kl": 0,
            }
            clipfracs = [0.0]
            approx_kls = [0.0]
            old_approx_kls = [0.0]

        # --- Logging ---
        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        ppo_update_time = time.time() - ppo_update_start_time
        writer.add_scalar(
            "charts/learning_rate", policy_optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar(
            "losses/value_loss", final_epoch_losses.get("v_loss", np.nan), global_step
        )
        writer.add_scalar(
            "losses/policy_loss", final_epoch_losses.get("pg_loss", np.nan), global_step
        )
        writer.add_scalar(
            "losses/entropy",
            final_epoch_losses.get("entropy_loss", np.nan),
            global_step,
        )
        writer.add_scalar(
            "losses/hjb_loss", final_epoch_losses.get("hjb_loss", np.nan), global_step
        )
        writer.add_scalar(
            "losses/old_approx_kl",
            final_epoch_losses.get("old_approx_kl", np.nan),
            global_step,
        )
        writer.add_scalar(
            "losses/approx_kl", final_epoch_losses.get("approx_kl", np.nan), global_step
        )
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/buffer_size", real_buffer.size(), global_step)
        total_iter_time = time.time() - iter_start_time
        current_sps = (
            int((args.num_envs * args.num_steps) / total_iter_time)
            if total_iter_time > 0
            else 0
        )
        writer.add_scalar("charts/SPS_iter", current_sps, global_step)
        writer.add_scalar("perf/model_train_time", model_train_time, global_step)
        writer.add_scalar(
            "perf/model_rollout_gen_time", model_rollout_gen_time, global_step
        )
        writer.add_scalar("perf/ppo_update_time", ppo_update_time, global_step)
        print(
            f"Iter {iteration}/{args.num_iterations}: SPS={current_sps}, ExplVar={explained_var:.3f}, ModelAccurate={model_is_accurate}, ValLossState={validation_loss_state:.4f}, ValR2={validation_r2_score:.3f}"
        )

    # ========================================================================
    # <<< End of Training >>>
    # ========================================================================
    # --- Saving & Evaluation ---
    if args.save_model:
        run_folder = f"runs/{run_name}"
        os.makedirs(run_folder, exist_ok=True)
        agent_model_path = f"{run_folder}/{args.exp_name}_agent.cleanrl_model"
        torch.save(agent.state_dict(), agent_model_path)
        print(f"Agent saved: {agent_model_path}")
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
    if args.save_model:  # Eval only if saved
        print("\nEvaluating agent performance...")
        eval_episodes = 10
        eval_seeds = range(args.seed + 100, args.seed + 100 + eval_episodes)
        eval_returns_raw = []
        for seed in eval_seeds:
            eval_envs_base = DummyVecEnv(
                [make_env(args.env_id, 0, False, f"{run_name}-eval-seed{seed}")]
            )
            eval_norm_envs = VecNormalize.load(norm_stats_path, eval_envs_base)
            eval_norm_envs.training = False
            eval_norm_envs.norm_reward = False
            eval_agent = Agent(eval_norm_envs).to(device)
            eval_agent.load_state_dict(
                torch.load(agent_model_path, map_location=device)
            )
            eval_agent.eval()
            obs_norm_np, _ = eval_norm_envs.reset(seed=seed)
            done = False
            episode_return_raw = 0
            while not done:
                with torch.no_grad():
                    action, _, _, _ = eval_agent.get_action_and_value(
                        torch.Tensor(obs_norm_np).to(device)
                    )
                obs_norm_np, _, terminated, truncated, info = eval_norm_envs.step(
                    action.cpu().numpy()
                )
                done = terminated[0] or truncated[0]
                raw_reward = eval_norm_envs.get_original_reward()
                episode_return_raw += raw_reward[0]
            eval_returns_raw.append(episode_return_raw)
            print(f"  Eval Seed {seed}: Raw Episodic Return={episode_return_raw:.2f}")
            eval_envs_base.close()
        mean_eval_return_raw = np.mean(eval_returns_raw)
        std_eval_return_raw = np.std(eval_returns_raw)
        print(
            f"Evaluation complete. Average Raw Return: {mean_eval_return_raw:.2f} +/- {std_eval_return_raw:.2f}"
        )
        for idx, episodic_return in enumerate(eval_returns_raw):
            writer.add_scalar("eval/raw_episodic_return", episodic_return, idx)
        if args.upload_model:
            print("Uploading to Hugging Face Hub...")
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args=args,
                episodic_returns=eval_returns_raw,
                repo_id=repo_id,
                algo="PPO-MBPO-TorchODE-HJB-ES",
                folder_path=run_folder,
                commit_message=f"Upload PPO+TorchODE+ES agent for {args.env_id} seed {args.seed} (Models/Stats separate)",
            )
            print(f"Models pushed to: {repo_id}")
            print(
                "NOTE: Requires manual upload of models/stats for full reproducibility."
            )

    # --- Cleanup ---
    norm_envs.close()
    writer.close()
    print("\nTraining finished.")
