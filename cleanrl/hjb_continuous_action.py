# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchode as to
from sklearn.metrics import r2_score
from torch import vmap
from torch.func import grad
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
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
    hjb_coef: float = 0.5
    """coefficient for HJB residual loss"""
    hjb_policy_steps: int = 1
    """Number of policy optimization steps per update iteration"""
    hjb_dynamic_threshold: float = 0.01
    """MSE threshold for dynamic model"""
    hjb_reward_threshold: float = 0.05
    """MSE threshold for reward model"""
    hjb_dynamic_patience: int = 500
    """Number of epochs to wait for improvement in dynamic model training"""
    hjb_dynamic_min_delta: float = 1e-5
    """Minimum improvement delta for early stopping in dynamic model training"""
    

    # Algorithm specific arguments
    env_id: str = "InvertedPendulum-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    vf_coef: float = 0.0
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ODEFunc(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 512),
            nn.Softplus(),
            nn.Linear(512, 512),
            nn.Softplus(),
            nn.Linear(512, 512),
            nn.Softplus(),
            nn.Linear(512, obs_dim),
        )
        self.dt = 0.05  # Should match environment timestep

    def forward(self, t, x, a_sequence):
        assert x.ndim == 2, f"x must be (batch, obs_dim), got {x.shape}"
        assert a_sequence.ndim == 3, f"a_sequence must be (batch, seq_len, action_dim), got {a_sequence.shape}"
        assert x.shape[0] == a_sequence.shape[0], "Batch size mismatch between x and a_sequence"
        
        idx = torch.clamp((t / self.dt).long(), 0, a_sequence.size(1)-1)
        a = a_sequence[torch.arange(x.size(0)), idx].unsqueeze(-1) if a_sequence.dim() == 2 else a_sequence[torch.arange(x.size(0)), idx]
        
        
        output = self.net(torch.cat([x, a], dim=-1))
        assert output.shape == x.shape, \
            f"Output shape {output.shape} should match input x shape {x.shape}"
        return output

class DynamicModel(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.ode_func = ODEFunc(obs_dim, action_dim)
        self.dt = 0.05
        
        # TorchODE components
        self.term = to.ODETerm(self.ode_func, with_args=True)
        self.step_method = to.Tsit5(term=self.term)
        self.step_size_controller = to.IntegralController(atol=1e-9, rtol=1e-6, term=self.term)
        self.adjoint = to.AutoDiffAdjoint(
            step_method=self.step_method,
            step_size_controller=self.step_size_controller,
        )

    def forward(self, initial_obs, action_sequences):
        assert initial_obs.ndim == 2, f"initial_obs must be (batch, obs_dim), got {initial_obs.shape}"
        assert action_sequences.ndim == 3, \
            f"action_sequences must be (batch, seq_len, action_dim), got {action_sequences.shape}"
        
        batch_size, seq_len = action_sequences.shape[:2]
        t_eval = torch.stack([torch.linspace(0, self.dt*seq_len, seq_len+1)]*batch_size)
        
        assert t_eval.shape == (batch_size, seq_len+1), \
            f"t_eval shape mismatch: {t_eval.shape} vs expected ({batch_size}, {seq_len+1})"
        
        problem = to.InitialValueProblem(
            y0=initial_obs,
            t_eval=t_eval.to(initial_obs.device),
        )
        sol = self.adjoint.solve(problem, args=action_sequences)
        
        assert sol.ys.shape == (batch_size, seq_len+1, self.ode_func.net[-1].out_features), \
            f"Solution shape {sol.ys.shape} vs expected ({batch_size}, {seq_len+1}, {self.ode_func.net[-1].out_features})"
        
        return sol.ys[:, 1:, :]


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), 256))
        self.fc2 = layer_init(nn.Linear(256, 256))
        self.fc_mu = layer_init(nn.Linear(256, np.prod(env.single_action_space.shape)), std=0.01)
        
        # Action scaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32
            )
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32
            )
        )

    def forward(self, x):
        assert x.ndim == 2, f"Input must be (batch, obs_dim), got {x.shape}"
        x = torch.relu(self.fc1(x))
        assert x.shape == (x.shape[0], 256), f"After fc1: {x.shape} vs (batch, 256)"
        x = torch.relu(self.fc2(x)) 
        assert x.shape == (x.shape[0], 256), f"After fc2: {x.shape} vs (batch, 256)"
        x = torch.tanh(self.fc_mu(x))
        assert x.shape == (x.shape[0], self.action_scale.shape[0]), \
            f"Final action shape {x.shape} vs ({x.shape[0]}, {self.action_scale.shape[0]})"
        return x * self.action_scale + self.action_bias

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor = Actor(envs)
    
    def get_value(self, x):
        return self.critic(x)
    
    def get_action(self, x):
        return self.actor(x)


class RewardModel(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + action_dim, 256)),
            nn.SiLU(),
            layer_init(nn.Linear(256, 256)),
            nn.SiLU(),
            layer_init(nn.Linear(256, 1)),
        )
    
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        return self.net(x).squeeze(-1)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    
    # Initialize dynamic model
    obs_dim = np.prod(envs.single_observation_space.shape)
    action_dim = np.prod(envs.single_action_space.shape)
    dynamic_model = DynamicModel(obs_dim, action_dim).to(device)
    dynamic_optimizer = optim.AdamW(dynamic_model.parameters(), lr=args.learning_rate)
    
    # Initialize reward model
    reward_model = RewardModel(obs_dim, action_dim).to(device)
    reward_optimizer = optim.AdamW(reward_model.parameters(), lr=args.learning_rate)
    
    # Separate optimizers for actor and critic
    actor_optimizer = optim.AdamW(agent.actor.parameters(), lr=args.learning_rate, eps=1e-5)
    critic_optimizer = optim.AdamW(agent.critic.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    # Ensure actions always have at least 1 dimension
    action_shape = envs.single_action_space.shape
    if not action_shape:  # Handle scalar action space
        action_shape = (1,)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape).to(device)
    # Ensure action storage maintains dimensionality
    if len(action_shape) == 0:
        actions = actions.unsqueeze(-1)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_observations = torch.zeros_like(obs).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        dynamic_model.train()
        reward_model.train()
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            actor_optimizer.param_groups[0]["lr"] = lrnow
            critic_optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Action selection with exploration noise
            with torch.no_grad():
                action = agent.get_action(next_obs)
                # Add Gaussian noise for exploration
                noise = torch.randn_like(action) * 0.1
                action = torch.clamp(action + noise, 
                                   torch.tensor(envs.single_action_space.low, device=device),
                                   torch.tensor(envs.single_action_space.high, device=device))
                values[step] = agent.get_value(next_obs).flatten()
            
            actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_observations[step] = torch.Tensor(next_obs).to(device)  # Store before reset
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        # 1. Identify complete trajectories across all environments
        trajectories = []
        current_start = 0

        # Find trajectory boundaries using done signals
        for step in range(args.num_steps):
            if dones[step].any():
                # For each environment that finished
                for env_idx in torch.where(dones[step])[0]:
                    trajectories.append({
                        'start': current_start,
                        'end': step,
                        'env_idx': env_idx.item(),
                        'length': step - current_start + 1
                    })
                current_start = step + 1

        # Handle final partial trajectory
        if current_start < args.num_steps:
            trajectories.append({
                'start': current_start,
                'end': args.num_steps - 1,
                'env_idx': 0,  # For single env case
                'length': args.num_steps - current_start
            })

        # 2. Split trajectories into train/validation
        traj_indices = torch.randperm(len(trajectories))
        split = int(0.8 * len(trajectories))
        train_traj_indices = traj_indices[:split]
        val_traj_indices = traj_indices[split:]

        # 3. Prepare trajectory datasets
        def prepare_trajectory_data(traj_indices):
            traj_obs = []
            traj_actions = []
            traj_next_obs = []
            traj_masks = []
            
            # Convert tensor indices to Python list
            traj_indices = traj_indices.tolist()
            
            # Randomly sample a batch of trajectories
            batch_size = 8  # Adjust based on available memory
            if len(traj_indices) > batch_size:
                sampled_indices = random.sample(traj_indices, batch_size)
            else:
                sampled_indices = traj_indices
            
            for idx in sampled_indices:
                traj = trajectories[idx]
                env_idx = traj['env_idx']
                
                # Extract trajectory data
                obs_slice = obs[traj['start']:traj['end']+1, env_idx]
                action_slice = actions[traj['start']:traj['end']+1, env_idx]
                next_obs_slice = next_observations[traj['start']:traj['end']+1, env_idx]
                
                # Create validity mask (all True except after termination)
                mask = torch.ones(traj['length'], dtype=bool)
                if dones[traj['end'], env_idx]:
                    mask[-1] = False
                
                traj_obs.append(obs_slice)
                traj_actions.append(action_slice)
                traj_next_obs.append(next_obs_slice)
                traj_masks.append(mask)
            
            return traj_obs, traj_actions, traj_next_obs, traj_masks

        # Prepare datasets
        train_obs, train_actions, train_next_obs, train_masks = prepare_trajectory_data(train_traj_indices)
        val_obs, val_actions, val_next_obs, val_masks = prepare_trajectory_data(val_traj_indices)

        # Log buffer statistics
        writer.add_scalar("buffer/obs_mean", obs.mean().item(), global_step)
        writer.add_scalar("buffer/obs_std", obs.std().item(), global_step)
        writer.add_scalar("buffer/next_obs_mean", next_observations.mean().item(), global_step)
        writer.add_scalar("buffer/next_obs_std", next_observations.std().item(), global_step)
        writer.add_scalar("buffer/actions_mean", actions.mean().item(), global_step)
        writer.add_scalar("buffer/actions_std", actions.std().item(), global_step)

        # Dynamic model evaluation check
        if len(trajectories) == 0:
            print("No trajectories found for training!")
        else:
            print(f"Training on {len(train_traj_indices)} trajectories, validating on {len(val_traj_indices)}")
            
            # Calculate initial validation metrics
            if len(val_obs) != len(val_traj_indices):
                print(f"Validation data filtered: {len(val_obs)}/{len(val_traj_indices)} trajectories valid")
                writer.add_scalar("debug/val_traj_filtered", len(val_traj_indices)-len(val_obs), iteration)
                
            if len(val_obs) > 0:
                with torch.no_grad():
                    initial_val_loss = 0
                    initial_val_steps = 0
                    for i in range(len(val_obs)):
                        val_obs_traj = val_obs[i]
                        val_actions_traj = val_actions[i]
                        val_next_obs_traj = val_next_obs[i]
                        val_mask_traj = val_masks[i]
                        
                        val_pred = dynamic_model(val_obs_traj[0].unsqueeze(0), val_actions_traj.unsqueeze(0))
                        val_steps = val_mask_traj.sum().item()
                        if val_steps > 0:
                            preds = val_pred[0, :val_mask_traj.shape[0]-1][val_mask_traj[:-1]].cpu().numpy()
                            trues = val_next_obs_traj[:-1][val_mask_traj[:-1]].cpu().numpy()
                            initial_val_loss += nn.MSELoss()(
                                val_pred[0, :val_mask_traj.shape[0]-1][val_mask_traj[:-1]],
                                val_next_obs_traj[:-1][val_mask_traj[:-1]]
                            ).item() * val_steps
                            
                            # Add check for R² calculation
                            if len(preds) > 0 and len(trues) > 0:
                                initial_val_r2 = r2_score(trues, preds)
                                writer.add_scalar("dynamic/initial_val_r2", initial_val_r2, iteration)
                            else:
                                initial_val_r2 = -1.0  # Indicate invalid value
                            
                            initial_val_steps += val_steps
                    
                    if initial_val_steps > 0:
                        initial_val_mse = initial_val_loss / initial_val_steps
                        writer.add_scalar("dynamic/initial_val_mse", initial_val_mse, iteration)
                    else:
                        initial_val_mse = float('inf')
                        print("No valid validation samples found for dynamic model")

            # Check if dynamic model is already good enough
            if len(val_traj_indices) > 0 and initial_val_steps > 0 and initial_val_mse <= args.hjb_dynamic_threshold:
                print(f"Dynamic model already good (val MSE {initial_val_mse:.4f} <= {args.hjb_dynamic_threshold}), skipping training")
                writer.add_scalar("dynamic/skipped_training", 1, iteration)
            else:
                writer.add_scalar("dynamic/skipped_training", 0, iteration)
                print(f"Dynamic model needs training (val MSE {initial_val_mse if initial_val_steps > 0 else 'N/A':.4f} > {args.hjb_dynamic_threshold})")
            
                best_val_mse = float('inf')
                patience_counter = 0
            
            print("Pretraining dynamic model on full dataset...")
            best_val_mse = float('inf')
            patience_counter = 0

            # Reduce number of epochs since we're using full dataset each time
            for pretrain_epoch in trange(100, desc="Dynamic Pretraining"):
                epoch_loss = 0.0
                total_steps = 0
                
                # Shuffle training trajectories each epoch
                traj_order = torch.randperm(len(train_obs))
                
                dynamic_model.train()
                dynamic_optimizer.zero_grad()
                
                for traj_idx in traj_order:
                    traj_obs = train_obs[traj_idx]
                    traj_actions = train_actions[traj_idx]
                    traj_next_obs = train_next_obs[traj_idx]
                    traj_mask = train_masks[traj_idx]
                    
                    # Skip invalid trajectories
                    if len(traj_obs) < 2 or traj_mask.sum() == 0:
                        continue
                        
                    # Forward pass
                    pred_traj = dynamic_model(
                        traj_obs[0].unsqueeze(0),
                        traj_actions.unsqueeze(0)
                    )
                    
                    # Calculate masked loss
                    valid_steps = traj_mask[:-1].sum().item()
                    if valid_steps == 0:
                        continue
                        
                    loss = nn.MSELoss()(
                        pred_traj[0, :len(traj_obs)-1][traj_mask[:-1]],
                        traj_next_obs[:-1][traj_mask[:-1]]
                    )
                    
                    # Scale loss by trajectory length and accumulate gradients
                    (loss * valid_steps).backward()
                    epoch_loss += loss.item() * valid_steps
                    total_steps += valid_steps
                
                if total_steps > 0:
                    # Average gradients and update
                    for param in dynamic_model.parameters():
                        if param.grad is not None:
                            param.grad /= total_steps
                    torch.nn.utils.clip_grad_norm_(dynamic_model.parameters(), args.max_grad_norm)
                    dynamic_optimizer.step()
                    dynamic_optimizer.zero_grad()
                    
                    # Log training metrics
                    avg_epoch_loss = epoch_loss / total_steps
                    writer.add_scalar("dynamic/train_loss", avg_epoch_loss, pretrain_epoch)
                
                # Full validation pass
                dynamic_model.eval()
                with torch.no_grad():
                    val_loss_sum = 0.0
                    val_steps_total = 0
                    all_preds = []
                    all_trues = []
                    for val_idx in range(len(val_obs)):
                        val_obs_traj = val_obs[val_idx]
                        val_actions_traj = val_actions[val_idx]
                        val_next_obs_traj = val_next_obs[val_idx]
                        val_mask_traj = val_masks[val_idx]
                        
                        if len(val_obs_traj) < 2 or val_mask_traj.sum() == 0:
                            continue
                            
                        val_pred = dynamic_model(
                            val_obs_traj[0].unsqueeze(0),
                            val_actions_traj.unsqueeze(0)
                        )
                        
                        valid_steps = val_mask_traj[:-1].sum().item()
                        if valid_steps > 0:
                            pred_slice = val_pred[0, :val_mask_traj.shape[0]-1][val_mask_traj[:-1]]
                            true_slice = val_next_obs_traj[:-1][val_mask_traj[:-1]]
                            
                            # Accumulate for R² calculation
                            all_preds.append(pred_slice.cpu().numpy())
                            all_trues.append(true_slice.cpu().numpy())
                            
                            # Existing MSE calculation
                            val_loss = nn.MSELoss()(pred_slice, true_slice).item()
                            val_loss_sum += val_loss * valid_steps
                            val_steps_total += valid_steps
                    
                    # Compute R² across all validation data
                    if len(all_preds) > 0:
                        all_preds = np.concatenate(all_preds)
                        all_trues = np.concatenate(all_trues)
                        
                        if len(all_preds) >= 2 and np.var(all_trues) > 1e-8:
                            val_r2 = r2_score(all_trues, all_preds)
                        else:
                            val_r2 = -1.0  # Indicate invalid calculation
                    else:
                        val_r2 = -1.0
                    
                    # Log metrics
                    if val_steps_total > 0:
                        val_mse = val_loss_sum / val_steps_total
                        writer.add_scalar("dynamic/val_mse", val_mse, pretrain_epoch)
                        writer.add_scalar("dynamic/val_r2", val_r2, pretrain_epoch)
                        
                        # Early stopping
                        if val_mse < (best_val_mse - args.hjb_dynamic_min_delta):
                            best_val_mse = val_mse
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= args.hjb_dynamic_patience:
                                print(f"Early stopping at epoch {pretrain_epoch}")
                                break
                    else:
                        val_mse = float('inf')

            print(f"Dynamic model training completed. Best validation MSE: {best_val_mse:.4f}")


        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.critic(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = agent.critic(obs[t + 1]).reshape(1, -1)
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_rewards = rewards.reshape(-1)
        
        # Reward model evaluation check
        print("Evaluating reward model...")
        
        # Use all transitions since rewards are valid even when done
        indices = torch.randperm(b_obs.size(0))
        split = int(0.8 * b_obs.size(0))
        train_idx_r, val_idx_r = indices[:split], indices[split:]
        
        with torch.no_grad():
            pred_val_r = reward_model(b_obs[val_idx_r], b_actions[val_idx_r])
            initial_val_mse_r = nn.MSELoss()(pred_val_r, b_rewards[val_idx_r]).item()
        
        writer.add_scalar("reward/initial_val_mse", initial_val_mse_r, iteration)
        
        if initial_val_mse_r <= args.hjb_reward_threshold:
            print(f"Reward model already good (val MSE {initial_val_mse_r:.4f} <= {args.hjb_reward_threshold}), skipping training")
        else:
            print("Pretraining reward model...")
            best_val_mse_r = float('inf')
            patience_counter_r = 0
            for pretrain_epoch in trange(5000, desc="Reward Pretraining"):
                # Training step
                reward_optimizer.zero_grad()
                pred_train = reward_model(b_obs[train_idx_r], b_actions[train_idx_r])
                loss = nn.MSELoss()(pred_train, b_rewards[train_idx_r])
                loss.backward()
                reward_optimizer.step()
                
                # Validation
                with torch.no_grad():
                    pred_val = reward_model(b_obs[val_idx_r], b_actions[val_idx_r])
                    val_mse = nn.MSELoss()(pred_val, b_rewards[val_idx_r]).item()
                    val_mae = nn.L1Loss()(pred_val, b_rewards[val_idx_r]).item()
                    val_r2 = r2_score(b_rewards[val_idx_r].cpu().numpy(), pred_val.cpu().numpy())
                    
                writer.add_scalar("reward/pretrain_train_mse", loss.item(), pretrain_epoch)
                writer.add_scalar("reward/pretrain_val_mse", val_mse, pretrain_epoch)
                writer.add_scalar("reward/pretrain_val_mae", val_mae, pretrain_epoch)
                writer.add_scalar("reward/pretrain_val_r2", val_r2, pretrain_epoch)
                
                # Early stopping
                if val_mse < (best_val_mse_r - 1e-5):
                    best_val_mse_r = val_mse
                    patience_counter_r = 0
                else:
                    if (patience_counter_r := patience_counter_r + 1) >= 5:
                        break

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Get current observations
                mb_obs = b_obs[mb_inds]
                
                # Policy update (actor)
                for _ in range(args.hjb_policy_steps):
                    compute_value_grad = grad(lambda x: agent.critic(x).squeeze())
                    dVdx = vmap(compute_value_grad, in_dims=(0))(mb_obs)  # (minibatch_size, obs_dim)

                    # Hamiltonian calculation
                    current_actions = agent.actor(mb_obs)  # (minibatch_size, action_dim)
                    f = dynamic_model(mb_obs, current_actions.unsqueeze(1))[:,0,:]  # (minibatch_size, obs_dim)
                    hamiltonian = reward_model(mb_obs, current_actions) + \
                                torch.einsum("bi,bi->b", dVdx, f)  # (minibatch_size,)

                    # Actor loss and update
                    actor_loss = -hamiltonian.mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
                    actor_optimizer.step() #implement projected gradient descent, to be sure that action

                # Critic update - uses online networks only
                current_v = agent.critic(mb_obs).squeeze()
                dVdx = vmap(compute_value_grad, in_dims=(0))(mb_obs)
                with torch.no_grad():
                    current_actions = agent.actor(mb_obs)
                    r = reward_model(mb_obs, current_actions)
                    f = dynamic_model(mb_obs, current_actions)
                hamiltonian = r + torch.einsum("...i,...i->...", dVdx, f)
                hjb_residual = hamiltonian + np.log(args.gamma)*current_v
                hjb_loss = 0.5 * (hjb_residual ** 2).mean()

                # Value loss
                v_loss = 0.5 * ((current_v - b_returns[mb_inds]) ** 2).mean()
                critic_loss = v_loss * args.vf_coef + hjb_loss * args.hjb_coef

                # Critic update
                critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                critic_optimizer.step()

                # Logging
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                writer.add_scalar("losses/hjb_residual", hjb_loss.item(), global_step)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", actor_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
            use_deterministic=True
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
