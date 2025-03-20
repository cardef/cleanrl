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
    hjb_coef: float = 0.5
    """coefficient for HJB residual loss"""
    hjb_policy_steps: int = 2
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
    env_id: str = "HalfCheetah-v4"
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
            nn.Linear(obs_dim + action_dim, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, obs_dim),
        )
        self.dt = 0.05  # Should match environment timestep

    def forward(self, t, x, a_sequence):
        # Calculate which action to use based on current time
        idx = torch.clamp((t / self.dt).long(), 0, a_sequence.size(1)-1)
        a = a_sequence[torch.arange(x.size(0)), idx]
        return self.net(torch.cat([x, a], dim=-1))

class DynamicModel(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.ode_func = ODEFunc(obs_dim, action_dim)
        self.dt = 0.05
        
        # TorchODE components
        self.term = to.ODETerm(self.ode_func, with_args=True)
        self.step_method = to.Tsit5(term=self.term)
        self.step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=self.term)
        self.adjoint = to.AutoDiffAdjoint(
            step_method=self.step_method,
            step_size_controller=self.step_size_controller,
        )

    def forward(self, initial_obs, action_sequences):
        batch_size, seq_len = action_sequences.shape[:2]
        
        # Create time evaluation points for entire trajectory
        t_eval = torch.stack([torch.linspace(0, self.dt*seq_len, seq_len+1)]*batch_size)
        problem = to.InitialValueProblem(
            y0=initial_obs,
            t_eval=t_eval.to(initial_obs.device),
            
        )
        sol = self.adjoint.solve(problem, args=action_sequences)
        
        # Return all predicted states except initial
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
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
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
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
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
        
        # Create trajectory validity mask
        dones_mask = dones.cpu().numpy().transpose(1, 0)  # Shape [num_envs, num_steps]
        traj_valid_mask = np.ones_like(dones_mask, dtype=bool)
        for env_idx in range(args.num_envs):
            done_indices = np.where(dones_mask[env_idx])[0]
            if len(done_indices) > 0:
                first_done = done_indices[0]
                traj_valid_mask[env_idx, first_done+1:] = False
        traj_valid_mask = torch.tensor(traj_valid_mask, device=device)

        # Reorganize data into trajectory-first format
        traj_obs = obs.permute(1, 0, 2)        # [num_envs, num_steps, obs_dim]
        traj_actions = actions.permute(1, 0, 2) # [num_envs, num_steps, action_dim]
        traj_next_obs = next_observations.permute(1, 0, 2)

        # Split into train/validation trajectories
        env_indices = torch.randperm(args.num_envs)
        split = int(0.8 * args.num_envs)
        train_idx, val_idx = env_indices[:split], env_indices[split:]

        train_obs = traj_obs[train_idx]
        train_actions = traj_actions[train_idx]
        train_next_obs = traj_next_obs[train_idx]
        train_valid_mask = traj_valid_mask[train_idx]

        val_obs = traj_obs[val_idx]
        val_actions = traj_actions[val_idx]
        val_next_obs = traj_next_obs[val_idx]
        val_valid_mask = traj_valid_mask[val_idx]

        # Dynamic model evaluation check
        print("Evaluating dynamic model...")
        with torch.no_grad():
            if len(val_obs) > 0:
                val_initial = val_obs[:, 0]
                val_pred = dynamic_model(val_initial, val_actions)
                
                val_loss = 0
                val_valid = 0
                for t in range(args.num_steps):
                    step_mask = val_valid_mask[:, t]
                    if step_mask.any():
                        val_loss += nn.MSELoss()(
                            val_pred[step_mask, t],
                            val_next_obs[step_mask, t]
                        )
                        val_valid += 1
                initial_val_mse = (val_loss / val_valid).item() if val_valid > 0 else float('inf')
            else:
                initial_val_mse = float('inf')
        
        writer.add_scalar("dynamic/initial_val_mse", initial_val_mse, iteration)
        
        if initial_val_mse <= args.hjb_dynamic_threshold:
            print(f"Dynamic model already good (val MSE {initial_val_mse:.4f} <= {args.hjb_dynamic_threshold}), skipping training")
        else:
            print("Pretraining dynamic model...")
            best_val_mse = float('inf')
            patience_counter = 0

            for pretrain_epoch in trange(5000, desc="Pretraining"):
                if len(train_obs) == 0:
                    break  # No training data available
                
                # Random batch of trajectories
                batch_idx = torch.randint(0, len(train_obs), (args.minibatch_size,))
                initial_obs = train_obs[batch_idx, 0]
                batch_actions = train_actions[batch_idx]
                batch_next_obs = train_next_obs[batch_idx]
                batch_mask = train_valid_mask[batch_idx]

                # Forward pass
                dynamic_optimizer.zero_grad()
                pred_traj = dynamic_model(initial_obs, batch_actions)
                
                # Calculate masked loss
                loss = 0
                valid_steps = 0
                for t in range(args.num_steps):
                    step_mask = batch_mask[:, t]
                    if step_mask.any():
                        loss += nn.MSELoss()(
                            pred_traj[step_mask, t], 
                            batch_next_obs[step_mask, t]
                        )
                        valid_steps += 1
                
                if valid_steps == 0:
                    continue
                    
                loss = loss / valid_steps
                loss.backward()
                dynamic_optimizer.step()

                # Validation
                with torch.no_grad():
                    if len(val_obs) > 0:
                        val_initial = val_obs[:, 0]
                        val_pred = dynamic_model(val_initial, val_actions)
                        
                        val_loss = 0
                        val_valid = 0
                        for t in range(args.num_steps):
                            step_mask = val_valid_mask[:, t]
                            if step_mask.any():
                                val_loss += nn.MSELoss()(
                                    val_pred[step_mask, t],
                                    val_next_obs[step_mask, t]
                                )
                                val_valid += 1
                        val_mse = (val_loss / val_valid).item() if val_valid > 0 else float('inf')
                    else:
                        val_mse = float('inf')

                # Early stopping logic
                if val_mse < (best_val_mse - args.hjb_dynamic_min_delta):
                    best_val_mse = val_mse
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= args.hjb_dynamic_patience:
                        break
            if len(val_obs) > 0:
                # Re-calculate final validation MSE
                with torch.no_grad():
                    val_initial = val_obs[:, 0]
                    val_pred = dynamic_model(val_initial, val_actions)
                    
                    val_loss = 0
                    val_valid = 0
                    for t in range(args.num_steps):
                        step_mask = val_valid_mask[:, t]
                        if step_mask.any():
                            val_loss += nn.MSELoss()(
                                val_pred[step_mask, t],
                                val_next_obs[step_mask, t]
                            )
                            val_valid += 1
                    final_val_mse = (val_loss / val_valid).item() if val_valid > 0 else float('inf')
                print(f"Pretraining complete. Final Val MSE: {final_val_mse:.4f}")
                writer.add_scalar("dynamic/final_val_mse", final_val_mse, iteration)
            else:
                print("Pretraining complete. No validation data available.")


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
                    dVdx = vmap(compute_value_grad, in_dims=(0))(mb_obs)

                    # Hamiltonian calculation
                    current_actions = agent.actor(mb_obs)
                    hamiltonian = reward_model(mb_obs, current_actions) + \
                                torch.einsum("...i,...i->...", dVdx, dynamic_model(mb_obs, current_actions))

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
