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
from torchdiffeq import odeint_adjoint as odeint
from sklearn.metrics import r2_score
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
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -1, 1))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -1, 1))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ODEFunc(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, 256)),
            nn.SiLU(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.SiLU(),
            layer_init(nn.Linear(256, input_dim)),
        )
        
    def forward(self, t, x):
        return self.net(x)

class DynamicModel(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.ode_func = ODEFunc(obs_dim + action_dim)
        self.dt = 0.05  # Should match environment timestep
        
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        next_x = odeint(self.ode_func, x, torch.tensor([0.0, self.dt],).to(x.device), method='rk4', options=dict(step_size=self.dt/5))[-1]
        return next_x[:, :obs.shape[1]]


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
    
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


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
    dynamic_optimizer = optim.AdamW(dynamic_model.parameters(), lr=1e-3)
    
    # Initialize reward model
    reward_model = RewardModel(obs_dim, action_dim).to(device)
    reward_optimizer = optim.AdamW(reward_model.parameters(), lr=1e-3)
    
    optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_observations = torch.zeros_like(obs).to(device)  # Store next observations

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

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
        
        # Dynamic model pretraining
        print("Pretraining dynamic model...")
        # Prepare data once before training loop
        b_obs = obs.reshape(-1, obs_dim)
        b_actions = actions.reshape(-1, action_dim)
        b_next_obs = next_observations.reshape(-1, obs_dim)
        b_dones = dones.reshape(-1)
        
        mask = b_dones == 0
        if mask.sum() == 0:
            print("No valid transitions for pretraining")
        else:
            # Split into train/validation (80/20)
            valid_obs = b_obs[mask]
            valid_actions = b_actions[mask]
            valid_next_obs = b_next_obs[mask]
            
            indices = torch.randperm(valid_obs.size(0))
            split = int(0.8 * valid_obs.size(0))
            train_idx, val_idx = indices[:split], indices[split:]
            
            train_obs, train_actions, train_next_obs = valid_obs[train_idx], valid_actions[train_idx], valid_next_obs[train_idx]
            val_obs, val_actions, val_next_obs = valid_obs[val_idx], valid_actions[val_idx], valid_next_obs[val_idx]

        best_val_mse = float('inf')
        patience_counter = 0
        patience, min_delta = 5, 1e-5  # Wait 5 epochs for >0.00001 improvement
        for pretrain_epoch in trange(5000, desc="Pretraining"):
            if mask.sum() == 0:
                break  # Skip if no valid data

            # Training step
            dynamic_optimizer.zero_grad()
            pred_train = dynamic_model(train_obs, train_actions)
            loss = nn.MSELoss()(pred_train, train_next_obs)
            loss.backward()
            dynamic_optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                # Training metrics
                train_mse = loss.item()
                train_mae = nn.L1Loss()(pred_train, train_next_obs).item()
                train_r2 = r2_score(train_next_obs.cpu().numpy(), pred_train.detach().cpu().numpy())
                
                # Validation metrics
                pred_val = dynamic_model(val_obs, val_actions)
                val_mse = nn.MSELoss()(pred_val, val_next_obs).item()
                val_mae = nn.L1Loss()(pred_val, val_next_obs).item()
                val_r2 = r2_score(val_next_obs.cpu().numpy(), pred_val.detach().cpu().numpy())
            
            # Log both training and validation metrics
            writer.add_scalar("dynamic/pretrain_train_mse", train_mse, pretrain_epoch)
            writer.add_scalar("dynamic/pretrain_train_mae", train_mae, pretrain_epoch) 
            writer.add_scalar("dynamic/pretrain_train_r2", train_r2, pretrain_epoch)
            writer.add_scalar("dynamic/pretrain_val_mse", val_mse, pretrain_epoch)
            writer.add_scalar("dynamic/pretrain_val_mae", val_mae, pretrain_epoch)
            writer.add_scalar("dynamic/pretrain_val_r2", val_r2, pretrain_epoch)
            
            # Early stopping check
            if val_mse < (best_val_mse - min_delta):
                best_val_mse = val_mse
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        print(f"Pretraining complete. Val MSE: {val_mse:.4f}, Val R²: {val_r2:.2f}")

        # Reward model pretraining
        print("Pretraining reward model...")
        b_rewards = rewards.reshape(-1)
        
        # Use all transitions since rewards are valid even when done
        indices = torch.randperm(b_obs.size(0))
        split = int(0.8 * b_obs.size(0))
        train_idx_r, val_idx_r = indices[:split], indices[split:]
        
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

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # Train dynamic model with current batch
                dynamic_optimizer.zero_grad()
                pred_next_obs = dynamic_model(b_obs[mb_inds], b_actions[mb_inds])
                dynamic_loss = nn.MSELoss()(pred_next_obs, b_next_obs[mb_inds])
                dynamic_loss.backward()
                dynamic_optimizer.step()
                
                # Log dynamic model metrics
                with torch.no_grad():
                    mse = dynamic_loss.item()
                    mae = nn.L1Loss()(pred_next_obs, b_next_obs[mb_inds]).item()
                    r2 = r2_score(b_next_obs[mb_inds].cpu().numpy(), pred_next_obs.detach().cpu().numpy())
                    writer.add_scalar("dynamic/mse", mse, global_step)

                # Train reward model
                reward_optimizer.zero_grad()
                pred_rewards = reward_model(b_obs[mb_inds], b_actions[mb_inds])
                reward_loss = nn.MSELoss()(pred_rewards, b_rewards[mb_inds])
                reward_loss.backward()
                reward_optimizer.step()
                
                writer.add_scalar("reward/mse", reward_loss.item(), global_step)
                writer.add_scalar("reward/mae", nn.L1Loss()(pred_rewards, b_rewards[mb_inds]).item(), global_step)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
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
