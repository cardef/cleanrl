# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torchode as to
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
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
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""


def make_env(env_id, seed, idx, capture_video, run_name):
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


# ALGO LOGIC: initialize agent here:

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
        # Ensure action has correct dimensionality
        a = a_sequence[torch.arange(x.size(0)), idx].unsqueeze(-1) if a_sequence.dim() == 2 else a_sequence[torch.arange(x.size(0)), idx]
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

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

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
    
class HJBCritic(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.SiLU(),
            layer_init(nn.Linear(256, 256)),
            nn.SiLU(),
            layer_init(nn.Linear(256, 1)),
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)  # Output shape: (batch_size,)


class HJBActor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = HJBActor(envs).to(device)
    critic = HJBCritic(envs).to(device)
    critic_optimizer = optim.AdamW(list(critic.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.AdamW(list(actor.parameters()), lr=args.learning_rate)
    
    # Initialize dynamic and reward models
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    action_dim = np.prod(envs.single_action_space.shape)
    dynamic_model = DynamicModel(obs_dim, action_dim).to(device)
    reward_model = RewardModel(obs_dim, action_dim).to(device)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

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

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            mb_obs = data.observations

            # Compute value and gradients
            current_v = critic(mb_obs)
            grads = torch.autograd.grad(
                current_v.sum(), mb_obs, create_graph=True, retain_graph=True
            )[0]

            # Get current actions from policy
            current_actions = actor(mb_obs)

            # Compute dynamics and rewards
            with torch.no_grad():  # Assuming fixed models
                r = reward_model(mb_obs, current_actions)
                f = dynamic_model.ode_func(
                    torch.tensor(0.0, device=device),
                    mb_obs,
                    current_actions.unsqueeze(1)
                )

            # Calculate HJB residual
            hamiltonian = r + torch.einsum("...i,...i->...", grads, f)
            hjb_residual = hamiltonian + np.log(args.gamma) * current_v
            critic_loss = 0.5 * (hjb_residual ** 2).mean()

            # Critic optimization
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            if global_step % args.policy_frequency == 0:
                # Compute value gradient for policy improvement
                with torch.no_grad():
                    current_v = critic(mb_obs)
                    grads = torch.autograd.grad(current_v.sum(), mb_obs)[0]
                
                # Get predicted dynamics
                current_actions = actor(mb_obs)
                f = dynamic_model.ode_func(
                    torch.tensor(0.0, device=device),
                    mb_obs,
                    current_actions.unsqueeze(1)
                )
                
                # Maximize Hamiltonian (HJB optimality condition)
                actor_loss = -(grads * f).sum(dim=1).mean()
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

            if global_step % 100 == 0:
                writer.add_scalar("losses/critic_values", current_q_values.mean().item(), global_step)
                writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), critic.state_dict()), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ddpg_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(HJBActor, HJBCritic),
            device=device,
            exploration_noise=args.exploration_noise,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DDPG", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
