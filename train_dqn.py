"""
This is my attempt to build the training process more or less by hand after some exploration
The goals are:
1. practice the process by hand for deeper understanding
2. cleaner code

"""

import os
from pathlib import Path
from datetime import datetime
import numpy as np
import collections
import matplotlib.pyplot as plt
import yaml

import gymnasium as gym
import ale_py
from gymnasium.wrappers import (
    AtariPreprocessing, 
    FrameStackObservation, 
    MaxAndSkipObservation, 
    ResizeObservation,
    GrayscaleObservation
)
from torch.utils.tensorboard import SummaryWriter
import argparse
import torch
import torch.nn as nn
from asteroids.gym_env import AsteroidsEnv


def get_device():
    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        return "cuda"
    # Check for MPS (Apple Silicon GPUs)
    elif torch.backends.mps.is_available():
        return "mps"
    # Fallback to CPU
    else:
        return "cpu"


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config_copy(config: dict, checkpoint_dir: Path) -> None:
    """Save a copy of the configuration to the checkpoint directory"""
    config_path = checkpoint_dir / "config.yaml"
    
    # Preserve the order from get_default_config()
    default_config = get_default_config()
    ordered_config = {}
    
    # First add all keys in the default order
    for key in default_config.keys():
        if key in config:
            ordered_config[key] = config[key]
    
    # Then add any additional keys that might exist in config but not in default
    for key, value in config.items():
        if key not in ordered_config:
            ordered_config[key] = value
    
    with open(config_path, 'w') as f:
        yaml.dump(ordered_config, f, default_flow_style=False, indent=2, sort_keys=False)
    print(f"Configuration saved to: {config_path}")


def get_default_config() -> dict:
    """Get default configuration values"""
    return {
        'game': 'beamrider',
        'max_steps': 100000,
        'replay_buffer_size': 100000,
        'learning_rate': 0.0001,
        'batch_size': 32,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.1,
        'epsilon_decay_frames': 100000,
        'checkpoint_interval': 10000,
        'eval_episode_interval': 5,
        'load_model': None,
        'comment': ''
    }
    
# game wrappers

class MultiBinaryToDiscreteCombinationWrapper(gym.ActionWrapper):
    """Convert MultiBinary action space to Discrete for DQN compatibility"""
    def __init__(self, env):
        super().__init__(env)
        # For 5 binary actions, we have 2^5 = 32 possible combinations
        self.n_combination = 2 ** env.action_space.n # type: ignore
        self.action_space = gym.spaces.Discrete(self.n_combination)

    def action(self, action):
        # Convert discrete action to MultiBinary
        binary_action = np.zeros(self.n_combination, dtype=np.int32)
        for i in range(self.n_combination):
            binary_action[i] = (action >> i) & 1
        return binary_action
    
class MultiBinaryToSingleDiscreteAction(gym.ActionWrapper):
    """Convert MultiBinary action space to Single discrete action for DQN compatibility"""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(env.action_space.n) # type: ignore

    def action(self, action: int):
        # Convert discrete action to MultiBinary
        multi_binary_action = np.zeros(self.action_space.n, dtype=np.int32) # type: ignore
        multi_binary_action[action] = 1
        return multi_binary_action

class ScaleObservation(gym.ObservationWrapper):
    """Scale pixel values to [0, 1]"""
    def __init__(self, env: gym.Env):
        assert isinstance(env.observation_space, gym.spaces.Box)
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return np.asarray(observation, dtype=np.float32) / 255.0

# env observation shape = (T, H, W, C)
#  T frame stacking time dimension (squeeze if no frame stacking)
#  C number of channels (squeezed for grayscale, 3 for RGB)
# = (4, 84, 84) = (T, H, W) for default grayscale_obs=True frame_stack=4
# = (4, 84, 84, 3) = (T, H, W, C) for grayscale_obs=False 
# = (84, 84) for frame_stack=1 and grayscale_obs=True

def make_atari_env(env_id:str, render_mode:str = "rgb_array", max_episode_steps:int = 10000, screen_size=(84,84),
                   frame_stack:int = 4, scale_obs:bool = True,
                   **kwargs):
    env = gym.make(env_id, render_mode=render_mode, max_episode_steps=max_episode_steps, frameskip=1) 
    # disable initial frame skipping because AtariPreprocessing does that, too
    env = AtariPreprocessing(env, screen_size=screen_size, scale_obs=scale_obs, **kwargs) # kwargs ex: grayscale_obs = True
    if frame_stack > 1:
        env = FrameStackObservation(env, frame_stack)
    return env

def make_asteroids_env(render_mode:str = "rgb_array", screen_size=(84,84), grayscale_obs:bool = True, single_action:bool = False,
                       scale_obs:bool = True, frame_stack:int = 4):
    env = AsteroidsEnv(render_mode=render_mode)
    env = MaxAndSkipObservation(env, skip=4)
    env = ResizeObservation(env, shape=screen_size)
    if grayscale_obs:
        env = GrayscaleObservation(env)
    if scale_obs:
        env = ScaleObservation(env)
    if frame_stack > 1:
        env = FrameStackObservation(env, frame_stack)
    if single_action:
        env = MultiBinaryToSingleDiscreteAction(env)
    else:
        env = MultiBinaryToDiscreteCombinationWrapper(env)
    return env


# DQN network
class AtariDQN(nn.Module):
    """Q Learning network mapping pixel observations to action values"""
    def __init__(self, input_shape, n_action):
        # input_shape = (4, 84, 84)=(T, H, W) for grayscale_obs
        # T is frame stacking time dimension
        # Note:
        # input shape (4, 84, 84)   -> conv output shape (64, 7, 7) -> flattened size 3136
        # input shape (4, 128, 128) -> conv output shape (64, 12, 12) -> flattened size 9216

        print("AtariDQN initialized with input shape:", input_shape)
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_action)
        )

    def _get_conv_out(self, input_shape):
        dummy = torch.zeros(1, *input_shape)
        output = self.conv(dummy)
        # print("conv output size:", output.shape[1:])
        return output.view(1, -1).size(1)

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size()[0], -1))


# Experience and replay buffer
Experience = collections.namedtuple("Experience", ["obs", "action", "reward", "done", "next_obs"])

class ExperienceBuffer:
    def __init__(self, capacity) -> None:
        self.buffer = collections.deque(maxlen=capacity)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def append(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size:int, as_torch_tensor:bool = False, device='cpu'):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        out = (
            np.array(states, dtype=np.float32), 
            np.array(actions, dtype=np.int64), 
            np.array(rewards, dtype=np.float32), 
            np.array(dones, dtype=np.bool), 
            np.array(next_states, dtype=np.float32)
            )
        if as_torch_tensor:
            out =(torch.tensor(x, device=device) for x in out)
        return out

# Agent: handle interaction with environment and keep experience buffer for replay
class Agent:
    def __init__(self, env:gym.Env, buffer:ExperienceBuffer):
        self.env = env
        self.experience_buffer = buffer
        self.curr_obs = None
        self.episode_reward = 0.0
        self.reset_env()

    def reset_env(self):
        self.episode_reward = 0.0
        self.curr_obs, _ = self.env.reset()

    def play_step(self, net:nn.Module, epsilon:float = 0.0, device:str="cpu", update_buffer=True) -> None | float:
        episode_reward = None
        # choose an action
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                obs = torch.tensor(self.curr_obs, device=device, dtype=torch.float32).unsqueeze(0) # add batch dimension
                q_values = net(obs)
                action = q_values.argmax(dim=1).item()
        
        next_obs, reward, truncated, terminated, _ = self.env.step(action)
        is_done = truncated or terminated
        self.episode_reward += reward # type: ignore

        # update experience buffer
        if update_buffer:
            self.experience_buffer.append(Experience(self.curr_obs, action, reward, is_done, next_obs))
        self.curr_obs = next_obs
        if is_done:
            episode_reward = self.episode_reward
            self.reset_env()

        return episode_reward


def train(env, config):
    device = get_device()
    print("Using device:", device)

    game = config['game']
    buffer_size = config['replay_buffer_size']
    max_steps = config['max_steps']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    epsilon_start = config['epsilon_start']
    epsilon_end = config['epsilon_end']
    epsilon_decay_frames = config['epsilon_decay_frames']
    checkpoint_interval = config['checkpoint_interval']
    eval_episode_interval = config['eval_episode_interval']
    comment = config['comment']
    gamma = config['gamma']

    current_time = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    run_id = f"{current_time}_{game}"
    if comment:
        run_id = f"{run_id}_{comment}"
    print('Run ID:', run_id)

    checkpoint_dir = Path('./checkpoints') / run_id
    log_dir = Path('./runs') / run_id

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the configuration
    save_config_copy(config, checkpoint_dir)

    writer = SummaryWriter(log_dir=log_dir)

    hyperparameters = {
        "game": game,
        "buffer_size": buffer_size,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay_frames": epsilon_decay_frames,
        "gamma": gamma,
        "frames": 0
        }
    metrics = {
        "best_eval_reward": float("-inf")
    }


    buffer = ExperienceBuffer(capacity=buffer_size)
    agent = Agent(env, buffer) # when agent play step, it updates the buffer by default
    net = AtariDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = AtariDQN(env.observation_space.shape, env.action_space.n).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Load saved model if specified
    if config.get('load_model'):
        try:
            print(f"Loading model from: {config['load_model']}")
            checkpoint = torch.load(config['load_model'], map_location=device)
            net.load_state_dict(checkpoint)
            tgt_net.load_state_dict(checkpoint)  # Start with same weights for target network
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(f"❌ Failed to load model from: {config['load_model']}")
            print("❌ Training stopped. Please check the model path and try again.")
            return

    # fill buffer before start training
    initial_experience_epsilon = config['epsilon_start']
    print(f"Filling initial experience buffer with epsilon {initial_experience_epsilon} ...")
    while len(buffer) < buffer_size:
        if len(buffer) % (buffer_size // 5) == 0:
            print(len(buffer))
        agent.play_step(net, epsilon=initial_experience_epsilon, device=device)
    agent.reset_env()
    print("Experience buffer filled", len(buffer))

    # start training
    frame_idx = 0
    episode_idx = 0
    best_eval_reward = float("-inf")
    best_eval_reward_frame_idx = 0
    try:
        while frame_idx < max_steps:
            frame_idx += 1
            epsilon = epsilon_start - (epsilon_start - epsilon_end) * (frame_idx / epsilon_decay_frames)
            epsilon = max(epsilon, epsilon_end)
            # play steps and update buffer until an episode is done
            episode_reward = agent.play_step(net, epsilon=epsilon, device=device)


            # sample from buffer 
            obs, actions, rewards, dones, next_obs = buffer.sample(batch_size=batch_size, as_torch_tensor=True, device=device)
            # train online network
            # 1. target/actual Q(s,a) = reward + gamma * max_a{tgt_net(next_state)}
            Q_target = rewards + gamma * tgt_net(next_obs).max(dim=1)[0] * (~dones) # if done set to 0
            Q_target = Q_target.detach() 
            # 2. current/predicted Q(s,a) = net(state)_action
            Q_pred = net(obs).gather(1, actions.unsqueeze(-1)).squeeze(-1)  # type: ignore
            # 3. loss
            loss = nn.MSELoss()(Q_pred, Q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # sync target network periodically
            if frame_idx % 1000 == 0:
                tgt_net.load_state_dict(net.state_dict())
                # print("Target network updated")

            if frame_idx % checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"dqn_{frame_idx}.pth"
                torch.save(net.state_dict(), str(checkpoint_path))

            # per episode operation 
            if episode_reward is not None:
                episode_idx += 1
                # logs
                print(f"frame {frame_idx}, (episode {episode_idx}), reward {episode_reward:.2f}, epsilon {epsilon:.2f}")
                writer.add_scalar("train/epsilon", epsilon, frame_idx)
                writer.add_scalar("train/reward", episode_reward, frame_idx)

                # evaluate deterministic (zero-epsilon) periodically
                if episode_idx % eval_episode_interval == 0:
                    print("Evaluating...")
                    rewards = []
                    for _ in range(5):
                        episode_reward = None
                        while episode_reward is None:
                            episode_reward = agent.play_step(net, epsilon=0, device=device, update_buffer=False)
                        rewards.append(episode_reward)
                    mean_eval_reward = sum(rewards) / len(rewards)
                    writer.add_scalar("eval/reward", mean_eval_reward, frame_idx)
                    print("Evaluation results:", mean_eval_reward, "best:", best_eval_reward)

                    if mean_eval_reward > best_eval_reward:
                        best_eval_reward = max(best_eval_reward, mean_eval_reward)
                        best_eval_reward_frame_idx = frame_idx
                        print("New best evaluation reward:", mean_eval_reward, "checkpoint saved")
                        torch.save(net.state_dict(), checkpoint_dir / f"dqn_best.pth")
                
                
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user. Current frame: {frame_idx}")
    finally:
        metrics["best_eval_reward"] = best_eval_reward
        writer.add_hparams(hyperparameters, metrics, global_step=best_eval_reward_frame_idx)
        writer.close()
        env.close()
        print("Training complete")


def main():
    parser = argparse.ArgumentParser(description='Train DQN agent on Beamrider or (not the standard) Asteroids game')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML configuration file')
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        print(f"✅ Loaded configuration from: {args.config}")
    except FileNotFoundError:
        print(f"❌ Config file not found: {args.config}")
        print("🔧 Creating default config.yaml...")
        config = get_default_config()
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
        print("✅ Default config.yaml created. Please modify it as needed and run again.")
        return
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Validate configuration
    if config['game'] not in ['beamrider', 'asteroids']:
        print(f"❌ Invalid game: {config['game']}. Must be 'beamrider' or 'asteroids'")
        return
    
    # Initialize environment
    if config['game'] == 'asteroids':
        env = make_asteroids_env(single_action=False) 
    elif config['game'] == 'beamrider':
        env = make_atari_env("ALE/BeamRider-v5", grayscale_obs=True, max_episode_steps=100000)
    else:
        raise ValueError(f"Unsupported game: {config['game']}")

    train(env, config)


if __name__ == "__main__":
    main()