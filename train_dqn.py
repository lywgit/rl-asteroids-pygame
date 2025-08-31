"""
This is my attempt to build the training process more or less by hand after some exploration
The goals are:
1. practice the process by hand for deeper understanding
2. cleaner code

"""

from csv import writer
from turtle import speed
import numpy as np
import collections
import matplotlib.pyplot as plt

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
    
# game wrappers

class MultiBinaryToDiscreteCombinationWrapper(gym.ActionWrapper):
    """Convert MultiBinary action space to Discrete for DQN compatibility"""
    def __init__(self, env):
        super().__init__(env)
        # For 5 binary actions, we have 2^5 = 32 possible combinations
        self.action_space = gym.spaces.Discrete(32)
        
    def action(self, action):
        # Convert discrete action to MultiBinary
        binary_action = np.zeros(5, dtype=np.int32)
        for i in range(5):
            binary_action[i] = (action >> i) & 1
        return binary_action
    
class MultiBinaryToSingleDiscreteAction(gym.ActionWrapper):
    """Convert MultiBinary action space to Single discrete action for DQN compatibility"""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(env.action_space.n)
        
    def action(self, action: int):
        # Convert discrete action to MultiBinary
        multi_binary_action = np.zeros(self.action_space.n, dtype=np.int32)
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

def make_asteroids_env(render_mode:str = "rgb_array", screen_size=(128,128), grayscale_obs:bool = True, 
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
    env = MultiBinaryToSingleDiscreteAction(env)
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
        self._reset()

    def _reset(self):
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
            self._reset()

        return episode_reward


def train(env, args):
    device = get_device()
    print("Using device:", device)
    
    # Initialize TensorBoard
    writer = SummaryWriter(comment=f"-{args.game}")

    buffer_size = args.replay_buffer_size
    max_steps = args.max_steps
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay_frames = 100000
    checkpoint_interval = 10000
    gamma = 0.99

    buffer = ExperienceBuffer(capacity=buffer_size)
    agent = Agent(env, buffer) # when agent play step, it updates the buffer
    net = AtariDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = AtariDQN(env.observation_space.shape, env.action_space.n).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # fill buffer before start training
    print("Filling experience buffer...")
    while len(buffer) < buffer_size:
        if len(buffer) % (buffer_size // 10) == 0:
            print(len(buffer))
        agent.play_step(net, epsilon=1)
    print("Experience buffer filled", len(buffer))

    # start training
    frame_idx = 0
    episode_idx = 0
    best_eval_reward = float("-inf")
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
            Q_pred = net(obs).gather(1, actions.unsqueeze(-1)).squeeze(-1)
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
                torch.save(net.state_dict(), f"dqn_{args.game}_{frame_idx}.pth")

            # update episode index 
            if episode_reward is not None:
                episode_idx += 1
                print(f"frame {frame_idx}, (episode {episode_idx}), reward {episode_reward:.2f}, epsilon {epsilon:.2f}")
                # logs
                writer.add_scalar("epsilon", epsilon, frame_idx)
                writer.add_scalar("reward", episode_reward, frame_idx)

            # evaluate with zero epsilon periodically
            if episode_reward is not None and episode_idx % 5 == 0:
                print("Evaluating...")
                rewards = []
                for _ in range(5):
                    episode_reward = None
                    while episode_reward is None:
                        episode_reward = agent.play_step(net, epsilon=0, device=device, update_buffer=False)
                    rewards.append(episode_reward)
                mean_eval_reward = sum(rewards) / len(rewards)
                print("Evaluation results:", mean_eval_reward, "best:", best_eval_reward)
                if mean_eval_reward > best_eval_reward:
                    print("New best evaluation reward:", mean_eval_reward)
                    torch.save(net.state_dict(), f"dqn_{args.game}_best.pth")
                best_eval_reward = max(best_eval_reward, mean_eval_reward)
                writer.add_scalar("eval_reward", mean_eval_reward, frame_idx)

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    finally:
        writer.close()
        env.close()
        print("Training complete")


def main():
    parser = argparse.ArgumentParser(description='Train DQN agent on Beamrider or (not the standard) Asteroids game')
    parser.add_argument('game', choices=['beamrider', 'asteroids'], help='Game to train on')
    parser.add_argument('--max_steps', type=int, help='Maximum training steps', default=100000)
    parser.add_argument('--replay_buffer_size', type=int, help='Replay buffer size', default=10000)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=256)
    args = parser.parse_args()
    print(args)
    # initialize environment
    if args.game == 'asteroids':
        env = make_asteroids_env()
    elif args.game == 'beamrider':
        env = make_atari_env("ALE/BeamRider-v5", grayscale_obs=True)
    else:
        raise

    train(env, args)



if __name__ == "__main__":
    main()