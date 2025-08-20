#!/usr/bin/env python3
"""
Asteroids DQN Training Script

This script implements Deep Q-Learning (DQN) for training an AI agent to play the Asteroids game.
Converted from the Jupyter notebook for easier experimentation and command-line usage.

Usage:
    python asteroids_dqn_training.py [--frames 50000] [--lr 1e-4] [--batch-size 32] [--test-only]
"""

import argparse
import sys
import os
import time
import collections
from typing import Tuple, Optional

# Add the project root to the path
sys.path.append(os.path.abspath('.'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import gymnasium as gym
import gymnasium.spaces
from torch.utils.tensorboard import SummaryWriter

from asteroids.gym_env import AsteroidsEnv


# =============================================================================
# Environment Wrappers
# =============================================================================

class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame"""
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs, info = self.env.reset()
        self._obs_buffer.append(obs)
        return obs, info


class ProcessFrame84(gym.ObservationWrapper):
    """Convert frames to 84x84 grayscale"""
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        # frame is (height, width, 3) from our Asteroids environment
        if len(frame.shape) == 3:
            # Convert to grayscale
            img = frame[:, :, 0] * 0.299 + frame[:, :, 1] * 0.587 + frame[:, :, 2] * 0.114
        else:
            img = frame
        
        # Resize to 84x84
        resized_screen = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        x_t = np.reshape(resized_screen, [84, 84, 1])
        return x_t.astype(np.uint8)

class ProcessFrameResized(gym.ObservationWrapper):
    """Convert frames to grayscale and resize to given resolution"""
    def __init__(self, env=None, resolution=256):
        super(ProcessFrameResized, self).__init__(env)
        self.resolution = resolution
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(resolution, resolution, 1), dtype=np.uint8
        )

    def observation(self, obs):
        return ProcessFrameResized.process(obs, self.resolution)

    @staticmethod
    def process(frame, resolution):
        # frame is (height, width, 3) from our Asteroids environment
        if len(frame.shape) == 3:
            # Convert to grayscale
            img = frame[:, :, 0] * 0.299 + frame[:, :, 1] * 0.587 + frame[:, :, 2] * 0.114
        else:
            img = frame

        # Resize to resolution x resolution
        resized_screen = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
        x_t = np.reshape(resized_screen, [resolution, resolution, 1])
        return x_t.astype(np.uint8)
    
class ImageToPyTorch(gym.ObservationWrapper):
    """Convert image format to PyTorch (C, H, W)"""
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, 
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.float32
        )

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Scale pixel values to [0, 1]"""
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    """Stack multiple frames"""
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        self.n_steps = n_steps
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0), 
            dtype=dtype
        )

    def reset(self, seed=None, options=None):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class MultiBinaryToDiscreteWrapper(gym.ActionWrapper):
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


def make_asteroids_env(render_mode='rgb_array'):
    """Create and wrap the Asteroids environment for DQN training
    
    Args:
        render_mode: Rendering mode for the environment ('rgb_array', 'human', or None)
    """
    env = AsteroidsEnv(render_mode=render_mode)
    env = MultiBinaryToDiscreteWrapper(env)  # Convert to discrete actions for DQN
    env = MaxAndSkipEnv(env, skip=4)  # Frame skipping
    env = ProcessFrameResized(env, resolution=256)  # Convert to res x res grayscale
    env = ImageToPyTorch(env)  # Convert to PyTorch format (C, H, W)
    env = BufferWrapper(env, 4)  # Stack 4 frames
    return ScaledFloatFrame(env)  # Scale to [0, 1]


# =============================================================================
# DQN Model
# =============================================================================

class AsteroidsDQN_V0(nn.Module):
    """DQN network for Asteroids game"""
    def __init__(self, input_shape, n_actions):
        super(AsteroidsDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

class AsteroidsDQN_V1(nn.Module):
    """DQN network for Asteroids game"""
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=16, stride=8),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
    

# =============================================================================
# Training Components
# =============================================================================

# choose network structure
AsteroidsDQN = AsteroidsDQN_V1

Experience = collections.namedtuple('Experience', 
                                   field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), 
                np.array(dones, dtype=np.uint8), np.array(next_states))


class Agent:
    """DQN Agent for interacting with the environment"""
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state, _ = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.asarray([self.state])  # Fix NumPy compatibility
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # Take step in environment
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu", gamma=0.99):
    """Calculate DQN loss"""
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    
    # Current Q values
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    
    # Next Q values from target network
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()
    
    expected_state_action_values = next_state_values * gamma + rewards_v

    return nn.MSELoss()(state_action_values, expected_state_action_values)


# =============================================================================
# Training and Testing Functions
# =============================================================================

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
    
def test_setup():
    """Test the complete setup before starting training"""
    print("Testing setup...")
    device = get_device()
    print(f'Device: {device}')
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')

    env = make_asteroids_env()
    print(f"Environment created with observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test environment reset and step
    state, info = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    # Test random action
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"After step - State shape: {next_state.shape}, Reward: {reward}")

    # Dual network architecture
    net = AsteroidsDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = AsteroidsDQN(env.observation_space.shape, env.action_space.n).to(device)
    print("Networks created successfully")
    print(f"Input shape: {env.observation_space.shape}")
    print(f"Number of actions: {env.action_space.n}")
    print(f"Network architecture:\n{net}")

    # Test forward pass
    state_tensor = torch.tensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = net(state_tensor)
    print(f"Q-values shape: {q_values.shape}")
    print(f"Sample Q-values: {q_values[0][:10]}")  # Show first 10 values

    env.close()
    print("Setup test completed successfully!")
    return True


def train_asteroids_dqn(
    max_frames: int = 50000,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    replay_size: int = 10000,
    sync_target_frames: int = 1000,
    replay_start_size: int = 10000,
    epsilon_decay_frames: int = 100000,
    epsilon_start: float = 1.0,
    epsilon_final: float = 0.02,
    gamma: float = 0.99,
    mean_reward_bound: float = 100.0,
    save_path: str = "asteroids-best.dat",
    load_model: Optional[str] = None,
    continue_training: bool = False
):
    """
    Train the DQN agent on Asteroids
    
    Args:
        max_frames: Maximum number of training frames
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        replay_size: Size of experience replay buffer
        sync_target_frames: Frequency of target network updates
        replay_start_size: Minimum buffer size before training starts
        epsilon_decay_frames: Number of frames over which to decay epsilon
        epsilon_start: Starting epsilon value
        epsilon_final: Final epsilon value
        gamma: Discount factor
        mean_reward_bound: Target mean reward to consider solved
        save_path: Path to save the best model
        load_model: Path to pre-trained model to load and continue training from
        continue_training: If True, continue from loaded model's training state
    """
    device = get_device()
    print(f'Training device: {device}')

    # Create environment
    env = make_asteroids_env()
    print(f"Training environment: obs_space={env.observation_space}, action_space={env.action_space}")

    # Create networks
    net = AsteroidsDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = AsteroidsDQN(env.observation_space.shape, env.action_space.n).to(device)
    
    # Load pre-trained model if specified
    if load_model and os.path.exists(load_model):
        print(f"Loading pre-trained model from: {load_model}")
        net.load_state_dict(torch.load(load_model, map_location=device))
        tgt_net.load_state_dict(torch.load(load_model, map_location=device))
        print("✅ Pre-trained model loaded successfully")
        if continue_training:
            print("🔄 Continuing training from loaded model")
    elif load_model:
        print(f"⚠️  Warning: Model file not found: {load_model}")
        print("Starting training from scratch...")
    
    # Initialize TensorBoard
    writer = SummaryWriter(comment="-asteroids")

    # Create training components
    buffer = ExperienceBuffer(replay_size)
    agent = Agent(env, buffer)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Training state
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None
    epsilon = epsilon_start

    # If continuing training, adjust epsilon based on experience
    if continue_training and load_model:
        # For continuing training, start with a lower epsilon since model is already trained
        epsilon = max(epsilon_final, epsilon_start * 0.5)  # Start with 50% of initial epsilon
        print(f"🎯 Adjusted starting epsilon for continued training: {epsilon:.3f}")

    print("Starting training...")
    print(f"Target: {mean_reward_bound} mean reward over 100 episodes")
    print(f"Training for max {max_frames} frames")
    print("-" * 60)

    try:
        while frame_idx < max_frames:
            frame_idx += 1
            epsilon = max(epsilon_final, epsilon_start - frame_idx / epsilon_decay_frames)

            reward = agent.play_step(net, epsilon, device=device)
            
            if reward is not None:
                total_rewards.append(reward)
                speed = (frame_idx - ts_frame) / (time.time() - ts)
                ts_frame = frame_idx
                ts = time.time()
                mean_reward = np.mean(total_rewards[-100:])
                
                print(f"{frame_idx:6d}: done {len(total_rewards):4d} games, "
                      f"mean reward {mean_reward:7.3f}, eps {epsilon:.2f}, "
                      f"speed {speed:5.2f} f/s")
                
                # Log to TensorBoard
                writer.add_scalar("epsilon", epsilon, frame_idx)
                writer.add_scalar("speed", speed, frame_idx)
                writer.add_scalar("reward_100", mean_reward, frame_idx)
                writer.add_scalar("reward", reward, frame_idx)
                
                # Save best model
                if best_mean_reward is None or best_mean_reward < mean_reward:
                    torch.save(net.state_dict(), save_path)
                    if best_mean_reward is not None:
                        print(f"★ Best mean reward updated {best_mean_reward:.3f} -> {mean_reward:.3f}, model saved")
                    best_mean_reward = mean_reward
                    
                # Check if solved
                if mean_reward > mean_reward_bound:
                    print(f"🎉 Solved in {frame_idx} frames! Mean reward: {mean_reward:.3f}")
                    break

            # Start training when buffer is full enough
            if len(buffer) < replay_start_size:
                continue

            # Update target network
            if frame_idx % sync_target_frames == 0:
                tgt_net.load_state_dict(net.state_dict())
                print(f"Target network updated at frame {frame_idx}")

            # Training step
            optimizer.zero_grad()
            batch = buffer.sample(batch_size)
            loss_t = calc_loss(batch, net, tgt_net, device=device, gamma=gamma)
            loss_t.backward()
            optimizer.step()
            
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    finally:
        writer.close()
        env.close()
        print(f"\nTraining completed after {frame_idx} frames")
        if len(total_rewards) > 0:
            final_mean = np.mean(total_rewards[-100:])
            print(f"Final mean reward (last 100): {final_mean:.3f}")
            if best_mean_reward:
                print(f"Best mean reward achieved: {best_mean_reward:.3f}")


# =============================================================================
# Main Function and CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train DQN agent on Asteroids game')
    parser.add_argument('--frames', type=int, default=10000,
                        help='Maximum number of training frames (default: 10000)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--replay-size', type=int, default=10000,
                        help='Experience replay buffer size (default: 10000)')
    parser.add_argument('--test-only', action='store_true',
                        help='Only run setup test, do not train')
    parser.add_argument('--save-path', type=str, default='asteroids-best.dat',
                        help='Path to save the best model (default: asteroids-best.dat)')
    parser.add_argument('--target-reward', type=float, default=100.0,
                        help='Target mean reward to consider solved (default: 100.0)')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to pre-trained model to load and continue training from')
    parser.add_argument('--continue-training', action='store_true',
                        help='Continue training from loaded model (adjusts epsilon and other parameters)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Asteroids DQN Training")
    print("=" * 60)
    
    # Test setup first
    if not test_setup():
        print("❌ Setup test failed!")
        return 1
    
    if args.test_only:
        print("✅ Test completed successfully. Exiting (--test-only specified).")
        return 0
    
    print("\n" + "=" * 60)
    print("🎯 Starting Training")
    print("=" * 60)
    
    # Start training
    train_asteroids_dqn(
        max_frames=args.frames,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        mean_reward_bound=args.target_reward,
        save_path=args.save_path,
        load_model=args.load_model,
        continue_training=args.continue_training
    )
    
    print("✅ Training completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
