#!/usr/bin/env python3
"""
BeamRider DQN Training Script

This script implements Deep Q-Learning (DQN) for training an AI agent to play the BeamRider game.
Adapted from the Asteroids DQN training script.

Usage:
    python beamrider_demo.py [--frames 50000] [--lr 1e-4] [--batch-size 32] [--test-only] [--demo-only]
"""

import argparse
import sys
import os
import time
import collections
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import gymnasium as gym
import gymnasium.spaces
from torch.utils.tensorboard import SummaryWriter
import ale_py


# =============================================================================
# Environment Wrappers
# =============================================================================

class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame"""
    def __init__(self, env, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated, truncated, info = None, None, None
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += float(reward)
            if terminated or truncated:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info


class ProcessFrame84(gym.ObservationWrapper):
    """Convert frames to 84x84 grayscale"""
    def __init__(self, env):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, observation):
        return ProcessFrame84.process(observation)

    @staticmethod
    def process(frame):
        # frame is (height, width, 3) from BeamRider environment
        if len(frame.shape) == 3:
            # Convert to grayscale
            img = frame[:, :, 0] * 0.299 + frame[:, :, 1] * 0.587 + frame[:, :, 2] * 0.114
        else:
            img = frame
        
        # Resize to 84x84
        resized_screen = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        x_t = np.reshape(resized_screen, [84, 84, 1])
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
    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


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

    def reset(self, **kwargs):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


def make_beamrider_env(render_mode='rgb_array', max_episode_steps=None):
    """Create and wrap the BeamRider environment for DQN training
    
    Args:
        render_mode: Rendering mode for the environment ('rgb_array', 'human', or None)
        max_episode_steps: Maximum steps per episode (None for default, -1 for unlimited)
    """
    # Register ALE environments with Gymnasium
    gym.register_envs(ale_py)
    
    env = gym.make("ALE/BeamRider-v5", render_mode=render_mode, max_episode_steps=max_episode_steps)
    env = MaxAndSkipEnv(env, skip=4)  # Frame skipping
    env = ProcessFrame84(env)  # Convert to 84x84 grayscale
    env = ImageToPyTorch(env)  # Convert to PyTorch format (C, H, W)
    env = BufferWrapper(env, 4)  # Stack 4 frames
    return ScaledFloatFrame(env)  # Scale to [0, 1]


# =============================================================================
# DQN Model
# =============================================================================

class BeamRiderDQN(nn.Module):
    """DQN network for BeamRider game"""
    def __init__(self, input_shape, n_actions):
        super().__init__()

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


# =============================================================================
# Training Components
# =============================================================================

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
            state_a = np.asarray([self.state])
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
    print("Testing BeamRider setup...")
    device = get_device()
    print(f'Device: {device}')
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')

    env = make_beamrider_env()
    print(f"Environment created with observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test environment reset and step
    state, info = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    # Test random action
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"After step - State shape: {next_state.shape}, Reward: {reward}")

    # Create networks
    net = BeamRiderDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = BeamRiderDQN(env.observation_space.shape, env.action_space.n).to(device)
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
    print("BeamRider setup test completed successfully!")
    return True


def train_beamrider_dqn(
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
    mean_reward_bound: float = 1000.0,  # BeamRider typically has higher scores
    save_path: str = "beamrider-best.dat",
    load_model: Optional[str] = None,
    continue_training: bool = False,
    checkpoint_interval: int = 10000,
    max_episode_steps: Optional[int] = None
):
    """
    Train the DQN agent on BeamRider
    """
    device = get_device()
    print(f'Training device: {device}')

    # Create environment
    env = make_beamrider_env(max_episode_steps=max_episode_steps)
    print(f"Training environment: obs_space={env.observation_space}, action_space={env.action_space}")

    # Create networks
    net = BeamRiderDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = BeamRiderDQN(env.observation_space.shape, env.action_space.n).to(device)
    
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
    writer = SummaryWriter(comment="-beamrider")

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
    recent_losses = collections.deque(maxlen=100)  # Track recent losses for averaging
    last_eval_episode = 0  # Track the last episode we evaluated to avoid duplicate evaluations

    # If continuing training, adjust epsilon based on experience
    if continue_training and load_model:
        epsilon = max(epsilon_final, epsilon_start * 0.5)
        print(f"🎯 Adjusted starting epsilon for continued training: {epsilon:.3f}")

    print("Starting BeamRider training...")
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
                
                # Include loss information in the output
                loss_info = ""
                if recent_losses:
                    mean_loss = np.mean(recent_losses)
                    loss_info = f", loss {mean_loss:.6f}"
                
                print(f"{frame_idx:6d}: done {len(total_rewards):4d} games, "
                      f"mean reward {mean_reward:7.3f}, eps {epsilon:.2f}"
                      f"{loss_info}, speed {speed:5.2f} f/s")
                
                # Log to TensorBoard
                writer.add_scalar("epsilon", epsilon, frame_idx)
                writer.add_scalar("speed", speed, frame_idx)
                writer.add_scalar("reward_100", mean_reward, frame_idx)
                writer.add_scalar("reward", reward, frame_idx)
                
                # Save best model based on deterministic (epsilon=0) performance every 50 episodes
                if len(total_rewards) % 50 == 0 and len(total_rewards) > last_eval_episode:
                    last_eval_episode = len(total_rewards)  # Mark this episode as evaluated
                    print(f"🎯 Evaluating model performance with epsilon=0 at episode {len(total_rewards)}...")
                    # Create separate environment for evaluation to avoid interfering with training
                    eval_env = make_beamrider_env(max_episode_steps=max_episode_steps)
                    eval_rewards = []
                    
                    for eval_ep in range(10):  # Test 10 episodes with deterministic play for more reliable evaluation
                        eval_state, _ = eval_env.reset()
                        eval_total_reward = 0.0
                        eval_done = False
                        
                        while not eval_done:
                            # Use deterministic action selection (epsilon=0)
                            state_a = np.asarray([eval_state])
                            state_v = torch.tensor(state_a).to(device)
                            with torch.no_grad():
                                q_vals_v = net(state_v)
                                _, act_v = torch.max(q_vals_v, dim=1)
                                action = int(act_v.item())
                            
                            eval_state, eval_reward, terminated, truncated, _ = eval_env.step(action)
                            eval_total_reward += float(eval_reward)
                            eval_done = terminated or truncated
                        
                        eval_rewards.append(eval_total_reward)
                    
                    eval_env.close()
                    eval_mean = np.mean(eval_rewards)
                    print(f"🔍 DEBUG: eval_mean_reward={eval_mean:.3f}, best_mean_reward={best_mean_reward}, training_mean={mean_reward:.3f}")
                    
                    if best_mean_reward is None or best_mean_reward < eval_mean:
                        torch.save(net.state_dict(), save_path)
                        if best_mean_reward is not None:
                            print(f"★ Best deterministic reward updated {best_mean_reward:.3f} -> {eval_mean:.3f}, model saved")
                        else:
                            print(f"★ First best model saved with deterministic reward {eval_mean:.3f}")
                        best_mean_reward = eval_mean
                    else:
                        print(f"  No update: deterministic {eval_mean:.3f} <= {best_mean_reward:.3f}")
                # else:
                    # print(f"🔍 DEBUG: training episode {len(total_rewards)}, reward={reward:.3f}, mean_100={mean_reward:.3f}")
                    
                # Check if solved
                if mean_reward > mean_reward_bound:
                    print(f"🎉 Solved in {frame_idx} frames! Mean reward: {mean_reward:.3f}")
                    break

            # Save model checkpoint every checkpoint_interval frames
            if frame_idx % checkpoint_interval == 0 or frame_idx == 1:
                base_name, ext = os.path.splitext(save_path)
                checkpoint_path = f"{base_name}_frame_{frame_idx}{ext}"
                torch.save(net.state_dict(), checkpoint_path)
                print(f"💾 Model checkpoint saved at frame {frame_idx}: {checkpoint_path}")

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
            
            # Log loss to TensorBoard and track for console output
            loss_value = loss_t.item()
            recent_losses.append(loss_value)
            writer.add_scalar("loss", loss_value, frame_idx)
            writer.add_scalar("loss_smoothed", loss_value, frame_idx)
            
            # Print loss info every 100 training steps for detailed monitoring
            if frame_idx % 100 == 0:
                mean_loss = np.mean(recent_losses) if recent_losses else 0.0
                print(f"    Frame {frame_idx}: Training loss: {loss_value:.6f}, Mean loss (last 100): {mean_loss:.6f}")
            
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


def demo_beamrider():
    """Original demo function for BeamRider environment"""
    # Register ALE environments with Gymnasium
    gym.register_envs(ale_py)
    
    # Create BeamRider environment
    env_id = "ALE/BeamRider-v5"
    print(f"Creating environment: {env_id}")
    
    try:
        env = gym.make(env_id)
        print(f"✓ Successfully created {env_id}")
        
        # Print environment information
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        try:
            action_meanings = env.unwrapped.get_action_meanings()
            print(f"Action meanings: {action_meanings}")
        except AttributeError:
            print("Action meanings not available")
        
        # Reset environment and get initial observation
        observation, info = env.reset()
        print(f"Initial observation shape: {observation.shape}")
        print(f"Initial info: {info}")
        
        # Take a few random actions
        print("\nTaking 5 random actions:")
        for step in range(5):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {step + 1}:")
            try:
                action_meanings = env.unwrapped.get_action_meanings()
                print(f"  Action: {action} ({action_meanings[action]})")
            except (AttributeError, IndexError):
                print(f"  Action: {action}")
            print(f"  Reward: {reward}")
            print(f"  Terminated: {terminated}")
            print(f"  Truncated: {truncated}")
            
            if terminated or truncated:
                print("  Episode ended, resetting...")
                observation, info = env.reset()
        
        env.close()
        print("\n✓ Environment closed successfully")
        
    except Exception as e:
        print(f"✗ Error creating environment: {e}")


def list_available_beamrider_envs():
    """List all available BeamRider-related environments"""
    print("Available BeamRider environments:")
    
    # Register ALE environments
    gym.register_envs(ale_py)
    
    beamrider_envs = []
    try:
        import gymnasium.envs.registration
        for env_id in gymnasium.envs.registration.registry.keys():
            if 'BeamRider' in env_id or 'Beam' in env_id:
                beamrider_envs.append(env_id)
    except Exception:
        # Fallback list of known BeamRider environments
        beamrider_envs = [
            "ALE/BeamRider-v5",
            "BeamRider-v0",
            "BeamRider-v4", 
            "BeamRiderNoFrameskip-v0",
            "BeamRiderNoFrameskip-v4"
        ]
    
    if beamrider_envs:
        for env_id in sorted(beamrider_envs):
            print(f"  - {env_id}")
    else:
        print("  No BeamRider environments found")
    
    return beamrider_envs


# =============================================================================
# Main Function and CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train DQN agent on BeamRider game')
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
    parser.add_argument('--demo-only', action='store_true',
                        help='Only run BeamRider demo, do not train')
    parser.add_argument('--save-path', type=str, default='beamrider-best.dat',
                        help='Path to save the best model (default: beamrider-best.dat)')
    parser.add_argument('--target-reward', type=float, default=1000.0,
                        help='Target mean reward to consider solved (default: 1000.0)')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to pre-trained model to load and continue training from')
    parser.add_argument('--continue-training', action='store_true',
                        help='Continue training from loaded model (adjusts epsilon and other parameters)')
    parser.add_argument('--checkpoint-interval', type=int, default=10000,
                        help='Interval (in frames) for saving model checkpoints (default: 10000)')
    parser.add_argument('--max-episode-steps', type=int, default=None,
                        help='Maximum steps per episode (default: None for environment default, -1 for unlimited)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 BeamRider DQN Training")
    print("=" * 60)
    
    if args.demo_only:
        print("Running BeamRider demo...")
        list_available_beamrider_envs()
        print()
        demo_beamrider()
        return 0
    
    # Test setup first
    if not test_setup():
        print("❌ Setup test failed!")
        return 1
    
    if args.test_only:
        print("✅ Test completed successfully. Exiting (--test-only specified).")
        return 0
    
    print("\n" + "=" * 60)
    print("🎯 Starting BeamRider Training")
    print("=" * 60)
    
    # Start training
    train_beamrider_dqn(
        max_frames=args.frames,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        mean_reward_bound=args.target_reward,
        save_path=args.save_path,
        load_model=args.load_model,
        continue_training=args.continue_training,
        checkpoint_interval=args.checkpoint_interval,
        max_episode_steps=args.max_episode_steps
    )
    
    print("✅ BeamRider training completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
