"""
Play script for DQN-trained models on Beamrider or Asteroids games.
Loads a trained model and plays the game with human-viewable rendering.
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import ale_py
from gymnasium.wrappers import (
    AtariPreprocessing, 
    FrameStackObservation, 
    MaxAndSkipObservation, 
    ResizeObservation,
    GrayscaleObservation
)
from asteroids.gym_env import AsteroidsEnv


def get_device():
    """Get the best available device for inference"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


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
    
def make_atari_env(env_id: str, render_mode: str = "human", max_episode_steps: int = 10000, 
                   screen_size=(84, 84), frame_stack: int = 4, scale_obs: bool = True, **kwargs):
    """Create Atari environment with standard preprocessing"""
    env = gym.make(env_id, render_mode=render_mode, max_episode_steps=max_episode_steps, frameskip=1) 
    env = AtariPreprocessing(env, screen_size=screen_size, scale_obs=scale_obs,**kwargs)
    if frame_stack > 1:
        env = FrameStackObservation(env, frame_stack)
    return env


def make_asteroids_env(render_mode: str = "human", screen_size=(128, 128), 
                      scale_obs:bool = True, grayscale_obs: bool = True, frame_stack: int = 4):
    """Create Asteroids environment with preprocessing"""
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


class AtariDQN(nn.Module):
    """Q Learning network mapping pixel observations to action values"""
    def __init__(self, input_shape, n_action):
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
        return output.view(1, -1).size(1)

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size()[0], -1))


def load_model(model_path: str, env, device: str):
    """Load a trained DQN model"""
    model = AtariDQN(env.observation_space.shape, env.action_space.n).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def play_game(env, model, device: str, num_episodes: int = 5, delay: float = 0.02):
    """Play the game using the trained model"""
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        step_count = 0
        
        print(f"\n🎮 Starting Episode {episode + 1}")
        
        while True:
            # Convert observation to tensor and add batch dimension
            obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            
            # Get action from model (greedy policy - no exploration)
            with torch.no_grad():
                q_values = model(obs_tensor)
                action = q_values.argmax(dim=1).item()
            
            # Take action
            obs, reward, truncated, terminated, _ = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Add small delay to make gameplay visible
            if delay > 0:
                time.sleep(delay)
            
            # Check if episode is done
            if truncated or terminated:
                break
        
        episode_rewards.append(episode_reward)
        print(f"📊 Episode {episode + 1} finished: {episode_reward:.2f} points in {step_count} steps")
    
    return episode_rewards


def main():
    parser = argparse.ArgumentParser(description='Play DQN-trained agent on Beamrider or Asteroids')
    parser.add_argument('game', choices=['beamrider', 'asteroids'], 
                       help='Game to play (beamrider or asteroids)')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to the trained model (.pth file)')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of episodes to play (default: 1)')
    parser.add_argument('--delay', type=float, default=0.0,
                       help='Delay between steps in seconds (default: 0.0)')
    parser.add_argument('--no-render', action='store_true',
                       help='Run without rendering (for evaluation)')
    
    args = parser.parse_args()
    
    # Get device
    device = get_device()
    print(f"🔧 Using device: {device}")
    
    # Create environment
    render_mode = "rgb_array" if args.no_render else "human"
    
    if args.game == 'asteroids':
        env = make_asteroids_env(render_mode=render_mode)
        print("🚀 Created Asteroids environment")
    elif args.game == 'beamrider':
        env = make_atari_env("ALE/BeamRider-v5", render_mode=render_mode, grayscale_obs=True)
        print("🛸 Created BeamRider environment")
    else:
        raise ValueError(f"Unknown game: {args.game}")
    
    print(f"📐 Environment info:")
    print(f"  - Observation space: {env.observation_space.shape}")
    print(f"  - Action space: {env.action_space.n}") # type: ignore
    
    # Load model
    try:
        model = load_model(args.model, env, device)
        print(f"✅ Loaded model from: {args.model}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Play the game
    try:
        print(f"\n🎯 Starting to play {args.episodes} episodes of {args.game}...")
        episode_rewards = play_game(env, model, device, args.episodes, args.delay)
        
        # Print statistics
        print(f"\n📈 Game Statistics:")
        print(f"  - Episodes played: {len(episode_rewards)}")
        print(f"  - Average reward: {np.mean(episode_rewards):.2f}")
        print(f"  - Best episode: {np.max(episode_rewards):.2f}")
        print(f"  - Worst episode: {np.min(episode_rewards):.2f}")
        print(f"  - Standard deviation: {np.std(episode_rewards):.2f}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Game interrupted by user")
    finally:
        env.close()
        print("🏁 Game session ended")


if __name__ == "__main__":
    main()
