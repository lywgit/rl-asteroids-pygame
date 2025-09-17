"""
Play script for DQN-trained models on Beamrider or Asteroids games.
- Loads a trained model and plays the game with human-viewable rendering
- Or see evaluation results faster without human-viewable rendering
- Recording of gameplay videos.
"""

import argparse
import os
from datetime import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import ale_py
from gymnasium.wrappers import RecordVideo

# Import shared components
from shared.models import AtariDQN, AtariDuelingDQN
from shared.environments import make_atari_env, make_py_asteroids_env
from shared.utils import get_device


def load_model(model_path: str, env, device: str):
    """Load a trained DQN model"""
    try:
        model = AtariDuelingDQN(env.observation_space.shape, env.action_space.n).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        model = AtariDQN(env.observation_space.shape, env.action_space.n).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def play_game(env, model, device: str, num_episodes: int = 5, delay: float = 0.0):
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
    parser.add_argument('game', choices=['py-asteroids', 'beamrider', 'asteroids' ], 
                       help='Game to play (py-asteroids, beamrider, asteroids)')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to the trained model (.pth file)')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of episodes to play (default: 1)')
    parser.add_argument('--delay', type=float, default=0.0,
                       help='Delay between steps in seconds (default: 0.0)')
    parser.add_argument('--no-render', action='store_true',
                       help='Run without rendering (for evaluation)')
    parser.add_argument('--record-video', action='store_true',
                       help='Record video of the gameplay (will force render_mode rgb_array)')

    args = parser.parse_args()
    
    # Get device
    device = get_device()
    print(f"🔧 Using device: {device}")
    
    # Create environment
    render_mode = "rgb_array" if args.no_render or args.record_video else "human"

    if args.game == 'py-asteroids':
        env = make_py_asteroids_env(render_mode=render_mode, action_mode="combination")
        print("🚀 Created Py-Asteroids environment. Should use correct action model: single or combination ")
    elif args.game == 'beamrider':
        env = make_atari_env("ALE/BeamRider-v5", render_mode=render_mode, grayscale_obs=True, max_episode_steps=100000)
        print("🛸 Created BeamRider environment")
    elif args.game == 'asteroids':
        env = make_atari_env("ALE/Asteroids-v5", render_mode=render_mode, grayscale_obs=True, max_episode_steps=100000)
        print("🚀 Created Asteroids environment")
    else:
        raise ValueError(f"Unknown game: {args.game}")

    if args.record_video:
        name_prefix = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.game}"
        env = RecordVideo(env, video_folder="videos/", episode_trigger=lambda x: True, name_prefix=name_prefix)
        os.makedirs("videos/", exist_ok=True)

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
