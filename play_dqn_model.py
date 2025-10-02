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
from shared.environments import make_atari_env, make_py_asteroids_env, atari_name_id_map, py_asteroids_name_id_map
from shared.utils import get_device
from shared.model_factory import load_model_checkpoint


def load_model(model_path: str, env, device: str):
    """Load a trained DQN model with automatic architecture detection"""
    try:
        model, model_config, training_info = load_model_checkpoint(model_path, env, device)
        model.eval()
        
        # Display model information
        print(f"🏗️  Model Architecture: {model_config.get_architecture_name()}")
        if model_config.game:
            print(f"🎮 Trained on: {model_config.game}")
        if training_info.get('best_eval_reward'):
            print(f"🏆 Best evaluation reward: {training_info['best_eval_reward']:.2f}")
        if training_info.get('frame_idx'):
            print(f"📊 Training frames: {training_info['frame_idx']:,}")
            
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def play_game(game, env, model, device: str, num_episodes: int = 5, delay: float = 0.0):
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
                # Use get_q_values() method to handle both standard and distributional models
                if hasattr(model, 'get_q_values'):
                    q_values = model.get_q_values(obs_tensor)
                else:
                    # Fallback for models without get_q_values method
                    q_values = model(obs_tensor)
                action = q_values.argmax(dim=1).item()
            if game == 'asteroids': # disable DOWN and DOWNFIRE so ship doesn't disappear
                if action in (5, 11):
                    action = 0 
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
    parser = argparse.ArgumentParser(description='Play DQN-trained agent on BeamRider or Asteroids')
    parser.add_argument('game', type=str,
                        help='Game to play (py-asteroids, or atari games: beamrider, asteroids, etc)')
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
    parser.add_argument('--clip-reward', action='store_true',
                        help='Clip rewards to [-1, 1] (default: False for evaluation)')

    args = parser.parse_args()
    
    # Get device
    device = get_device()
    print(f"🔧 Using device: {device}")
    
    # Create environment
    render_mode = "rgb_array" if args.no_render or args.record_video else "human"
    game: str = args.game.lower()
    clip_reward: bool = args.clip_reward  # Use user-specified reward clipping
    # Use frame skip of 3 for some games for flickering issue, 4 for others
    frame_skip = 3 if game in {'asteroids', 'spaceinvaders'} else 4  
    if game.startswith('py-asteroids'):
        config_version = py_asteroids_name_id_map.get(game, game) # ex: py-asteroids or py-asteroids-v1
        env = make_py_asteroids_env(action_mode="ale", render_mode=render_mode, config_version=config_version, clip_reward=clip_reward) 
    else: 
        env_id = atari_name_id_map.get(game, game)
        try:
            env = make_atari_env(env_id, render_mode=render_mode, clip_reward=clip_reward, frame_skip=frame_skip) # No reward clipping for evaluation
            print(f"🚀 Created Atari environment (without reward clipping): {env_id}")
        except Exception as e:
            raise ValueError(f"Unsupported game: {game}. Error: {e}")

    if args.record_video:
        if game in {'asteroids', 'spaceinvaders'}:
            print('Enabling smooth rendering')
            from shared.wrappers import MaxRender
            env = MaxRender(env) # Smooth rendering for Atari games (particularly ALE Asteroids)
        name_prefix = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.game}"
        # Assume default fps is 60, but with frame skip of 4 it's effectively 15 fps, make video at twice that to save viewing time
        try:
            env.metadata["render_fps"] = 2 * (env.metadata["render_fps"] // frame_skip)
        except:
            print("⚠️  No preset render_fps found, defaulting to 30 fps")
            env.metadata["render_fps"] = 2 * 15
        

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
        episode_rewards = play_game(args.game, env, model, device, args.episodes, args.delay)
        
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
