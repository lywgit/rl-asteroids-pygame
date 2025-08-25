#!/usr/bin/env python3
"""
Test script for the trained DQN model
"""

import sys
import os
import torch
import numpy as np

# Add the project root to the path
sys.path.append(os.path.abspath('.'))

from asteroids_dqn_training import AsteroidsDQN, make_asteroids_env


def test_trained_model(model_path="asteroids-best.dat", num_episodes=3, render=True):
    """Test the trained DQN model"""
    
    # Get device
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create environment with appropriate render mode
    render_mode = 'human' if render else 'rgb_array'
    env = make_asteroids_env(render_mode=render_mode)
    
    print(f"Environment: obs_space={env.observation_space}, action_space={env.action_space}")
    
    # Create and load the network
    action_space_size = getattr(env.action_space, 'n', 32)  # Default to 32 if not available
    net = AsteroidsDQN(env.observation_space.shape, action_space_size).to(device)
    
    if os.path.exists(model_path):
        print(f"Loading trained model from: {model_path}")
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.eval()
        print("✅ Model loaded successfully")
    else:
        print(f"❌ Model file not found: {model_path}")
        print("🔄 Using untrained network")
    
    # Test the model
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0.0  # Initialize as float
        steps = 0
        done = False
        
        print(f"\n--- Episode {episode + 1} ---")
        
        while not done and steps < 1000:  # Max 1000 steps per episode
            # Use the trained network with the processed state
            if len(state.shape) == 3:  # Add batch dimension if needed
                state_tensor = torch.tensor(state).unsqueeze(0).to(device)
            else:
                state_tensor = torch.tensor(state).to(device)
            
            with torch.no_grad():
                q_values = net(state_tensor)
                action = int(torch.argmax(q_values, dim=1).item())
            
            # Step in the environment
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)  # Explicitly convert to float
            
            if render:
                env.render()
            
            steps += 1
            
            # Print progress
            if steps % 100 == 0:
                print(f"  Step {steps}: Score={info.get('score', 0)}, Reward={total_reward:.1f}")
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1} finished:")
        print(f"  Steps: {steps}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Score: {info.get('score', 'N/A')}")
        print(f"  Survival reward: {info.get('survival_reward', 0):.1f}")
        print(f"  Level: {info.get('level', 1)}")
    
    env.close()
    
    # Summary
    print(f"\n--- Summary over {num_episodes} episodes ---")
    print(f"Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"Std reward: {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    
    return episode_rewards


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained DQN model')
    parser.add_argument('--model', type=str, default='asteroids-best.dat',
                        help='Path to trained model file')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of test episodes')
    parser.add_argument('--no-render', action='store_true',
                        help='Run without visual rendering (faster)')
    
    args = parser.parse_args()
    
    print("🚀 Testing Trained Asteroids DQN Model")
    print("=" * 50)
    
    rewards = test_trained_model(
        model_path=args.model,
        num_episodes=args.episodes,
        render=not args.no_render
    )
    
    print("\n✅ Testing completed!")
