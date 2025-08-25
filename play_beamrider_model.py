#!/usr/bin/env python3
"""
Test script for the trained BeamRider DQN model
"""

import sys
import os
import torch
import numpy as np
import time

# Add the project root to the path
sys.path.append(os.path.abspath('.'))

from beamrider_dqn_training import BeamRiderDQN, make_beamrider_env


def test_trained_beamrider_model(model_path="beamrider-best.dat", num_episodes=3, render=True, delay=0.05):
    """Test the trained BeamRider DQN model"""
    
    # Get device
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create environment with appropriate render mode
    render_mode = 'human' if render else 'rgb_array'
    env = make_beamrider_env(render_mode=render_mode)
    
    print(f"Environment: obs_space={env.observation_space}, action_space={env.action_space}")
    
    # Create and load the network
    action_space_size = getattr(env.action_space, 'n', 9)  # BeamRider has 9 actions
    net = BeamRiderDQN(env.observation_space.shape, action_space_size).to(device)
    
    if os.path.exists(model_path):
        print(f"Loading trained model from: {model_path}")
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.eval()
        print("✅ Model loaded successfully")
    else:
        print(f"❌ Model file not found: {model_path}")
        print("🔄 Using untrained network for demonstration")
    
    # Test the model
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        lives = info.get('lives', 3)
        game_started = False
        action_counts = {i: 0 for i in range(9)}  # Track action frequencies
        steps_since_life_lost = 0  # Track steps since life was lost
        
        print(f"\n--- Episode {episode + 1} ---")
        print(f"Starting with {lives} lives")
        
        while not done and steps < 2000:  # Max 2000 steps per episode (BeamRider can be longer)
            # Get current lives first
            current_lives = info.get('lives', lives)
            
            # Use the trained network with the processed state
            if len(state.shape) == 3:  # Add batch dimension if needed
                state_tensor = torch.tensor(state).unsqueeze(0).to(device)
            else:
                state_tensor = torch.tensor(state).to(device)
            
            with torch.no_grad():
                q_values = net(state_tensor)
                action = int(torch.argmax(q_values, dim=1).item())
            
            # Auto-start logic - BeamRider requires movement to start, not just fire
            
            # Force movement action for first few steps to start the game
            if steps < 3 and not game_started:
                # Cycle through movement actions to ensure game starts
                start_actions = [3, 4, 1]  # RIGHT, LEFT, FIRE
                action = start_actions[steps % len(start_actions)]
                if steps == 0:
                    print(f"  Auto-moving to start game...")
            
            # Check for life loss and force restart
            elif current_lives < lives:
                # Life was lost, need to restart with movement
                action = 3 if steps % 2 == 0 else 4  # Alternate between RIGHT and LEFT
                print(f"  Life lost! Auto-moving to start new life (lives: {current_lives})...")
                lives = current_lives  # Update tracked lives
                game_started = False  # Reset game started flag for new life
                steps_since_life_lost = 0  # Reset counter
            
            # Continue forcing movement for a few steps after life loss
            elif steps_since_life_lost < 3 and not game_started:
                action = 3 if steps_since_life_lost % 2 == 0 else 4  # Alternate movement
                steps_since_life_lost += 1
            
            # Track action frequencies
            action_counts[action] += 1
            
            # Step in the environment (if not already done in the life-start loop)
            if not done:
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += float(reward)
                
                # Check if game has started (receiving any reward or after a few steps means game is active)
                if not game_started and (float(reward) > 0 or steps > 5):
                    game_started = True
                    if float(reward) > 0:
                        print(f"  Game started! First reward received at step {steps}")
                    else:
                        print(f"  Game assumed started at step {steps}")
                
                if render:
                    env.render()
                    if delay > 0:
                        time.sleep(delay)  # Add delay to make it watchable
                
                steps += 1
                steps_since_life_lost += 1  # Increment life loss counter
                
                # Print progress when rewards are received or periodically
                if steps % 200 == 0 or float(reward) > 0:
                    score = total_reward  # In BeamRider, reward is typically the score
                    print(f"  Step {steps}: Score={score:.0f}, Lives={current_lives}, Last reward={reward}")
        
        episode_rewards.append(total_reward)
        
        # Show action distribution
        action_meanings = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'UPRIGHT', 'UPLEFT', 'RIGHTFIRE', 'LEFTFIRE']
        print(f"Episode {episode + 1} finished:")
        print(f"  Steps: {steps}")
        print(f"  Total score/reward: {total_reward:.0f}")
        print(f"  Final lives: {info.get('lives', 0)}")
        print(f"  Game started: {game_started}")
        print(f"  Action distribution:")
        for i, count in action_counts.items():
            if count > 0:
                percentage = (count / steps) * 100
                print(f"    {action_meanings[i]:10}: {count:4d} ({percentage:5.1f}%)")
        
        # Add a small pause between episodes if rendering
        if render and episode < num_episodes - 1:
            print("  Press Enter to continue to next episode...")
            input()
    
    env.close()
    
    # Summary
    print(f"\n--- Summary over {num_episodes} episodes ---")
    print(f"Mean score: {np.mean(episode_rewards):.2f}")
    print(f"Std score: {np.std(episode_rewards):.2f}")
    print(f"Min score: {np.min(episode_rewards):.2f}")
    print(f"Max score: {np.max(episode_rewards):.2f}")
    
    return episode_rewards


def compare_models(model_paths, num_episodes_per_model=3):
    """Compare multiple trained models"""
    print("🔬 Comparing Multiple BeamRider Models")
    print("=" * 60)
    
    results = {}
    
    for model_path in model_paths:
        print(f"\nTesting model: {model_path}")
        print("-" * 40)
        
        if os.path.exists(model_path):
            rewards = test_trained_beamrider_model(
                model_path=model_path,
                num_episodes=num_episodes_per_model,
                render=False  # No rendering for comparison
            )
            results[model_path] = rewards
        else:
            print(f"❌ Model not found: {model_path}")
            results[model_path] = []
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("📊 MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'Mean Score':<12} {'Max Score':<12} {'Episodes':<10}")
    print("-" * 60)
    
    for model_path, rewards in results.items():
        if rewards:
            mean_score = np.mean(rewards)
            max_score = np.max(rewards)
            num_eps = len(rewards)
            model_name = os.path.basename(model_path)
            print(f"{model_name:<30} {mean_score:<12.1f} {max_score:<12.1f} {num_eps:<10}")
        else:
            model_name = os.path.basename(model_path)
            print(f"{model_name:<30} {'N/A':<12} {'N/A':<12} {'0':<10}")
    
    return results


def watch_random_agent(episodes=1, render=True):
    """Watch a random agent play for comparison"""
    print("🎲 Watching Random Agent Play BeamRider")
    print("=" * 50)
    
    render_mode = 'human' if render else 'rgb_array'
    env = make_beamrider_env(render_mode=render_mode)
    
    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        action_counts = {i: 0 for i in range(9)}  # Track action frequencies
        
        print(f"\n--- Random Agent Episode {episode + 1} ---")
        
        while not done and steps < 1000:
            action = env.action_space.sample()  # Random action
            action_counts[action] += 1
            
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            
            if render:
                env.render()
                time.sleep(0.05)  # Slow down for viewing
            
            steps += 1
            
            if steps % 100 == 0:
                print(f"  Random agent - Step {steps}: Score={total_reward:.0f}")
        
        print(f"Random agent final score: {total_reward:.0f}")
        
        # Show action distribution for random agent
        action_meanings = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'UPRIGHT', 'UPLEFT', 'RIGHTFIRE', 'LEFTFIRE']
        print(f"  Action distribution:")
        for i, count in action_counts.items():
            if count > 0:
                percentage = (count / steps) * 100
                print(f"    {action_meanings[i]:10}: {count:4d} ({percentage:5.1f}%)")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained BeamRider DQN model')
    parser.add_argument('--model', type=str, default='beamrider-best.dat',
                        help='Path to trained model file')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of test episodes')
    parser.add_argument('--no-render', action='store_true',
                        help='Run without visual rendering (faster)')
    parser.add_argument('--delay', type=float, default=0.05,
                        help='Delay between frames in seconds (default: 0.05)')
    parser.add_argument('--compare', nargs='+', type=str,
                        help='Compare multiple models (provide multiple model paths)')
    parser.add_argument('--random', action='store_true',
                        help='Watch random agent play instead')
    
    args = parser.parse_args()
    
    if args.random:
        watch_random_agent(episodes=args.episodes, render=not args.no_render)
    elif args.compare:
        compare_models(args.compare, num_episodes_per_model=args.episodes)
    else:
        print("🚀 Testing Trained BeamRider DQN Model")
        print("=" * 50)
        
        rewards = test_trained_beamrider_model(
            model_path=args.model,
            num_episodes=args.episodes,
            render=not args.no_render,
            delay=args.delay
        )
        
        print("\n✅ Testing completed!")
        
        # Suggest next steps
        print("\n💡 Try these commands:")
        print(f"  # Watch AI play with slower speed:")
        print(f"  uv run play_beamrider_model.py --model {args.model} --delay 0.1")
        print(f"  # Compare different checkpoints:")
        print(f"  uv run play_beamrider_model.py --compare beamrider-best.dat beamrider-best_frame_5000.dat --episodes 5")
        print(f"  # Watch random agent:")
        print(f"  uv run play_beamrider_model.py --random --episodes 1")
