"""Training utilities and helper functions."""

import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

def setup_training_directories(run_name=None):
    """Setup directories for a training run."""
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_dir = Path(f"logs/training/{run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    dirs = {
        'run_dir': run_dir,
        'checkpoints': Path("saved_models/checkpoints"),
        'best_models': Path("saved_models/best_models"),
        'logs': run_dir / "logs",
        'plots': run_dir / "plots",
        'videos': run_dir / "videos"
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def save_training_config(config, run_dir):
    """Save training configuration."""
    config_path = run_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def plot_training_progress(scores, rewards, losses=None, save_path=None):
    """Plot training progress."""
    fig, axes = plt.subplots(2, 2 if losses is not None else 1, figsize=(12, 8))
    if losses is None:
        axes = [axes[0], axes[1], None, None]
    else:
        axes = axes.flatten()
    
    # Plot scores
    axes[0].plot(scores)
    axes[0].set_title('Training Scores')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Score')
    axes[0].grid(True)
    
    # Plot rewards
    axes[1].plot(rewards)
    axes[1].set_title('Training Rewards')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Total Reward')
    axes[1].grid(True)
    
    # Plot losses if available
    if losses is not None and axes[2] is not None:
        axes[2].plot(losses)
        axes[2].set_title('Training Loss')
        axes[2].set_xlabel('Training Step')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True)
        
        # Plot moving average of scores
        if len(scores) > 10:
            window_size = min(100, len(scores) // 10)
            moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
            axes[3].plot(range(window_size-1, len(scores)), moving_avg)
            axes[3].set_title(f'Moving Average Score (window={window_size})')
            axes[3].set_xlabel('Episode')
            axes[3].set_ylabel('Average Score')
            axes[3].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

class TrainingLogger:
    """Logger for training metrics."""
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'episodes': [],
            'scores': [],
            'rewards': [],
            'steps': [],
            'losses': [],
            'epsilon': []  # For epsilon-greedy strategies
        }
        
        self.log_file = self.log_dir / "training_log.json"
    
    def log_episode(self, episode, score, reward, steps, loss=None, epsilon=None):
        """Log metrics for an episode."""
        self.metrics['episodes'].append(episode)
        self.metrics['scores'].append(score)
        self.metrics['rewards'].append(reward)
        self.metrics['steps'].append(steps)
        
        if loss is not None:
            self.metrics['losses'].append(loss)
        if epsilon is not None:
            self.metrics['epsilon'].append(epsilon)
        
        # Save every 10 episodes
        if episode % 10 == 0:
            self.save_log()
    
    def save_log(self):
        """Save metrics to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def plot_progress(self, save_path=None):
        """Plot training progress."""
        if save_path is None:
            save_path = self.log_dir / "training_progress.png"
        
        losses = self.metrics['losses'] if self.metrics['losses'] else None
        plot_training_progress(
            self.metrics['scores'], 
            self.metrics['rewards'], 
            losses, 
            save_path
        )

def evaluate_model(model, env, num_episodes=10, max_steps=1000, render=False):
    """
    Evaluate a model's performance.
    
    Returns:
        dict: Evaluation results with statistics
    """
    scores = []
    rewards = []
    steps_taken = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        
        while not done and step_count < max_steps:
            action = model(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if render:
                env.render()
        
        scores.append(info['score'])
        rewards.append(total_reward)
        steps_taken.append(step_count)
    
    results = {
        'num_episodes': num_episodes,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': np.max(scores),
        'min_score': np.min(scores),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'max_reward': np.max(rewards),
        'min_reward': np.min(rewards),
        'mean_steps': np.mean(steps_taken),
        'scores': scores,
        'rewards': rewards,
        'steps': steps_taken
    }
    
    return results
