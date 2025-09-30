"""
Exploration metrics for tracking exploration behavior in DQN agents.
Particularly useful for noisy networks where epsilon is not meaningful.
"""
import numpy as np
import torch
from collections import deque
from typing import Dict, Optional
import math


class ExplorationMetrics:
    """
    Track exploration-related metrics for DQN agents.
    
    Metrics include:
    - Action entropy and diversity
    - Q-value statistics
    - Recent action distribution
    """
    
    def __init__(self, n_actions: int, window_size: int = 1000):
        self.n_actions = n_actions
        self.window_size = window_size
        
        # Track recent actions for entropy/diversity calculation
        self.recent_actions = deque(maxlen=window_size)
        
        # Track Q-value statistics
        self.recent_q_values = deque(maxlen=window_size)
        self.recent_max_q_values = deque(maxlen=window_size)
        
        # Action counts for current window
        self.action_counts = np.zeros(n_actions)
        
    def add_step(self, action: int, q_values: Optional[torch.Tensor] = None):
        """Add a step's data for metric calculation."""
        # Remove oldest action's contribution if at capacity
        if len(self.recent_actions) == self.window_size:
            old_action = self.recent_actions[0]
            self.action_counts[old_action] -= 1
        
        # Add new action
        self.recent_actions.append(action)
        self.action_counts[action] += 1
        
        # Add Q-value statistics if provided
        if q_values is not None:
            q_values_np = q_values.detach().cpu().numpy().flatten()
            self.recent_q_values.append(q_values_np)
            self.recent_max_q_values.append(np.max(q_values_np))
    
    def get_action_entropy(self) -> float:
        """
        Calculate action entropy over the recent window.
        Higher entropy = more uniform exploration.
        """
        if len(self.recent_actions) == 0:
            return 0.0
        
        # Calculate action probabilities
        total_actions = len(self.recent_actions)
        probabilities = self.action_counts / total_actions
        
        # Filter out zero probabilities
        probabilities = probabilities[probabilities > 0]
        
        if len(probabilities) <= 1:
            return 0.0
        
        # Calculate entropy: H = -sum(p * log(p))
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def get_action_diversity(self) -> float:
        """
        Calculate action diversity (normalized entropy).
        Returns value between 0 and 1, where 1 = perfectly uniform distribution.
        """
        entropy = self.get_action_entropy()
        max_entropy = math.log2(self.n_actions)  # Maximum possible entropy
        
        if max_entropy == 0:
            return 1.0
        
        return entropy / max_entropy
    
    def get_action_distribution(self) -> Dict[int, float]:
        """Get the current action distribution as percentages."""
        if len(self.recent_actions) == 0:
            return {i: 0.0 for i in range(self.n_actions)}
        
        total_actions = len(self.recent_actions)
        return {
            action: (count / total_actions * 100)
            for action, count in enumerate(self.action_counts)
        }
    
    def get_q_value_stats(self) -> Dict[str, float]:
        """Get Q-value statistics for exploration assessment."""
        if not self.recent_q_values:
            return {
                'q_mean': 0.0,
                'q_std': 0.0,
                'q_max_mean': 0.0,
                'q_max_std': 0.0,
                'q_variance': 0.0
            }
        
        # Flatten all Q-values
        all_q_values = np.concatenate(list(self.recent_q_values))
        max_q_values = np.array(list(self.recent_max_q_values))
        
        return {
            'q_mean': float(np.mean(all_q_values)),
            'q_std': float(np.std(all_q_values)),
            'q_max_mean': float(np.mean(max_q_values)),
            'q_max_std': float(np.std(max_q_values)),
            'q_variance': float(np.var(all_q_values))
        }
    
    def get_exploration_metrics(self) -> Dict[str, float]:
        """Get all exploration metrics in one call."""
        metrics = {
            'action_entropy': self.get_action_entropy(),
            'action_diversity': self.get_action_diversity(),
            'recent_actions_count': len(self.recent_actions)
        }
        
        # Add Q-value statistics
        metrics.update(self.get_q_value_stats())
        
        # Add action distribution (top 3 most common actions)
        action_dist = self.get_action_distribution()
        sorted_actions = sorted(action_dist.items(), key=lambda x: x[1], reverse=True)
        
        for i, (action, percentage) in enumerate(sorted_actions[:3]):
            metrics[f'top_{i+1}_action'] = action
            metrics[f'top_{i+1}_action_pct'] = percentage
        
        return metrics
    
    def reset(self):
        """Reset all metrics."""
        self.recent_actions.clear()
        self.recent_q_values.clear()
        self.recent_max_q_values.clear()
        self.action_counts.fill(0)