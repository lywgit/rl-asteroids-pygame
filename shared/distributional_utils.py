"""
Utilities for distributional Q-learning (C51/Categorical DQN).

This module provides functions for categorical projection and distributional loss
calculation used in distributional reinforcement learning algorithms.
"""

import torch
import torch.nn.functional as F
import numpy as np


def categorical_projection(next_dist, rewards, dones, gamma, support, n_atoms, v_min, v_max):
    """
    Project distributional Bellman update onto categorical support.
    
    This implements the categorical projection algorithm from the C51 paper.
    For each atom z_j in the support, we compute T_z_j = r + γ * z_j and project
    it back onto the nearest atoms in the support.
    
    Args:
        next_dist: [batch_size, n_atoms] - Next state action-value distribution
        rewards: [batch_size] - Immediate rewards  
        dones: [batch_size] - Terminal state indicators
        gamma: float - Discount factor (or gamma^n for n-step)
        support: [n_atoms] - Support of the categorical distribution
        n_atoms: int - Number of atoms in categorical distribution
        v_min: float - Minimum value of support
        v_max: float - Maximum value of support
        
    Returns:
        target_dist: [batch_size, n_atoms] - Target distribution after projection
    """
    batch_size = next_dist.size(0)
    device = next_dist.device
    
    # Expand tensors for broadcasting
    rewards = rewards.unsqueeze(1)  # [batch_size, 1]
    dones = dones.unsqueeze(1)      # [batch_size, 1]
    support = support.unsqueeze(0)  # [1, n_atoms]
    
    # Compute Bellman operator: T_z = r + γ * (1 - done) * z
    # Shape: [batch_size, n_atoms]
    tz = rewards + gamma * (1 - dones.float()) * support
    
    # Clamp to support bounds [v_min, v_max]
    tz = torch.clamp(tz, v_min, v_max)
    
    # Compute projection indices and weights
    # b = (tz - v_min) / delta_z, where delta_z = (v_max - v_min) / (n_atoms - 1)
    delta_z = (v_max - v_min) / (n_atoms - 1)
    b = (tz - v_min) / delta_z
    
    # Lower and upper indices for projection
    l = b.floor().long()    # [batch_size, n_atoms]
    u = b.ceil().long()     # [batch_size, n_atoms]
    
    # Handle edge case where l == u (exact match)
    # When b is an integer, l == u, so we need to handle this case
    l[(u > 0) * (l == u)] -= 1
    u[(l < (n_atoms - 1)) * (l == u)] += 1
    
    # Initialize target distribution
    target_dist = torch.zeros(batch_size, n_atoms, device=device)
    
    # Distribute probability mass using linear interpolation
    # For each batch and each atom j in next_dist:
    # - Find where T_z_j projects to (between atoms l and u)  
    # - Distribute next_dist[batch, j] proportionally between l and u
    
    # Flatten indices for scatter_add
    offset = torch.linspace(0, (batch_size - 1) * n_atoms, batch_size, device=device).long()
    offset = offset.unsqueeze(1).expand(batch_size, n_atoms)
    
    # Project probabilities to lower indices
    # Weight = (u - b), where u is upper index and b is floating projection
    l_indices = (l + offset).view(-1)
    u_indices = (u + offset).view(-1)
    
    target_dist_flat = target_dist.view(-1)
    next_dist_flat = next_dist.view(-1)
    
    # Distribute probability mass
    # P(l) += p * (u - b) and P(u) += p * (b - l)
    # Use non-in-place operations to avoid gradient issues
    lower_weights = next_dist_flat * (u.float() - b).view(-1)
    upper_weights = next_dist_flat * (b - l.float()).view(-1)
    
    target_dist_flat = target_dist_flat.scatter_add(0, l_indices, lower_weights)
    target_dist_flat = target_dist_flat.scatter_add(0, u_indices, upper_weights)
    
    target_dist = target_dist_flat.view(batch_size, n_atoms)
    
    return target_dist


def distributional_loss(pred_dist, target_dist, reduction='mean'):
    """
    Compute cross-entropy loss between predicted and target distributions.
    
    Args:
        pred_dist: [batch_size, n_atoms] - Predicted probability distribution
        target_dist: [batch_size, n_atoms] - Target probability distribution  
        reduction: str - Reduction method ('mean', 'sum', 'none')
        
    Returns:
        loss: scalar or [batch_size] - Cross-entropy loss
    """
    # Clamp to avoid log(0)
    pred_dist = torch.clamp(pred_dist, min=1e-8, max=1.0)
    
    # Cross-entropy: -sum(target * log(pred))
    cross_entropy = -torch.sum(target_dist * torch.log(pred_dist), dim=1)
    
    if reduction == 'mean':
        return cross_entropy.mean()
    elif reduction == 'sum':
        return cross_entropy.sum()
    elif reduction == 'none':
        return cross_entropy
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def kl_divergence(pred_dist, target_dist, reduction='mean'):
    """
    Compute KL divergence between predicted and target distributions.
    Can be used as an alternative priority metric for prioritized replay.
    
    Args:
        pred_dist: [batch_size, n_atoms] - Predicted probability distribution
        target_dist: [batch_size, n_atoms] - Target probability distribution
        reduction: str - Reduction method ('mean', 'sum', 'none')
        
    Returns:
        kl_div: scalar or [batch_size] - KL divergence
    """
    # Clamp to avoid log(0) and division by 0
    pred_dist = torch.clamp(pred_dist, min=1e-8, max=1.0)
    target_dist = torch.clamp(target_dist, min=1e-8, max=1.0)
    
    # KL(target || pred) = sum(target * log(target / pred))
    kl_div = torch.sum(target_dist * torch.log(target_dist / pred_dist), dim=1)
    
    if reduction == 'mean':
        return kl_div.mean()
    elif reduction == 'sum':
        return kl_div.sum()
    elif reduction == 'none':
        return kl_div
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def estimate_value_range(buffer, percentile=99.5):
    """
    Estimate appropriate value range [v_min, v_max] from experience buffer.
    
    This function samples from the buffer to estimate the range of Q-values
    that will be encountered during training, which helps set appropriate
    bounds for the categorical distribution support.
    
    Args:
        buffer: Experience buffer containing (s, a, r, s', done) tuples
        percentile: float - Percentile for outlier removal
        
    Returns:
        v_min: float - Estimated minimum Q-value
        v_max: float - Estimated maximum Q-value
    """
    if len(buffer) < 1000:
        # Not enough data, use default range
        return -10.0, 10.0
    
    # Sample rewards from buffer to estimate Q-value range
    sample_size = min(len(buffer), 10000)
    
    # Sample random experiences
    if hasattr(buffer, 'sample'):
        # Standard buffer interface
        try:
            _, _, rewards, _, _ = buffer.sample(sample_size)
            if hasattr(rewards, 'cpu'):
                rewards = rewards.cpu().numpy()
        except:
            # Fallback to accessing buffer data directly
            rewards = []
            indices = np.random.choice(len(buffer), size=sample_size, replace=False)
            for idx in indices:
                exp = buffer.buffer[idx % len(buffer.buffer)]
                rewards.append(exp.reward)
            rewards = np.array(rewards)
    else:
        # Direct buffer access
        rewards = []
        indices = np.random.choice(len(buffer), size=sample_size, replace=False)
        for idx in indices:
            exp = buffer.buffer[idx % len(buffer.buffer)]
            rewards.append(exp.reward)
        rewards = np.array(rewards)
    
    # Estimate Q-value range using reward statistics and typical episode lengths
    # Q-values are roughly: immediate_reward + discounted_future_rewards
    # For most Atari games, episodes are 100-1000 steps, so we can estimate:
    
    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards)
    reward_min = np.percentile(rewards, 100 - percentile)
    reward_max = np.percentile(rewards, percentile)
    
    # Estimate Q-value bounds (heuristic)
    # Assume typical episode length and discount factor
    gamma = 0.99
    typical_episode_length = 200
    geometric_sum = (1 - gamma**typical_episode_length) / (1 - gamma)
    
    # Conservative bounds
    v_min = reward_min * geometric_sum - 3 * abs(reward_std) * geometric_sum
    v_max = reward_max * geometric_sum + 3 * abs(reward_std) * geometric_sum
    
    # Add safety margins and reasonable bounds
    v_min = max(v_min, -100.0)  # Not too negative
    v_max = min(v_max, 1000.0)  # Not too positive
    
    # Ensure minimum range
    if v_max - v_min < 10:
        center = (v_max + v_min) / 2
        v_min = center - 5
        v_max = center + 5
    
    return float(v_min), float(v_max)


# Game-specific value ranges (manually tuned based on game characteristics)
GAME_VALUE_RANGES = {
    'py-asteroids': {'v_min': -10.0, 'v_max': 50.0},    # Custom game, moderate scores
    'pong': {'v_min': -21.0, 'v_max': 21.0},            # Game ends at ±21
    'beamrider': {'v_min': -10.0, 'v_max': 100.0},      # Variable scoring
    'enduro': {'v_min': -10.0, 'v_max': 50.0},          # Moderate scores
    'spaceinvaders': {'v_min': -10.0, 'v_max': 100.0},  # Variable scoring
    'centipede': {'v_min': -10.0, 'v_max': 100.0},      # Variable scoring
    'asteroids': {'v_min': -10.0, 'v_max': 100.0},      # Atari Asteroids
}


def get_value_range_for_game(game_name, buffer=None, default_v_min=-10.0, default_v_max=10.0):
    """
    Get appropriate value range for a specific game.
    
    Args:
        game_name: str - Name of the game
        buffer: Optional experience buffer for dynamic estimation
        default_v_min: float - Default minimum value
        default_v_max: float - Default maximum value
        
    Returns:
        v_min: float - Minimum value for support
        v_max: float - Maximum value for support
    """
    game_name = game_name.lower()
    
    # Try game-specific ranges first
    if game_name in GAME_VALUE_RANGES:
        range_config = GAME_VALUE_RANGES[game_name]
        return range_config['v_min'], range_config['v_max']
    
    # Try dynamic estimation from buffer
    if buffer is not None:
        try:
            return estimate_value_range(buffer)
        except Exception as e:
            print(f"Warning: Failed to estimate value range from buffer: {e}")
    
    # Fallback to defaults
    return default_v_min, default_v_max