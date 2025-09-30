"""
Noisy Network components for exploration in DQN.

Implements factorized Gaussian noise as described in:
"Noisy Networks for Exploration" (Fortunato et al., 2017)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class NoisyLinear(nn.Module):
    """
    Noisy linear layer with factorized Gaussian noise.
    
    This layer adds learnable noise to both weights and biases, providing
    state-dependent exploration that's more sophisticated than epsilon-greedy.
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters for weights
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        
        # Learnable parameters for biases
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Register buffers for efficient noise storage
        self.register_buffer('epsilon_input', torch.zeros(in_features))
        self.register_buffer('epsilon_output', torch.zeros(out_features))
        
        # Initialize parameters and generate initial noise
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize learnable parameters."""
        # Initialize mu (mean) parameters
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        
        # Initialize sigma (noise scale) parameters
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """Generate new factorized Gaussian noise."""
        # Generate noise and update buffers safely
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Use detached assignment to avoid gradient tracking issues
        self.epsilon_input = epsilon_in.detach()
        self.epsilon_output = epsilon_out.detach()
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise using the sign(x) * sqrt(|x|) function."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights and biases."""
        if self.training:
            # Generate fresh noise for each forward pass during training
            self.reset_noise()
            # Use factorized noise: W = μ + σ * (ε_out ⊗ ε_in)
            weight = self.weight_mu + self.weight_sigma * self.epsilon_output.outer(self.epsilon_input)
            bias = self.bias_mu + self.bias_sigma * self.epsilon_output
        else:
            # Use only mean parameters during evaluation
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def get_noise_stats(self) -> dict:
        """Get statistics about the current noise levels for logging."""
        if not self.training:
            return {
                'weight_sigma_mean': 0.0,
                'weight_sigma_std': 0.0, 
                'bias_sigma_mean': 0.0,
                'bias_sigma_std': 0.0,
                'noise_scale': 0.0,
                'signal_to_noise_ratio': float('inf')
            }
        
        # Current noise magnitudes
        weight_noise = self.weight_sigma * self.epsilon_output.outer(self.epsilon_input)
        bias_noise = self.bias_sigma * self.epsilon_output
        
        # Signal-to-noise ratios (higher = less exploration)
        weight_snr = (self.weight_mu.abs().mean() / (self.weight_sigma.abs().mean() + 1e-8)).item()
        bias_snr = (self.bias_mu.abs().mean() / (self.bias_sigma.abs().mean() + 1e-8)).item()
        
        return {
            'weight_sigma_mean': self.weight_sigma.abs().mean().item(),
            'weight_sigma_std': self.weight_sigma.std().item(),
            'bias_sigma_mean': self.bias_sigma.abs().mean().item(), 
            'bias_sigma_std': self.bias_sigma.std().item(),
            'noise_scale': (weight_noise.abs().mean() + bias_noise.abs().mean()).item() / 2,
            'signal_to_noise_ratio': (weight_snr + bias_snr) / 2
        }