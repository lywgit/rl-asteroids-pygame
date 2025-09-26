"""
Neural network models for Atari DQN.
"""
import torch
import torch.nn as nn
from typing import Tuple
from .noisy_networks import NoisyLinear


def _linear_layer(in_features: int, out_features: int, noisy: bool = False, std_init: float = 0.5):
    """Create either a standard Linear layer or NoisyLinear layer based on the noisy parameter."""
    if noisy:
        return NoisyLinear(in_features, out_features, std_init=std_init)
    else:
        return nn.Linear(in_features, out_features)


class AtariDQN(nn.Module):
    """
    Standard Atari DQN network with optional dueling and noisy architectures.
    Outputs Q-values directly.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], n_action: int, 
                 dueling: bool = False, noisy: bool = False, std_init: float = 0.5):
        super(AtariDQN, self).__init__()
        
        self.input_shape = input_shape
        self.n_action = n_action
        self.dueling = dueling
        self.noisy = noisy
        
        # Print initialization info
        arch_parts = []
        if dueling:
            arch_parts.append("Dueling")
        if noisy:
            arch_parts.append("Noisy")
        arch_parts.append("DQN")
        
        arch_name = " ".join(arch_parts)
        print(f"AtariDQN initialized: {arch_name}, input_shape: {input_shape}")
        
        # Shared convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out = self._get_conv_out(input_shape)
        
        if dueling:
            # Dueling architecture: separate value and advantage streams
            self.value_stream = nn.Sequential(
                _linear_layer(conv_out, 512, noisy=noisy, std_init=std_init),
                nn.ReLU(),
                _linear_layer(512, 1, noisy=noisy, std_init=std_init)  # Single value
            )
            self.advantage_stream = nn.Sequential(
                _linear_layer(conv_out, 512, noisy=noisy, std_init=std_init),
                nn.ReLU(),
                _linear_layer(512, n_action, noisy=noisy, std_init=std_init)  # Advantage values
            )
        else:
            # Standard architecture: single output stream
            self.fc = nn.Sequential(
                _linear_layer(conv_out, 512, noisy=noisy, std_init=std_init),
                nn.ReLU(),
                _linear_layer(512, n_action, noisy=noisy, std_init=std_init)  # Q-values
            )

    def _get_conv_out(self, input_shape: Tuple[int, int, int]) -> int:
        """Calculate output size of convolutional layers."""
        dummy = torch.zeros(1, *input_shape)
        output = self.conv(dummy)
        return output.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Returns:
            [batch_size, n_actions] - Q-values
        """
        batch_size = x.size(0)
        conv_out = self.conv(x).view(batch_size, -1)
        
        if self.dueling:
            # Dueling architecture
            value = self.value_stream(conv_out)  # [batch_size, 1]
            advantage = self.advantage_stream(conv_out)  # [batch_size, n_actions]
            
            # Dueling combination: V(s) + A(s,a) - mean(A(s,·))
            qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return qvals
        else:
            # Standard architecture
            qvals = self.fc(conv_out)
            return qvals

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Get Q-values from the network (same as forward for standard DQN)."""
        return self.forward(x)


class AtariDistributionalDQN(nn.Module):
    """
    Distributional Atari DQN network with optional dueling and noisy architectures.
    Outputs probability distributions over value atoms.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], n_action: int, 
                 n_atoms: int = 51, v_min: float = -10.0, v_max: float = 10.0, 
                 dueling: bool = False, noisy: bool = False, std_init: float = 0.5):
        super(AtariDistributionalDQN, self).__init__()
        
        self.input_shape = input_shape
        self.n_action = n_action
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.dueling = dueling
        self.noisy = noisy
        
        # Print initialization info
        arch_parts = ["Distributional"]
        if dueling:
            arch_parts.append("Dueling")
        if noisy:
            arch_parts.append("Noisy")
        arch_parts.append("DQN")
        
        arch_name = " ".join(arch_parts)
        print(f"AtariDistributionalDQN initialized: {arch_name} (C51, {n_atoms} atoms), input_shape: {input_shape}")
        print(f"Support range: [{v_min}, {v_max}]")
        
        # Register distributional support as buffer
        support = torch.linspace(v_min, v_max, n_atoms)
        self.register_buffer('support', support)
        
        # Shared convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out = self._get_conv_out(input_shape)
        
        if dueling:
            # Dueling architecture: separate value and advantage streams
            self.value_stream = nn.Sequential(
                _linear_layer(conv_out, 512, noisy=noisy, std_init=std_init),
                nn.ReLU(),
                _linear_layer(512, n_atoms, noisy=noisy, std_init=std_init)  # Value distribution
            )
            self.advantage_stream = nn.Sequential(
                _linear_layer(conv_out, 512, noisy=noisy, std_init=std_init),
                nn.ReLU(),
                _linear_layer(512, n_action * n_atoms, noisy=noisy, std_init=std_init)  # Advantage distributions
            )
        else:
            # Standard architecture: single output stream
            self.fc = nn.Sequential(
                _linear_layer(conv_out, 512, noisy=noisy, std_init=std_init),
                nn.ReLU(),
                _linear_layer(512, n_action * n_atoms, noisy=noisy, std_init=std_init)  # Action-value distributions
            )

    def _get_conv_out(self, input_shape: Tuple[int, int, int]) -> int:
        """Calculate output size of convolutional layers."""
        dummy = torch.zeros(1, *input_shape)
        output = self.conv(dummy)
        return output.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Returns:
            [batch_size, n_actions, n_atoms] - probability distributions
        """
        batch_size = x.size(0)
        conv_out = self.conv(x).view(batch_size, -1)
        
        if self.dueling:
            # Dueling architecture
            value_logits = self.value_stream(conv_out)  # [batch_size, n_atoms]
            advantage_logits = self.advantage_stream(conv_out).view(
                batch_size, self.n_action, self.n_atoms
            )  # [batch_size, n_actions, n_atoms]
            
            # Dueling combination per atom: V(s) + A(s,a) - mean(A(s,·))
            value_expanded = value_logits.unsqueeze(1)  # [batch_size, 1, n_atoms]
            advantage_mean = advantage_logits.mean(dim=1, keepdim=True)  # [batch_size, 1, n_atoms]
            combined_logits = value_expanded + advantage_logits - advantage_mean
            
            # Apply softmax over atoms to get probability distributions
            probs = torch.softmax(combined_logits, dim=2)
            return probs
        else:
            # Standard architecture
            logits = self.fc(conv_out)  # [batch_size, n_actions * n_atoms]
            logits = logits.view(batch_size, self.n_action, self.n_atoms)
            
            # Apply softmax over atoms to get probability distributions
            probs = torch.softmax(logits, dim=2)
            return probs

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values by computing expected values from probability distributions.
        
        Returns:
            [batch_size, n_actions] - Expected Q-values
        """
        probs = self.forward(x)  # [batch_size, n_actions, n_atoms]
        
        # Compute expected Q-values from probability distributions
        # probs: [batch_size, n_actions, n_atoms]
        # support: [n_atoms]
        q_values = torch.sum(probs * torch.as_tensor(self.support).view(1, 1, -1), dim=2)
        return q_values


# Backward compatibility aliases
class AtariDuelingDQN(AtariDQN):
    """Backward compatibility alias for Dueling DQN."""
    def __init__(self, input_shape: Tuple[int, int, int], n_action: int):
        super().__init__(input_shape, n_action, dueling=True)


class AtariDistributionalDuelingDQN(AtariDistributionalDQN):
    """Backward compatibility alias for Distributional Dueling DQN."""
    def __init__(self, input_shape: Tuple[int, int, int], n_action: int,
                 n_atoms: int = 51, v_min: float = -10.0, v_max: float = 10.0):
        super().__init__(input_shape, n_action, n_atoms=n_atoms, v_min=v_min, v_max=v_max, dueling=True)