"""
Neural network models for DQN training and inference.
"""

import torch
import torch.nn as nn
from typing import Tuple


class AtariDQN(nn.Module):
    """Q Learning network mapping pixel observations to action values"""
    def __init__(self, input_shape, n_action):
        # input_shape = (4, 84, 84)=(T, H, W) for grayscale_obs
        # T is frame stacking time dimension
        # Note:
        # input shape (4, 84, 84)   -> conv output shape (64, 7, 7) -> flattened size 3136
        # input shape (4, 128, 128) -> conv output shape (64, 12, 12) -> flattened size 9216

        print("AtariDQN initialized with input shape:", input_shape)
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
        # print("conv output size:", output.shape[1:])
        return output.view(1, -1).size(1)

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size()[0], -1))


class AtariDuelingDQN(nn.Module):
    """Dueling DQN network: shared conv, separate value and advantage streams"""
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
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_action)
        )

    def _get_conv_out(self, input_shape):
        dummy = torch.zeros(1, *input_shape)
        output = self.conv(dummy)
        return output.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return qvals


class AtariDistributionalDQN(nn.Module):
    """Distributional DQN using categorical distributions (C51)"""
    def __init__(self, input_shape, n_action, n_atoms=51, v_min=-10.0, v_max=10.0):
        print(f"AtariDistributionalDQN initialized with input shape: {input_shape}, n_atoms: {n_atoms}, v_range: [{v_min}, {v_max}]")
        super().__init__()
        
        self.n_action = n_action
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Register support as buffer (not trainable parameter)
        support = torch.linspace(v_min, v_max, n_atoms)
        self.register_buffer('support', support)
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out = self._get_conv_out(input_shape)
        
        # Output layer: [n_action * n_atoms] then reshape to [n_action, n_atoms]
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_action * n_atoms)
        )

    def _get_conv_out(self, input_shape):
        dummy = torch.zeros(1, *input_shape)
        output = self.conv(dummy)
        return output.view(1, -1).size(1)

    def forward(self, x):
        batch_size = x.size(0)
        conv_out = self.conv(x).view(batch_size, -1)
        logits = self.fc(conv_out)
        
        # Reshape to [batch_size, n_action, n_atoms]
        logits = logits.view(batch_size, self.n_action, self.n_atoms)
        
        # Apply softmax over atoms dimension to get probability distributions
        probs = torch.softmax(logits, dim=2)
        return probs
    
    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Get expected Q-values by taking expectation over atoms"""
        probs = self.forward(x)
        # Cast to tensor to satisfy type checker - self.support is registered as a buffer
        support = torch.as_tensor(self.support)
        q_values = (probs * support).sum(dim=2)
        return q_values


class AtariDistributionalDuelingDQN(nn.Module):
    """Distributional Dueling DQN with categorical distributions"""
    def __init__(self, input_shape, n_action, n_atoms=51, v_min=-10.0, v_max=10.0):
        print(f"AtariDistributionalDuelingDQN initialized with input shape: {input_shape}, n_atoms: {n_atoms}, v_range: [{v_min}, {v_max}]")
        super().__init__()
        
        self.n_action = n_action
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Register support as buffer (not trainable parameter)
        support = torch.linspace(v_min, v_max, n_atoms)
        self.register_buffer('support', support)
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out = self._get_conv_out(input_shape)
        
        # Value stream: outputs [n_atoms] (state value distribution)
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_atoms)
        )
        
        # Advantage stream: outputs [n_action * n_atoms] 
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_action * n_atoms)
        )

    def _get_conv_out(self, input_shape):
        dummy = torch.zeros(1, *input_shape)
        output = self.conv(dummy)
        return output.view(1, -1).size(1)

    def forward(self, x):
        batch_size = x.size(0)
        conv_out = self.conv(x).view(batch_size, -1)
        
        # Value stream: [batch_size, n_atoms]
        value_logits = self.value_stream(conv_out)
        
        # Advantage stream: [batch_size, n_action, n_atoms]
        advantage_logits = self.advantage_stream(conv_out).view(batch_size, self.n_action, self.n_atoms)
        
        # Dueling combination: V(s) + A(s,a) - mean(A(s,·)) for each atom
        # Broadcast value across actions: [batch_size, 1, n_atoms] 
        value_expanded = value_logits.unsqueeze(1)
        advantage_mean = advantage_logits.mean(dim=1, keepdim=True)
        
        # Combine using dueling formula per atom
        combined_logits = value_expanded + advantage_logits - advantage_mean
        
        # Apply softmax over atoms dimension to get probability distributions
        probs = torch.softmax(combined_logits, dim=2)
        return probs
    
    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Get expected Q-values by taking expectation over atoms"""
        probs = self.forward(x)
        # Cast to tensor to satisfy type checker - self.support is registered as a buffer
        support = torch.as_tensor(self.support)
        q_values = (probs * support).sum(dim=2)
        return q_values
