"""
Neural network models for DQN training and inference.
"""

import torch
import torch.nn as nn


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
