import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TorchMockModel(nn.Module):
    """
    A PyTorch-based mock AI model for Asteroids that takes the RGB observation
    and outputs actions via a simple neural network.
    Output: np.ndarray of shape (5,) with binary values for [thrust, backward, left, right, shoot]
    """
    def __init__(self, input_height=720, input_width=1280, input_channels=3):
        super(TorchMockModel, self).__init__()
        
        # Simple CNN feature extractor
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        
        # Calculate the actual size by running a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            conv_out_size = x.view(1, -1).size(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)  # 5 actions: [thrust, backward, left, right, shoot]
        
        # Initialize weights randomly for mock behavior
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with small random values for mock behavior"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network"""
        # x shape: (batch_size, height, width, channels) or (height, width, channels)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Convert from HWC to CHW format for PyTorch
        x = x.permute(0, 3, 1, 2).float()
        
        # Normalize pixel values to [0, 1]
        x = x / 255.0
        
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Apply sigmoid to get probabilities, then threshold at 0.5 for binary actions
        x = torch.sigmoid(x)
        
        return x
    
    def __call__(self, obs):
        """
        Interface method to match the MockModel class
        Args:
            obs: numpy array of shape (height, width, 3) - RGB observation
        Returns:
            numpy array of shape (5,) with binary values for actions
        """
        # Convert observation to tensor
        obs_tensor = torch.from_numpy(obs).float()
        
        # Forward pass (no gradient computation needed for inference)
        with torch.no_grad():
            action_probs = self.forward(obs_tensor)
            
            # Convert probabilities to binary actions (threshold at 0.5)
            actions = (action_probs > 0.5).float()
            
            # Remove batch dimension and convert to numpy
            actions = actions.squeeze(0).numpy().astype(np.int32)
        
        return actions


class RandomTorchMockModel:
    """
    A simpler random mock model using PyTorch for comparison
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __call__(self, obs):
        """
        Generate random actions with some biases to make it more interesting
        """
        # Create random probabilities with some biases
        probs = torch.rand(5, device=self.device)
        
        # Bias towards certain actions to make gameplay more interesting
        probs[0] *= 0.7  # thrust - moderate probability
        probs[1] *= 0.1  # backward - low probability
        probs[2] *= 0.4  # left - moderate probability
        probs[3] *= 0.4  # right - moderate probability
        probs[4] *= 0.3  # shoot - moderate probability
        
        # Convert to binary actions
        actions = (probs > 0.3).cpu().numpy().astype(np.int32)
        
        return actions


if __name__ == "__main__":
    # Test the model with a dummy observation
    model = TorchMockModel()
    dummy_obs = np.random.randint(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    
    print("Testing TorchMockModel...")
    action = model(dummy_obs)
    print(f"Action output: {action}")
    print(f"Action shape: {action.shape}")
    print(f"Action type: {type(action)}")
    
    # Test random model
    random_model = RandomTorchMockModel()
    print("\nTesting RandomTorchMockModel...")
    random_action = random_model(dummy_obs)
    print(f"Random action output: {random_action}")
    print(f"Random action shape: {random_action.shape}")
    print(f"Random action type: {type(random_action)}")
