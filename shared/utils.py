"""
Utility functions for device detection and other common operations.
"""

import torch


def get_device():
    """Get the best available device for training/inference"""
    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        return "cuda"
    # Check for MPS (Apple Silicon GPUs)
    elif torch.backends.mps.is_available():
        return "mps"
    # Fallback to CPU
    else:
        return "cpu"
