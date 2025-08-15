"""Base model interface with save/load capabilities."""

from abc import ABC, abstractmethod
import torch
import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np

class BaseModel(ABC):
    """Abstract base class for all Asteroids AI models."""
    
    def __init__(self, model_name="base_model"):
        self.model_name = model_name
        self.training_info = {
            "created_at": datetime.now().isoformat(),
            "total_episodes": 0,
            "best_score": 0,
            "best_reward": 0.0,
            "training_time_hours": 0.0
        }
    
    @abstractmethod
    def __call__(self, obs):
        """Generate actions from observations."""
        pass
    
    def save(self, path, include_metadata=True):
        """
        Save model to disk.
        
        Args:
            path (str): Path to save the model
            include_metadata (bool): Whether to save training metadata
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self, 'state_dict'):  # PyTorch model
            save_dict = {
                'model_state_dict': self.state_dict(),
                'model_name': self.model_name,
                'model_class': self.__class__.__name__,
                'training_info': self.training_info
            }
            
            # Save model architecture parameters if available
            if hasattr(self, 'get_config'):
                save_dict['model_config'] = self.get_config()
            
            torch.save(save_dict, path)
            
            if include_metadata:
                metadata_path = path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump({
                        'model_name': self.model_name,
                        'model_class': self.__class__.__name__,
                        'training_info': self.training_info,
                        'saved_at': datetime.now().isoformat(),
                        'file_size_mb': round(os.path.getsize(path) / 1024 / 1024, 2)
                    }, f, indent=2)
        else:
            # For non-PyTorch models, save as pickle or custom format
            raise NotImplementedError(f"Save not implemented for {self.__class__.__name__}")
    
    @classmethod
    def load(cls, path, device='cpu'):
        """
        Load model from disk.
        
        Args:
            path (str): Path to the saved model
            device (str): Device to load the model on
            
        Returns:
            BaseModel: Loaded model instance
        """
        path = Path(path)
        
        if path.suffix == '.pth':
            checkpoint = torch.load(path, map_location=device)
            
            # Create model instance (requires model_config if available)
            if 'model_config' in checkpoint:
                model = cls(**checkpoint['model_config'])
            else:
                model = cls()  # Use default parameters
            
            # Load state dict if it's a PyTorch model
            if hasattr(model, 'load_state_dict'):
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()  # Set to evaluation mode
            
            # Restore metadata
            model.model_name = checkpoint.get('model_name', 'loaded_model')
            model.training_info = checkpoint.get('training_info', {})
            
            return model
        else:
            raise NotImplementedError(f"Load not implemented for {path.suffix} files")
    
    def update_training_info(self, **kwargs):
        """Update training information."""
        self.training_info.update(kwargs)


class ModelRegistry:
    """Registry for tracking saved models and their performance."""
    
    def __init__(self, registry_path="saved_models/model_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()
    
    def _load_registry(self):
        """Load existing registry or create new one."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {"models": [], "best_score": 0, "best_reward": 0.0}
    
    def register_model(self, model_path, model_name, score=0, reward=0.0, **metadata):
        """Register a new model in the registry."""
        entry = {
            "model_name": model_name,
            "path": str(model_path),
            "score": score,
            "reward": reward,
            "registered_at": datetime.now().isoformat(),
            **metadata
        }
        
        self.registry["models"].append(entry)
        
        # Update best scores
        if score > self.registry["best_score"]:
            self.registry["best_score"] = score
            self.registry["best_score_model"] = model_name
        
        if reward > self.registry["best_reward"]:
            self.registry["best_reward"] = reward
            self.registry["best_reward_model"] = model_name
        
        self._save_registry()
    
    def _save_registry(self):
        """Save registry to disk."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def get_best_models(self, metric="score", top_k=5):
        """Get top k models by specified metric."""
        sorted_models = sorted(
            self.registry["models"], 
            key=lambda x: x.get(metric, 0), 
            reverse=True
        )
        return sorted_models[:top_k]
    
    def get_model_by_name(self, model_name):
        """Find model entry by name."""
        for model in self.registry["models"]:
            if model["model_name"] == model_name:
                return model
        return None
