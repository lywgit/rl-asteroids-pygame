"""
Model factory functions for creating and loading DQN models with proper configuration.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
import numpy as np
from pathlib import Path

from .model_config import ModelConfig
from .models import AtariDQN, AtariDistributionalDQN


def create_model_from_config(model_config: ModelConfig, device: str = 'cpu') -> nn.Module:
    """
    Create a DQN model instance from ModelConfig.
    
    Args:
        model_config: Model configuration containing architecture parameters
        device: Device to place the model on
        
    Returns:
        Configured model instance
    """
    if model_config.distributional_dqn:
        # Create distributional model
        model = AtariDistributionalDQN(
            input_shape=model_config.input_shape,
            n_action=model_config.n_action,
            n_atoms=model_config.n_atoms or 51,  # Default fallback
            v_min=model_config.v_min or -10.0,   # Default fallback
            v_max=model_config.v_max or 10.0,    # Default fallback
            dueling=model_config.dueling_dqn,
            noisy=model_config.noisy_networks,
            std_init=model_config.noisy_std_init
        )
    else:
        # Create standard model
        model = AtariDQN(
            input_shape=model_config.input_shape,
            n_action=model_config.n_action,
            dueling=model_config.dueling_dqn,
            noisy=model_config.noisy_networks,
            std_init=model_config.noisy_std_init
        )
    
    return model.to(device)


def save_model_checkpoint(model: nn.Module, model_config: ModelConfig, 
                         training_info: Dict[str, Any], checkpoint_path: str) -> None:
    """
    Save model checkpoint with complete metadata.
    
    Args:
        model: The trained model to save
        model_config: Model configuration
        training_info: Training information and metrics
        checkpoint_path: Path to save the checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config.to_dict(),
        'training_info': training_info,
        'version': '2.0'  # Checkpoint format version
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"✅ Model checkpoint saved: {checkpoint_path}")
    print(f"   Architecture: {model_config.get_architecture_name()}")


def load_model_checkpoint(checkpoint_path: str, env, device: str = 'cpu') -> Tuple[nn.Module, ModelConfig, Dict[str, Any]]:
    """
    Load model checkpoint with automatic architecture reconstruction.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        env: Environment (needed for legacy checkpoint inference)
        device: Device to load the model on
        
    Returns:
        Tuple of (model, model_config, training_info)
    """
    print(f"📂 Loading model checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    # Determine checkpoint format version
    checkpoint_version = checkpoint.get('version', '1.0')
    
    if checkpoint_version == '2.0':
        # Enhanced checkpoint format
        print("   Format: Enhanced (v2.0)")
        state_dict = checkpoint['model_state_dict']
        model_config = ModelConfig.from_dict(checkpoint['model_config'])
        training_info = checkpoint.get('training_info', {})
        
    elif checkpoint_version == '1.0' or isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint:
        # Legacy checkpoint format (just state_dict)
        print("   Format: Legacy (v1.0) - inferring architecture...")
        
        # Handle both raw state_dict and wrapped state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint:
            # Raw state_dict format
            state_dict = checkpoint
        else:
            # Wrapped but old format
            state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Infer model configuration from state_dict
        model_config = infer_model_config_from_state_dict(state_dict, env)
        training_info = {}
        
    else:
        raise ValueError(f"Unsupported checkpoint version: {checkpoint_version}")
    
    # Create model from configuration
    print(f"   Architecture: {model_config.get_architecture_name()}")
    model = create_model_from_config(model_config, device)
    
    # Load weights
    try:
        model.load_state_dict(state_dict)
        print("✅ Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")
    
    return model, model_config, training_info


def infer_model_config_from_state_dict(state_dict: Dict[str, torch.Tensor], env) -> ModelConfig:
    """
    Infer model configuration from state_dict structure (for legacy checkpoints).
    
    Args:
        state_dict: Model state dictionary
        env: Environment to get input/output dimensions
        
    Returns:
        Inferred ModelConfig
    """
    # Analyze state_dict keys to determine architecture
    keys = list(state_dict.keys())
    
    # Check for dueling architecture
    has_value_stream = any(key.startswith('value_stream') for key in keys)
    has_advantage_stream = any(key.startswith('advantage_stream') for key in keys)
    dueling_dqn = has_value_stream and has_advantage_stream
    
    # Check for noisy networks
    noisy_networks = any('weight_mu' in key or 'weight_sigma' in key for key in keys)
    
    # Check for distributional DQN and infer action count
    distributional_dqn = False
    n_atoms = 51
    n_action = env.action_space.n  # Initial guess from environment
    
    # Method 1: Check for 'support' buffer (definitive indicator)
    if 'support' in keys:
        distributional_dqn = True
        # Get n_atoms from support buffer size
        n_atoms = state_dict['support'].shape[0]
        print(f"   Distributional detected by 'support' buffer: {n_atoms} atoms")
        
        # Now infer the actual action count from the model structure
        final_layer_keys = []
        if noisy_networks:
            if dueling_dqn:
                final_layer_keys = [k for k in keys if k.startswith('advantage_stream') and k.endswith('.weight_mu')]
            else:
                final_layer_keys = [k for k in keys if k.startswith('fc.') and k.endswith('.weight_mu')]
        else:
            if dueling_dqn:
                final_layer_keys = [k for k in keys if k.startswith('advantage_stream') and k.endswith('.weight')]
            else:
                final_layer_keys = [k for k in keys if k.startswith('fc.') and k.endswith('.weight')]
        
        if final_layer_keys:
            # Get the last layer (highest index)
            final_layer_key = sorted(final_layer_keys)[-1]
            final_layer_size = state_dict[final_layer_key].shape[0]
            n_action = final_layer_size // n_atoms
            print(f"   Inferred actions from model structure: {n_action} (layer size: {final_layer_size} ÷ {n_atoms} atoms)")
    
    else:
        # Method 2: Infer from final layer output size
        final_layer_keys = []
        
        # For noisy networks, look for weight_mu parameters
        if noisy_networks:
            if dueling_dqn:
                final_layer_keys = [k for k in keys if k.startswith('advantage_stream') and k.endswith('.weight_mu')]
            else:
                final_layer_keys = [k for k in keys if k.startswith('fc.') and k.endswith('.weight_mu')]
        else:
            # For standard networks, look for weight parameters
            if dueling_dqn:
                final_layer_keys = [k for k in keys if k.startswith('advantage_stream') and k.endswith('.weight')]
            else:
                final_layer_keys = [k for k in keys if k.startswith('fc.') and k.endswith('.weight')]
        
        if final_layer_keys:
            # Get the last layer (highest index)
            final_layer_key = sorted(final_layer_keys)[-1]
            final_layer_size = state_dict[final_layer_key].shape[0]
            
            print(f"   Final layer '{final_layer_key}' output size: {final_layer_size}")
            print(f"   Environment actions: {n_action}")
            
            # Check if this is likely distributional
            if final_layer_size > n_action and final_layer_size % n_action == 0:
                # Try common atom values
                for candidate_atoms in [51, 21, 101]:  # Common C51 values
                    if final_layer_size % candidate_atoms == 0:
                        candidate_actions = final_layer_size // candidate_atoms
                        # Prefer the candidate that makes more sense
                        if candidate_actions >= n_action:  # Actions should be >= env actions
                            n_atoms = candidate_atoms
                            n_action = candidate_actions
                            distributional_dqn = True
                            print(f"   Distributional inferred: {n_atoms} atoms, {n_action} actions")
                            break
            
            # If not distributional, the layer size should match action count
            if not distributional_dqn:
                n_action = final_layer_size
                print(f"   Standard DQN: {n_action} actions")
    
    # For distributional models, we need to estimate v_min/v_max
    v_min = None
    v_max = None
    if distributional_dqn:
        from .distributional_utils import get_value_range_for_game
        # Try to get game name from common patterns, default to generic range
        try:
            game_name = 'asteroids'  # Default fallback
            v_min, v_max = get_value_range_for_game(game_name)
        except:
            v_min, v_max = -10.0, 10.0  # Safe default
    
    print(f"   Inferred: {'Distributional ' if distributional_dqn else ''}{'Dueling ' if dueling_dqn else ''}{'Noisy ' if noisy_networks else ''}DQN")
    if distributional_dqn:
        print(f"   Atoms: {n_atoms}, Range: [{v_min}, {v_max}]")
    
    return ModelConfig(
        input_shape=env.observation_space.shape,
        n_action=n_action,
        dueling_dqn=dueling_dqn,
        distributional_dqn=distributional_dqn,
        noisy_networks=noisy_networks,
        n_atoms=n_atoms if distributional_dqn else None,
        v_min=v_min,
        v_max=v_max,
        # Note: We can't infer these from state_dict, use defaults
        double_dqn=False,  # This is a training technique, not architectural
        prioritized_replay=False,  # This affects buffer, not model
        n_step_learning=False,  # This affects training, not model
        created_at=None,
        game=None
    )