# Model Config Save/Load Implementation Plan

## Overview
Implement a comprehensive model configuration system that saves model architecture and hyperparameter information alongside PyTorch state dictionaries, enabling automatic model reconstruction during loading.

## Current Issues
1. **Model checkpoints only save `state_dict()`** - No architecture information
2. **Manual model creation required** - Users must know exact architecture parameters  
3. **play_dqn_model.py tries multiple architectures** - Error-prone fallback approach
4. **No versioning or metadata** - Cannot track model training context
5. **Inconsistent loading across DQN technique combinations** - 8 possible architecture combinations not fully supported

## Goals
1. **Automatic Model Reconstruction** - Load model with correct architecture automatically
2. **Complete Metadata Storage** - Save all necessary config and training information
3. **Backward Compatibility** - Support loading existing checkpoints
4. **Future-Proof Design** - Easy to extend with new DQN techniques
5. **MLOps Ready** - Proper versioning and reproducibility

## Implementation Plan

### Phase 1: Enhanced Checkpoint Format

#### 1.1 Create ModelConfig Class
```python
# shared/model_config.py
@dataclass
class ModelConfig:
    """Complete model configuration for saving/loading"""
    # Architecture parameters
    input_shape: Tuple[int, int, int]
    n_action: int
    
    # DQN technique flags  
    double_dqn: bool = False
    dueling_dqn: bool = False
    distributional_dqn: bool = False
    noisy_networks: bool = False
    prioritized_replay: bool = False
    n_step_learning: bool = False
    
    # Distributional parameters (when applicable)
    n_atoms: Optional[int] = None
    v_min: Optional[float] = None
    v_max: Optional[float] = None
    
    # Noisy networks parameters
    noisy_std_init: float = 0.5
    
    # Multi-step parameters
    n_steps: int = 1
    
    # Training metadata
    game: str
    training_config_hash: Optional[str] = None
    model_version: str = "2.0"
    created_at: Optional[str] = None
    framework_version: str = "asteroids-ai-v2.0"
```

#### 1.2 Enhanced Checkpoint Structure
```python
# New checkpoint format
checkpoint = {
    'model_state_dict': net.state_dict(),
    'model_config': model_config.to_dict(),
    'training_info': {
        'frame_idx': frame_idx,
        'episode_idx': episode_idx,
        'best_eval_reward': best_eval_reward,
        'training_config': config,
        'hyperparameters': hyperparameters,
        'created_at': datetime.now().isoformat(),
        'device_used': str(device)
    },
    'version': '2.0'
}
```

#### 1.3 Model Factory Function
```python
# shared/model_factory.py
def create_model_from_config(model_config: ModelConfig, device: str = 'cpu') -> nn.Module:
    """Create model instance from ModelConfig"""
    
def load_model_checkpoint(checkpoint_path: str, device: str = 'cpu') -> Tuple[nn.Module, ModelConfig, dict]:
    """Load model with full metadata from checkpoint"""
    
def save_model_checkpoint(model: nn.Module, model_config: ModelConfig, 
                         training_info: dict, checkpoint_path: str) -> None:
    """Save model with complete metadata"""
```

### Phase 2: Training Integration

#### 2.1 Update train_dqn.py
- [x] Extract model config from training config
- [x] Use ModelConfig in create_dqn_networks()  
- [x] Update checkpoint saving (both regular and best)
- [x] Add config validation before training

#### 2.2 Enhanced Checkpoint Saving
```python
# Replace current torch.save(net.state_dict(), path)
model_config = ModelConfig.from_training_config(config, env)
save_model_checkpoint(
    model=net,
    model_config=model_config, 
    training_info={
        'frame_idx': frame_idx,
        'episode_idx': episode_idx,
        'best_eval_reward': best_eval_reward,
        # ... other metadata
    },
    checkpoint_path=checkpoint_path
)
```

### Phase 3: Loading and Inference

#### 3.1 Update play_dqn_model.py
- [x] Remove trial-and-error loading approach
- [x] Use model factory for automatic reconstruction
- [x] Support both old and new checkpoint formats
- [x] Display model architecture information

#### 3.2 Backward Compatibility Layer
```python
def load_legacy_checkpoint(checkpoint_path: str, env, device: str) -> nn.Module:
    """Load old-format checkpoints with intelligent architecture detection"""
    # Try to infer architecture from state_dict shapes
    # Fallback to multiple architecture attempts
```

### Phase 4: Validation and Testing

#### 4.1 Architecture Combination Testing
- [x] Test all 8 DQN technique combinations (2^3):
  - Standard DQN
  - Dueling DQN  
  - Distributional DQN
  - Dueling + Distributional
  - Noisy DQN
  - Dueling + Noisy
  - Distributional + Noisy  
  - Dueling + Distributional + Noisy

#### 4.2 Round-trip Testing
```python
def test_model_save_load_roundtrip():
    """Test that saved models can be loaded correctly"""
    # For each architecture combination:
    # 1. Create model with config
    # 2. Save checkpoint
    # 3. Load checkpoint  
    # 4. Verify architecture matches
    # 5. Verify state_dict compatibility
```

### Phase 5: Documentation and Migration

#### 5.1 Update Documentation
- [x] Update README.md with new checkpoint format
- [x] Add migration guide for existing checkpoints  
- [x] Document ModelConfig usage

#### 5.2 Migration Utilities
```python
# utils/migrate_checkpoints.py  
def migrate_old_checkpoint(old_path: str, new_path: str, model_config: ModelConfig):
    """Convert old checkpoint to new format"""
```

## File Structure Changes

```
shared/
├── model_config.py          # New: ModelConfig dataclass
├── model_factory.py         # New: Model creation and loading utilities  
├── models.py               # Updated: Add config integration
└── checkpoint_utils.py     # New: Checkpoint handling utilities

utils/
└── migrate_checkpoints.py  # New: Legacy checkpoint migration

train_dqn.py               # Updated: Use new checkpoint format
play_dqn_model.py          # Updated: Use model factory
```

## Benefits

### 1. **Automatic Model Reconstruction**
```python
# Before (manual architecture guessing)
try:
    model = AtariDuelingDQN(env.observation_space.shape, env.action_space.n)
    model.load_state_dict(torch.load(path))
except:
    model = AtariDQN(env.observation_space.shape, env.action_space.n)  
    model.load_state_dict(torch.load(path))

# After (automatic reconstruction)
model, config, training_info = load_model_checkpoint(path, device)
print(f"Loaded {config.get_architecture_name()} trained on {training_info['game']}")
```

### 2. **Complete Reproducibility**
- Exact training configuration preserved
- Model architecture automatically reconstructed
- Training metadata available for analysis

### 3. **MLOps Integration Ready**
- Model versioning built-in
- Training lineage tracking
- Experiment metadata preservation

### 4. **Error Prevention** 
- No more architecture guessing
- Validate model compatibility before loading
- Clear error messages for unsupported formats

## Implementation Priority

1. **High Priority** (Required for TODO completion):
   - ModelConfig class and model factory
   - Updated checkpoint saving in train_dqn.py
   - Updated loading in play_dqn_model.py
   - Backward compatibility for existing checkpoints

2. **Medium Priority** (Nice to have):
   - Migration utilities for old checkpoints
   - Enhanced metadata and versioning
   - Comprehensive testing suite

3. **Low Priority** (Future enhancements):
   - Web-based model inspector
   - Checkpoint compression
   - Distributed training support

## Success Criteria

- [x] All 8 DQN architecture combinations save/load correctly
- [x] play_dqn_model.py works without architecture guessing
- [x] Existing checkpoints remain loadable (backward compatibility)
- [x] Model architecture displayed when loading
- [x] Training configuration preserved in checkpoints
- [x] No breaking changes to existing workflows

## Timeline

- **Week 1**: Phase 1 - ModelConfig and enhanced checkpoint format
- **Week 2**: Phase 2 - Training integration and checkpoint saving  
- **Week 3**: Phase 3 - Loading updates and backward compatibility
- **Week 4**: Phase 4 - Testing and validation
- **Week 5**: Phase 5 - Documentation and migration tools

This plan addresses the core TODO requirement while setting up a robust foundation for future MLOps enhancements.