"""
Model configuration for saving and loading DQN models.
"""

from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, Any
from datetime import datetime


@dataclass
class ModelConfig:
    """
    Complete model configuration for saving/loading DQN models.
    Contains all parameters needed to reconstruct the exact model architecture.
    """
    
    # Model architecture parameters
    input_shape: Tuple[int, int, int]
    n_action: int
    
    # DQN technique flags
    double_dqn: bool = False
    dueling_dqn: bool = False
    distributional_dqn: bool = False
    noisy_networks: bool = False
    prioritized_replay: bool = False
    n_step_learning: bool = False
    
    # Distributional DQN parameters (when distributional_dqn=True)
    n_atoms: Optional[int] = None
    v_min: Optional[float] = None
    v_max: Optional[float] = None
    
    # Noisy networks parameters (when noisy_networks=True)
    noisy_std_init: float = 0.5
    
    # Multi-step learning parameters (when n_step_learning=True)
    n_steps: int = 1
    
    # Optional metadata (not used for model creation)
    game: Optional[str] = None
    created_at: Optional[str] = None
    
    @classmethod
    def from_training_config(cls, config: Dict[str, Any], env) -> 'ModelConfig':
        """Create ModelConfig from training configuration and environment."""
        from .distributional_utils import get_value_range_for_game
        
        # Extract DQN technique flags
        double_dqn = config.get('double_dqn', False)
        dueling_dqn = config.get('dueling_dqn', False)
        distributional_dqn = config.get('distributional_dqn', False)
        noisy_networks = config.get('noisy_networks', False)
        prioritized_replay = config.get('prioritized_replay', False)
        n_step_learning = config.get('n_step_learning', False)
        
        # Distributional parameters
        n_atoms = None
        v_min = None
        v_max = None
        if distributional_dqn:
            n_atoms = config.get('n_atoms', 51)
            v_min_config = config.get('v_min')
            v_max_config = config.get('v_max')
            
            if v_min_config is None or v_max_config is None:
                # Auto-estimate value range
                game_name = config['game']
                estimated_v_min, estimated_v_max = get_value_range_for_game(game_name)
                v_min = v_min_config if v_min_config is not None else estimated_v_min
                v_max = v_max_config if v_max_config is not None else estimated_v_max
            else:
                v_min = v_min_config
                v_max = v_max_config
        
        # Other parameters
        noisy_std_init = config.get('noisy_std_init', 0.5)
        n_steps = config.get('n_steps', 1)
        
        return cls(
            input_shape=env.observation_space.shape,
            n_action=env.action_space.n,
            double_dqn=double_dqn,
            dueling_dqn=dueling_dqn,
            distributional_dqn=distributional_dqn,
            noisy_networks=noisy_networks,
            prioritized_replay=prioritized_replay,
            n_step_learning=n_step_learning,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
            noisy_std_init=noisy_std_init,
            n_steps=n_steps,
            game=config.get('game'),
            created_at=datetime.now().isoformat()
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary (e.g., loaded from checkpoint)."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ModelConfig to dictionary for saving."""
        return asdict(self)
    
    def get_architecture_name(self) -> str:
        """Get a human-readable name for the model architecture."""
        parts = []
        
        if self.distributional_dqn:
            parts.append("Distributional")
        if self.dueling_dqn:
            parts.append("Dueling")
        if self.noisy_networks:
            parts.append("Noisy")
        
        parts.append("DQN")
        
        technique_parts = []
        if self.double_dqn:
            technique_parts.append("Double")
        if self.prioritized_replay:
            technique_parts.append("PER")
        if self.n_step_learning:
            technique_parts.append(f"{self.n_steps}-step")
        
        base_name = " ".join(parts)
        if technique_parts:
            return f"{base_name} ({'+'.join(technique_parts)})"
        return base_name
    
    def __str__(self) -> str:
        """String representation showing key architecture details."""
        arch_name = self.get_architecture_name()
        details = []
        
        if self.distributional_dqn:
            details.append(f"{self.n_atoms} atoms")
        if self.noisy_networks:
            details.append(f"σ={self.noisy_std_init}")
        
        if details:
            return f"{arch_name} [{', '.join(details)}]"
        return arch_name