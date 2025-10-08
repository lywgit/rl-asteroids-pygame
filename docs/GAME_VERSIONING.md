# Game Versioning System Documentation

## Overview

The py-asteroids game now supports versioning to allow precise control over game difficulty and parameters for research purposes. This enables fair comparison between different experiments and easy reproduction of results.

## Current Versions

### py-asteroids-v1 (Default)
- **Player speed**: 300
- **Asteroid spawn rate**: 0.8 seconds
- **Level duration**: 30 seconds
- **Difficulty scaling**: 15% speed increase per level
- **Status**: Original baseline configuration

## Adding New Versions

### Step 1: Define Configuration Class

```python
# In asteroids/entities/game_configs.py
@dataclass
class GameConfigV2(GameConfig):
    """Harder version - faster gameplay"""
    player_speed: int = 400  # Faster player (was 300)
    asteroid_spawn_rate: float = 0.6  # More frequent spawning (was 0.8)
    level_duration: float = 25.0  # Shorter levels (was 30.0)
    asteroid_speed_increase_per_level: float = 0.20  # Faster scaling (was 0.15)
    # ... other parameters same as base class defaults
```

### Step 2: Register Version

```python
# In asteroids/entities/game_configs.py
GAME_CONFIGS['py-asteroids-v2'] = GameConfigV2()
```

## Usage Examples

### Human Play
```bash
# Play original version
python play_py_asteroids_human.py --version py-asteroids-v1

# Play harder version (when v2 is implemented)
python play_py_asteroids_human.py --version py-asteroids-v2
```

### Training/Environment Creation
```python
# Default version
env = make_py_asteroids_env(action_mode="ale")

# Specific version
env = make_py_asteroids_env(action_mode="ale", config_version="py-asteroids-v1")

# Direct environment creation
env = AsteroidsEnv(render_mode="human", config_version="py-asteroids-v1")
```

### Checking Available Versions
```python
from asteroids.entities.game_configs import GAME_CONFIGS
print("Available versions:", list(GAME_CONFIGS.keys()))
```

## Configuration Parameters

### Display Settings
- `screen_width`: Game window width (default: 800)
- `screen_height`: Game window height (default: 600)

### Asteroid Settings
- `asteroid_min_radius`: Minimum asteroid size (default: 20)
- `asteroid_kinds`: Number of asteroid size variants (default: 3)
- `asteroid_spawn_rate`: Time between spawns in seconds (default: 0.8)
- `asteroid_max_radius`: Calculated as `min_radius * kinds`

### Player Settings
- `player_radius`: Player ship size (default: 20)
- `player_speed`: Movement speed (default: 300)
- `player_turn_speed`: Rotation speed (default: 300)
- `player_shot_speed`: Bullet velocity (default: 1000)
- `player_shoot_cooldown`: Time between shots (default: 0.2)

### Game Settings
- `shot_radius`: Bullet size (default: 5)
- `survival_reward_per_second`: Reward for staying alive (default: 0.0)
- `level_duration`: Seconds per difficulty level (default: 30.0)
- `asteroid_speed_increase_per_level`: Speed multiplier per level (default: 0.15)

## Research Benefits

### Experiment Reproducibility
```python
# Always specify version in research
env = make_py_asteroids_env(
    action_mode="ale", 
    config_version="py-asteroids-v1"
)
```

### Version Comparison
```python
# Easy to compare different difficulty levels
results_v1 = train_agent(config_version="py-asteroids-v1")
results_v2 = train_agent(config_version="py-asteroids-v2")
```

### Configuration Tracking
```python
# Get config details for logging
config = get_game_config("py-asteroids-v1")
print(f"Training with player_speed={config.player_speed}")
```

## Implementation Details

### Clean Architecture
- **No Global Constants**: Each entity receives only the config parameters it needs
- **Property Extraction**: Config values are extracted as instance properties for performance
- **Dependency Injection**: Config flows down through constructors
- **Backward Compatibility**: All existing code continues to work

### Performance Optimized
```python
class Player:
    def __init__(self, x, y, config):
        # Extract as properties for fast access
        self.player_speed = config.player_speed
        self.screen_width = config.screen_width
        # Keep config reference only for creating child objects
        self._config = config
    
    def move(self, dt):
        # Direct property access (fast)
        self.position += forward * self.player_speed * dt
```

## Future Versions

### Suggested Variations
- **py-asteroids-v2**: Harder difficulty (faster, more frequent)
- **py-asteroids-survival**: Focus on survival rewards
- **py-asteroids-dense**: More asteroids, smaller screen
- **py-asteroids-precision**: Slower but more precise gameplay

### Research Applications
- **Curriculum Learning**: Start with v1, progress to harder versions
- **Difficulty Analysis**: Compare agent performance across versions
- **Transfer Learning**: Train on one version, test on others
- **Environment Robustness**: Ensure agents work across variations

## Migration Notes

### From Old Constants System
- ✅ **No code changes needed**: All existing imports work
- ✅ **Gradual migration**: Can update files one by one if desired
- ✅ **Backward compatible**: `constants.py` backed up but no longer used

### Version Specification
- **Default behavior**: Uses `py-asteroids-v1` if no version specified
- **Explicit specification**: Always specify version in research code
- **Error handling**: Clear error messages for invalid versions