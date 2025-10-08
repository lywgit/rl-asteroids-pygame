# ALE Actions Wrapper Implementation

## Overview

This document describes the implementation of the `AleActionsWrapper` that maps the standard 18 ALE (Arcade Learning Environment) actions to meaningful combinations for the custom py-asteroids game.

## Problem Statement

The custom py-asteroids game uses a `MultiBinary(5)` action space with actions `[thrust, backward, left, right, shoot]`, while standard Atari games in ALE use 18 discrete actions. This difference made it difficult to:

1. Compare performance with standard Atari RL benchmarks
2. Use existing pre-trained models
3. Apply research findings from ALE environments

## Solution

The `AleActionsWrapper` provides a standardized interface by mapping all 18 ALE actions to meaningful combinations of py-asteroids actions.

### ALE Actions Mapping

| ALE ID | Action Name | py-asteroids Mapping | Description |
|--------|-------------|---------------------|-------------|
| 0 | NOOP | `[0,0,0,0,0]` | No operation |
| 1 | FIRE | `[0,0,0,0,1]` | Shoot only |
| 2 | UP | `[1,0,0,0,0]` | Thrust forward |
| 3 | RIGHT | `[0,0,0,1,0]` | Rotate right |
| 4 | LEFT | `[0,0,1,0,0]` | Rotate left |
| 5 | DOWN | `[0,1,0,0,0]` | Thrust backward |
| 6 | UPRIGHT | `[1,0,0,1,0]` | Thrust + rotate right |
| 7 | UPLEFT | `[1,0,1,0,0]` | Thrust + rotate left |
| 8 | DOWNRIGHT | `[0,1,0,1,0]` | Backward + rotate right |
| 9 | DOWNLEFT | `[0,1,1,0,0]` | Backward + rotate left |
| 10 | UPFIRE | `[1,0,0,0,1]` | Thrust + shoot |
| 11 | RIGHTFIRE | `[0,0,0,1,1]` | Rotate right + shoot |
| 12 | LEFTFIRE | `[0,0,1,0,1]` | Rotate left + shoot |
| 13 | DOWNFIRE | `[0,1,0,0,1]` | Backward + shoot |
| 14 | UPRIGHTFIRE | `[1,0,0,1,1]` | Thrust + rotate right + shoot |
| 15 | UPLEFTFIRE | `[1,0,1,0,1]` | Thrust + rotate left + shoot |
| 16 | DOWNRIGHTFIRE | `[0,1,0,1,1]` | Backward + rotate right + shoot |
| 17 | DOWNLEFTFIRE | `[0,1,1,0,1]` | Backward + rotate left + shoot |

## Usage

### Environment Creation

```python
from shared.environments import make_py_asteroids_env

# Create environment with ALE-compatible actions
env = make_py_asteroids_env(action_mode="ale")
print(f"Action space: {env.action_space}")  # Discrete(18)
```

### Training

```python
# In train_dqn.py or similar training scripts
env = make_py_asteroids_env(action_mode="ale")  # Instead of "combination"

# The rest of the training code remains the same
# The model will now work with 18 discrete actions
net = AtariDQN(env.observation_space.shape, env.action_space.n)  # n=18
```

### Comparison of Action Modes

| Mode | Action Space | Total Actions | Use Case |
|------|-------------|---------------|----------|
| `"single"` | `Discrete(5)` | 5 | Simple single-action control |
| `"combination"` | `Discrete(32)` | 32 | All possible binary combinations |
| `"ale"` | `Discrete(18)` | 18 | **ALE-compatible standard actions** |

## Implementation Details

### Class Structure

```python
class AleActionsWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(18)
        self.action_map = { ... }  # 18 action mappings
    
    def action(self, action: int):
        return np.array(self.action_map[action], dtype=np.int32)
```

### Integration Points

1. **shared/wrappers.py**: Contains the `AleActionsWrapper` implementation
2. **shared/environments.py**: Updated `make_py_asteroids_env()` to support `action_mode="ale"`
3. **Training scripts**: Updated comments to include the new action mode option


## Benefits

1. **Standard Compatibility**: Direct compatibility with ALE action space
2. **Research Integration**: Easy comparison with existing Atari RL research
3. **Meaningful Actions**: All 18 actions map to sensible game behaviors
4. **No Redundancy**: Unlike "combination" mode, no meaningless action combinations
5. **Optimal Coverage**: Covers all important movement and shooting combinations


## References

- [ALE Documentation](https://ale.farama.org/environments/)
- [Gymnasium ActionWrapper Documentation](https://gymnasium.farama.org/api/wrappers/action_wrappers/)