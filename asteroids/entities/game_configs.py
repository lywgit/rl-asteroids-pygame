"""
Game configuration system for py-asteroids versions.
"""
from dataclasses import dataclass
from typing import Dict, Type

@dataclass
class GameConfig:
    """Base configuration for py-asteroids game versions."""
    # Display
    screen_width: int
    screen_height: int
    
    # Asteroids
    asteroid_min_radius: int
    asteroid_kinds: int
    asteroid_spawn_rate: float
    
    # Player
    player_radius: int
    player_speed: int
    player_turn_speed: int
    player_shot_speed: int
    player_shoot_cooldown: float
    
    # Shots
    shot_radius: int
    
    # Rewards & Difficulty
    survival_reward_per_second: float
    level_duration: float
    asteroid_speed_increase_per_level: float
    
    @property
    def asteroid_max_radius(self) -> int:
        return self.asteroid_min_radius * self.asteroid_kinds

@dataclass
class GameConfigV1(GameConfig):
    """Original py-asteroids configuration - v1.0"""
    screen_width: int = 800
    screen_height: int = 600
    asteroid_min_radius: int = 20
    asteroid_kinds: int = 3
    asteroid_spawn_rate: float = 0.8
    player_radius: int = 20
    player_speed: int = 300
    player_turn_speed: int = 300
    player_shot_speed: int = 1000
    player_shoot_cooldown: float = 0.2
    shot_radius: int = 5
    survival_reward_per_second: float = 0.0
    level_duration: float = 30.0
    asteroid_speed_increase_per_level: float = 0.15

# Version registry
GAME_CONFIGS: Dict[str, GameConfig] = {
    'py-asteroids-v1': GameConfigV1(),
}

def get_game_config(version: str = 'py-asteroids-v1') -> GameConfig:
    """Get game configuration for specified version."""
    if version not in GAME_CONFIGS:
        raise ValueError(f"Unknown game version: {version}. Available: {list(GAME_CONFIGS.keys())}")
    return GAME_CONFIGS[version]