import pygame
import random
from .asteroid import Asteroid
from .game_configs import GameConfig

class AsteroidField(pygame.sprite.Sprite):
    def __init__(self, config: GameConfig, *groups):
        super().__init__(*groups)
        
        # Extract config parameters as instance properties
        self.screen_width = config.screen_width
        self.screen_height = config.screen_height
        self.asteroid_min_radius = config.asteroid_min_radius
        self.asteroid_max_radius = config.asteroid_max_radius
        self.asteroid_kinds = config.asteroid_kinds
        self.asteroid_spawn_rate = config.asteroid_spawn_rate
        
        # Keep reference to config for creating child objects
        self._config = config
        
        self.spawn_timer = 0.0
        self.speed_multiplier = 1.0  # Difficulty scaling multiplier
        
        # Define spawn edges using config values
        self.edges = [
            [
                pygame.Vector2(1, 0),
                lambda y: pygame.Vector2(-self.asteroid_max_radius, y * self.screen_height),
            ],
            [
                pygame.Vector2(-1, 0),
                lambda y: pygame.Vector2(
                    self.screen_width + self.asteroid_max_radius, y * self.screen_height
                ),
            ],
            [
                pygame.Vector2(0, 1),
                lambda x: pygame.Vector2(x * self.screen_width, -self.asteroid_max_radius),
            ],
            [
                pygame.Vector2(0, -1),
                lambda x: pygame.Vector2(
                    x * self.screen_width, self.screen_height + self.asteroid_max_radius
                ),
            ],
        ]

    def spawn(self, radius, position, velocity):
        asteroid = Asteroid(position.x, position.y, radius, self._config)
        asteroid.velocity = velocity

    def update(self, dt):
        self.spawn_timer += dt
        if self.spawn_timer > self.asteroid_spawn_rate:
            self.spawn_timer = 0
            edge = random.choice(self.edges)
            speed = random.randint(40, 100) * self.speed_multiplier  # Apply difficulty scaling
            velocity = edge[0] * speed
            velocity = velocity.rotate(random.randint(-30, 30))
            position = edge[1](random.uniform(0, 1))
            kind = random.randint(1, self.asteroid_kinds)
            self.spawn(self.asteroid_min_radius * kind, position, velocity)
