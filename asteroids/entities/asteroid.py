import random
import pygame
from .circleshape import CircleShape
from .game_configs import GameConfig

class Asteroid(CircleShape):
    def __init__(self, x:float, y:float, radius:float, config: GameConfig):
        super().__init__(x, y, radius)
        
        # Extract config parameters as instance properties
        self.asteroid_min_radius = config.asteroid_min_radius
        self.screen_width = config.screen_width
        self.screen_height = config.screen_height
        
        # Keep reference to config for creating child objects (splitting)
        self._config = config

    def draw(self, screen):
        pygame.draw.circle(screen, "white", self.position, self.radius, width=2)

    def update(self, dt):
        self.position += self.velocity * dt
        
        # Wrap around screen edges
        if self.position.x < -self.radius:
            self.position.x = self.screen_width + self.radius
        elif self.position.x > self.screen_width + self.radius:
            self.position.x = -self.radius
            
        if self.position.y < -self.radius:
            self.position.y = self.screen_height + self.radius
        elif self.position.y > self.screen_height + self.radius:
            self.position.y = -self.radius

    def split(self):
        self.kill()
        if self.radius <= self.asteroid_min_radius:
            return
        rot_angle = random.uniform(20, 50)
        left_velocity = self.velocity.rotate(-rot_angle) * 1.2
        right_velocity = self.velocity.rotate(rot_angle) * 1.2
        new_radius = self.radius - self.asteroid_min_radius
        left_asteroid = Asteroid(self.position[0], self.position[1], new_radius, self._config)
        left_asteroid.velocity = left_velocity
        right_asteroid = Asteroid(self.position[0], self.position[1], new_radius, self._config)
        right_asteroid.velocity = right_velocity
