import random
import pygame
from .circleshape import CircleShape
from .constants import ASTEROID_MIN_RADIUS

class Asteroid(CircleShape):
    def __init__(self, x:float, y:float, radius:float):
        super().__init__(x, y, radius)

    def draw(self, screen):
        pygame.draw.circle(screen, "white", self.position, self.radius, width=2)

    def update(self, dt):
        self.position += self.velocity * dt

    def split(self):
        self.kill()
        if self.radius <= ASTEROID_MIN_RADIUS:
            return
        rot_angle = random.uniform(20, 50)
        left_velocity = self.velocity.rotate(-rot_angle) * 1.2
        right_velocity = self.velocity.rotate(rot_angle) * 1.2
        new_radius = self.radius - ASTEROID_MIN_RADIUS
        left_asteroid = Asteroid(self.position[0], self.position[1], new_radius)
        left_asteroid.velocity = left_velocity
        right_asteroid = Asteroid(self.position[0], self.position[1], new_radius)
        right_asteroid.velocity = right_velocity
