import pygame
from .circleshape import CircleShape

class Shot(CircleShape):
    def __init__(self, x:float, y:float, radius:float):
        super().__init__(x, y, radius)

    def draw(self, screen):
        pygame.draw.circle(screen, "white", self.position, self.radius, width=2)

    def update(self, dt:float):
        self.position += self.velocity * dt
