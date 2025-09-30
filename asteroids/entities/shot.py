import pygame
from .circleshape import CircleShape
from .game_configs import GameConfig

class Shot(CircleShape):
    def __init__(self, x:float, y:float, radius:float, config: GameConfig):
        super().__init__(x, y, radius)
        
        # Extract config parameters as instance properties
        self.screen_width = config.screen_width
        self.screen_height = config.screen_height

    def draw(self, screen):
        pygame.draw.circle(screen, "white", self.position, self.radius, width=2)

    def update(self, dt:float):
        self.position += self.velocity * dt
        
        # Remove shot when it goes off-screen
        if (self.position.x < -self.radius or self.position.x > self.screen_width + self.radius or
            self.position.y < -self.radius or self.position.y > self.screen_height + self.radius):
            self.kill()
