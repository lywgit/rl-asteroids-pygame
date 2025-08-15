import pygame
from .circleshape import CircleShape
from .shot import Shot
from .constants import PLAYER_RADIUS, PLAYER_TURN_SPEED, PLAYER_SPEED, SHOT_RADIUS, PLAYER_SHOT_SPEED, PLAYER_SHOOT_COOLDOWN

class Player(CircleShape):
    def __init__(self, x: float, y: float):
        super().__init__(x, y, PLAYER_RADIUS)
        self.rotation = 0.0
        self.shoot_timer = 0.0

    def triangle(self):
        forward = pygame.Vector2(0, 1).rotate(self.rotation)
        right = pygame.Vector2(0, 1).rotate(self.rotation + 90) * self.radius / 1.5
        a = self.position + forward * self.radius
        b = self.position - forward * self.radius - right
        c = self.position - forward * self.radius + right
        return [a, b, c]

    def draw(self, screen):
        pygame.draw.polygon(screen, "white", self.triangle(), 2)

    def rotate(self, dt:float):
        self.rotation += PLAYER_TURN_SPEED * dt

    def move(self, dt:float):
        forward = pygame.math.Vector2(0,1).rotate(self.rotation)
        self.position += forward * PLAYER_SPEED * dt

    def update(self, dt:float):
        self.shoot_timer = max(0, self.shoot_timer-dt)
        # Keyboard input will be handled outside for agent/human play

    def shoot(self):
        if self.shoot_timer > 0:
            return
        shot = Shot(self.position[0], self.position[1], SHOT_RADIUS)
        shot.velocity = PLAYER_SHOT_SPEED * pygame.math.Vector2(0,1).rotate(self.rotation)
        self.shoot_timer = PLAYER_SHOOT_COOLDOWN
