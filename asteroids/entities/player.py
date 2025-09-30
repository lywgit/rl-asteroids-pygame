import pygame
from .circleshape import CircleShape
from .shot import Shot
from .game_configs import GameConfig

class Player(CircleShape):
    def __init__(self, x: float, y: float, config: GameConfig):
        super().__init__(x, y, config.player_radius)
        
        # Extract config parameters as instance properties
        self.player_speed = config.player_speed
        self.player_turn_speed = config.player_turn_speed
        self.player_shot_speed = config.player_shot_speed
        self.player_shoot_cooldown = config.player_shoot_cooldown
        self.shot_radius = config.shot_radius
        self.screen_width = config.screen_width
        self.screen_height = config.screen_height
        
        # Keep reference to config for creating child objects
        self._config = config
        
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
        self.rotation += self.player_turn_speed * dt

    def move(self, dt:float):
        forward = pygame.math.Vector2(0,1).rotate(self.rotation)
        self.position += forward * self.player_speed * dt
        
        # Wrap around screen edges
        if self.position.x < 0:
            self.position.x = self.screen_width
        elif self.position.x > self.screen_width:
            self.position.x = 0
            
        if self.position.y < 0:
            self.position.y = self.screen_height
        elif self.position.y > self.screen_height:
            self.position.y = 0

    def update(self, dt:float):
        self.shoot_timer = max(0, self.shoot_timer-dt)
        # Keyboard input will be handled outside for agent/human play

    def shoot(self):
        if self.shoot_timer > 0:
            return
        shot = Shot(self.position[0], self.position[1], self.shot_radius, self._config)
        shot.velocity = self.player_shot_speed * pygame.math.Vector2(0,1).rotate(self.rotation)
        self.shoot_timer = self.player_shoot_cooldown
