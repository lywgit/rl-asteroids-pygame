# --- Final AsteroidsGame using local entities classes ---
import pygame
from .entities.constants import SCREEN_WIDTH, SCREEN_HEIGHT, SURVIVAL_REWARD_PER_SECOND, LEVEL_DURATION, ASTEROID_SPEED_INCREASE_PER_LEVEL
from .entities.player import Player
from .entities.asteroid import Asteroid
from .entities.shot import Shot
from .entities.asteroidfield import AsteroidField

class AsteroidsGame:
    def __init__(self, width=SCREEN_WIDTH, height=SCREEN_HEIGHT):
        self.width = width
        self.height = height
        self._dt = 0.016  # default timestep (60 FPS)
        self.reset()

    def reset(self):
        self.updatable = pygame.sprite.Group()
        self.drawable  = pygame.sprite.Group()
        self.asteroids = pygame.sprite.Group()
        self.shots     = pygame.sprite.Group()

        Asteroid.set_groups(self.asteroids, self.updatable, self.drawable)
        Player.set_groups(self.updatable, self.drawable)
        Shot.set_groups(self.shots, self.updatable, self.drawable)

        self.player = Player(self.width//2, self.height//2)
        self.asteroid_field = AsteroidField(self.updatable)

        self.score = 0
        self.done = False
        self.game_time = 0.0  # Track total game time for survival reward and difficulty
        self.current_level = 1
        self.last_survival_reward_time = 0.0
        self.total_survival_reward = 0.0  # Track cumulative survival reward for display

    def step(self, action):
        # action: array-like of 5 binary values [thrust, backward, left, right, shoot]
        
        # Update game time
        self.game_time += self._dt
        
        # Check for level progression and difficulty increase
        new_level = int(self.game_time // LEVEL_DURATION) + 1
        if new_level > self.current_level:
            self.current_level = new_level
            self._increase_difficulty()
        
        # thrust (forward)
        if action[0]:
            self.player.move(self._dt)
        # backward
        if action[1]:
            self.player.move(-self._dt)
        # left
        if action[2]:
            self.player.rotate(-self._dt)
        # right
        if action[3]:
            self.player.rotate(self._dt)
        # shoot
        if action[4]:
            self.player.shoot()

        self.updatable.update(self._dt)

        for asteroid in list(self.asteroids):
            for shot in list(self.shots):
                if asteroid.collides_with(shot):
                    asteroid.split()
                    shot.kill()
                    self.score += 1

        for asteroid in list(self.asteroids):
            if asteroid.collides_with(self.player):
                self.done = True

    def render(self, screen):
        screen.fill((0, 0, 0))
        for obj in self.drawable:
            obj.draw(screen)
        
        # Display score, level, and time on screen
        self._draw_ui(screen)
        
        pygame.display.flip()
    
    def _draw_ui(self, screen):
        """Draw UI elements like score, level, and time"""
        font = pygame.font.Font(None, 36)
        
        # Score
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))
        
        # Survival reward
        survival_text = font.render(f"Survival: {self.total_survival_reward:.1f}", True, (255, 255, 255))
        screen.blit(survival_text, (10, 50))
        
        # Level
        level_text = font.render(f"Level: {self.current_level}", True, (255, 255, 255))
        screen.blit(level_text, (10, 90))
        
        # Time
        minutes = int(self.game_time // 60)
        seconds = int(self.game_time % 60)
        time_text = font.render(f"Time: {minutes:02d}:{seconds:02d}", True, (255, 255, 255))
        screen.blit(time_text, (10, 130))
    
    def _increase_difficulty(self):
        """Increase game difficulty by making asteroids faster"""
        speed_multiplier = 1.0 + (ASTEROID_SPEED_INCREASE_PER_LEVEL * (self.current_level - 1))
        
        # Apply speed increase to existing asteroids
        for asteroid in self.asteroids:
            asteroid.velocity *= (1.0 + ASTEROID_SPEED_INCREASE_PER_LEVEL)
        
        # Update asteroid field to spawn faster asteroids
        self.asteroid_field.speed_multiplier = speed_multiplier

    def get_state(self):
        return {
            'player_pos': list(self.player.position),
            'asteroids': [list(a.position) for a in self.asteroids],
            'shots': [list(s.position) for s in self.shots],
            'score': self.score,
            'level': self.current_level,
            'game_time': self.game_time
        }

    def is_done(self):
        return self.done
