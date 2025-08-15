
# --- Final AsteroidsGame using local asteroids_core classes ---
import pygame
from asteroids_core.constants import SCREEN_WIDTH, SCREEN_HEIGHT
from asteroids_core.player import Player
from asteroids_core.asteroid import Asteroid
from asteroids_core.shot import Shot
from asteroids_core.asteroidfield import AsteroidField

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

    def step(self, action):
        # action: array-like of 5 binary values [thrust, backward, left, right, shoot]
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
        pygame.display.flip()

    def get_state(self):
        return {
            'player_pos': list(self.player.position),
            'asteroids': [list(a.position) for a in self.asteroids],
            'shots': [list(s.position) for s in self.shots],
            'score': self.score
        }

    def is_done(self):
        return self.done
