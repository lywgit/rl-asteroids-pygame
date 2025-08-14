import pygame

class CircleShape(pygame.sprite.Sprite):
    _groups = ()

    @classmethod
    def set_groups(cls, *groups):
        cls._groups = groups

    def __init__(self, x:float, y:float, radius:float):
        super().__init__(*self._groups)
        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(0, 0)
        self.radius = radius

    def draw(self, screen):
        pass

    def update(self, dt):
        pass

    def collides_with(self, other):
        distance = self.position.distance_to(other.position)
        radius_sum = self.radius + other.radius
        return distance <= radius_sum
