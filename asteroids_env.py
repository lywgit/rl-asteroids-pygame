import gymnasium as gym
from gymnasium import spaces
import numpy as np
from asteroids_game import AsteroidsGame
import pygame

class AsteroidsEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.game = AsteroidsGame()
        # Example: ship x, y; up to 5 asteroids (x, y each); score
        obs_low = np.array([0, 0] + [0, 0]*5 + [0], dtype=np.float32)
        obs_high = np.array([self.game.width, self.game.height]*1 + [self.game.width, self.game.height]*5 + [np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        # MultiBinary(5): [left, right, thrust, shoot, backward]
        self.action_space = spaces.MultiBinary(5)
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        obs = self._get_obs()
        if self.render_mode == "human":
            self._init_render()
            self.render()
        return obs, {}

    def step(self, action):
        # action: array-like of 5 binary values [left, right, thrust, shoot, backward]
        self.game.step(action)
        obs = self._get_obs()
        reward = self._get_reward()
        done = self.game.is_done()
        info = {}
        if self.render_mode == "human":
            self.render()
        return obs, reward, done, False, info

    def render(self):
        if self.render_mode == "human":
            if self.screen is None:
                self._init_render()
            self.game.render(self.screen)
            if self.clock is not None:
                self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def _init_render(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.game.width, self.game.height))
        pygame.display.set_caption("Asteroids AI (Gymnasium)")
        self.clock = pygame.time.Clock()

    def _get_obs(self):
        state = self.game.get_state()
        # Pad asteroids to 5 for fixed obs size
        asteroids = state['asteroids'][:5] + [[0,0]]*(5-len(state['asteroids']))
        obs = np.array(list(state['player_pos']) + [coord for ast in asteroids for coord in ast] + [state['score']], dtype=np.float32)
        return obs

    def _get_reward(self):
        # Placeholder: reward = score
        return float(self.game.score)
