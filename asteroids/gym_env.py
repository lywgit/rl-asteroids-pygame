import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .game import AsteroidsGame
import pygame

class AsteroidsEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.game = AsteroidsGame()
        # Observation is the rendered RGB screen
        self.observation_shape = (self.game.height, self.game.width, 3)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)
        # MultiBinary(5): [left, right, thrust, backward, shoot]
        self.action_space = spaces.MultiBinary(5)
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.previous_score = 0  # Track previous score for incremental rewards

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self.previous_score = 0  # Reset score tracking
        obs = self._get_obs()
        if self.render_mode == "human":
            self._init_render()
            self.render()
        return obs, {
            "score": self.game.score,
            "survival_reward": self.game.total_survival_reward,
            "total_reward": self.game.score + self.game.total_survival_reward,
            "level": self.game.current_level,
            "game_time": self.game.game_time
        }

    def step(self, action):
        # action: array-like of 5 binary values [thrust, backward, left, right, shoot]
        if self.game.is_done():
            # If already done, ignore further steps until reset
            obs = self._get_obs()
            reward = 0.0
            terminated = True
            truncated = False
            info = {
                "score": self.game.score,
                "survival_reward": self.game.total_survival_reward,
                "total_reward": self.game.score + self.game.total_survival_reward,
                "level": self.game.current_level,
                "game_time": self.game.game_time
            }
            return obs, reward, terminated, truncated, info

        self.game.step(action)
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self.game.is_done()
        truncated = False
        info = {
            "score": self.game.score,
            "survival_reward": self.game.total_survival_reward,
            "total_reward": self.game.score + self.game.total_survival_reward,
            "level": self.game.current_level,
            "game_time": self.game.game_time
        }
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.screen is None:
                self._init_render()
            self.game.render(self.screen)
            pygame.display.flip()  # Update the display for human mode
            if self.clock is not None:
                self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def _init_render(self):
        pygame.init()
        pygame.font.init()  # Initialize font system for UI text
        if self.render_mode == "human":
            # Set window position to make it more visible
            # import os
            # os.environ['SDL_VIDEODRIVER'] = 'cocoa'  # Force Cocoa driver on macOS
            # os.environ['SDL_VIDEO_WINDOW_POS'] = '100,100'  # Position window at (100,100)
            
            self.screen = pygame.display.set_mode((self.game.width, self.game.height))
            pygame.display.set_caption("🎮 Asteroids")
            self.clock = pygame.time.Clock()
            
            # Try to bring window to front
            pygame.display.flip()
        else:
            # For rgb_array mode, create an off-screen surface (no window)
            self.screen = pygame.Surface((self.game.width, self.game.height))
            self.clock = pygame.time.Clock()

    def _get_obs(self):
        # Render to surface and return the RGB array
        if self.screen is None or not isinstance(self.screen, pygame.Surface):
            self._init_render()
        if self.screen is None or not isinstance(self.screen, pygame.Surface):
            raise RuntimeError("self.screen is not a valid pygame.Surface after initialization!")
        # Draw the game to the screen (don't flip for off-screen rendering)
        self._render_to_surface(self.screen)
        # Get the RGB array (width, height, 3)
        obs = pygame.surfarray.array3d(self.screen)
        # Transpose to (height, width, 3) if needed by your RL framework
        obs = np.transpose(obs, (1, 0, 2))
        return obs

    def _render_to_surface(self, surface):
        """Render game to surface without display updates"""
        surface.fill((0, 0, 0))
        for obj in self.game.drawable:
            obj.draw(surface)
        # Display score, level, and time on screen
        self.game._draw_ui(surface)

    def _get_reward(self):
        """Calculate reward including score and survival bonus"""
        from .entities.constants import SURVIVAL_REWARD_PER_SECOND
        
        # Incremental score reward (change since last step)
        current_score = self.game.score
        score_reward = float(current_score - self.previous_score)
        self.previous_score = current_score
        
        # Survival reward: reward for staying alive longer
        # Give survival reward every second
        survival_reward = 0.0
        if self.game.game_time >= self.game.last_survival_reward_time + 1.0:
            survival_reward = SURVIVAL_REWARD_PER_SECOND
            self.game.last_survival_reward_time = self.game.game_time
            self.game.total_survival_reward += survival_reward  # Update display counter
        
        total_reward = score_reward + survival_reward
        return total_reward
