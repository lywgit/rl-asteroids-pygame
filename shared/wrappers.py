"""
Gymnasium environment wrappers for action space conversion and observation preprocessing.
"""

import numpy as np
import gymnasium as gym
from collections import deque


class MultiBinaryToSingleDiscreteAction(gym.ActionWrapper):
    """Convert MultiBinary action space to Single discrete action for DQN compatibility"""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(env.action_space.n) # type: ignore

    def action(self, action: int):
        # Convert discrete action to MultiBinary
        multi_binary_action = np.zeros(self.action_space.n, dtype=np.int32) # type: ignore
        multi_binary_action[action] = 1
        return multi_binary_action


class MultiBinaryToDiscreteCombinationWrapper(gym.ActionWrapper):
    """Convert MultiBinary action space to Discrete for DQN compatibility"""
    def __init__(self, env):
        super().__init__(env)
        # For n binary actions, we have 2^n possible combinations
        self.n_combination = 2 ** env.action_space.n # type: ignore
        self.action_space = gym.spaces.Discrete(self.n_combination)

    def action(self, action):
        # Convert discrete action to MultiBinary
        binary_action = np.zeros(self.n_combination, dtype=np.int32)
        for i in range(self.n_combination):
            binary_action[i] = (action >> i) & 1
        return binary_action


class ScaleObservation(gym.ObservationWrapper):
    """Scale pixel values to [0, 1]"""
    def __init__(self, env: gym.Env):
        assert isinstance(env.observation_space, gym.spaces.Box)
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return np.asarray(observation, dtype=np.float32) / 255.0

class AleActionsWrapper(gym.ActionWrapper):
    """Convert py-asteroids MultiBinary action space to ALE-compatible 18 discrete actions.
    
    Maps the standard 18 ALE actions to meaningful combinations for py-asteroids:
    [thrust, backward, left, right, shoot]
    
    ALE Actions:
    0: NOOP, 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN,
    6: UPRIGHT, 7: UPLEFT, 8: DOWNRIGHT, 9: DOWNLEFT,
    10: UPFIRE, 11: RIGHTFIRE, 12: LEFTFIRE, 13: DOWNFIRE,
    14: UPRIGHTFIRE, 15: UPLEFTFIRE, 16: DOWNRIGHTFIRE, 17: DOWNLEFTFIRE
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(18)
        
        # Define action mapping from ALE discrete actions to py-asteroids MultiBinary
        # py-asteroids actions: [thrust, backward, left, right, shoot]
        self.action_map = {
            0:  [0, 0, 0, 0, 0],  # NOOP
            1:  [0, 0, 0, 0, 1],  # FIRE
            2:  [1, 0, 0, 0, 0],  # UP (thrust forward)
            3:  [0, 0, 0, 1, 0],  # RIGHT (rotate right)
            4:  [0, 0, 1, 0, 0],  # LEFT (rotate left)
            5:  [0, 1, 0, 0, 0],  # DOWN (thrust backward)
            6:  [1, 0, 0, 1, 0],  # UPRIGHT (thrust + rotate right)
            7:  [1, 0, 1, 0, 0],  # UPLEFT (thrust + rotate left)
            8:  [0, 1, 0, 1, 0],  # DOWNRIGHT (backward + rotate right)
            9:  [0, 1, 1, 0, 0],  # DOWNLEFT (backward + rotate left)
            10: [1, 0, 0, 0, 1],  # UPFIRE (thrust + shoot)
            11: [0, 0, 0, 1, 1],  # RIGHTFIRE (rotate right + shoot)
            12: [0, 0, 1, 0, 1],  # LEFTFIRE (rotate left + shoot)
            13: [0, 1, 0, 0, 1],  # DOWNFIRE (backward + shoot)
            14: [1, 0, 0, 1, 1],  # UPRIGHTFIRE (thrust + rotate right + shoot)
            15: [1, 0, 1, 0, 1],  # UPLEFTFIRE (thrust + rotate left + shoot)
            16: [0, 1, 0, 1, 1],  # DOWNRIGHTFIRE (backward + rotate right + shoot)
            17: [0, 1, 1, 0, 1],  # DOWNLEFTFIRE (backward + rotate left + shoot)
        }

    def action(self, action: int):
        """Convert ALE discrete action to py-asteroids MultiBinary action."""
        if action not in self.action_map:
            raise ValueError(f"Invalid action {action}. Must be in range [0, 17]")
        return np.array(self.action_map[action], dtype=np.int32)


class MaxRender(gym.Wrapper):
    def __init__(self, env, capacity=2):
        super().__init__(env)
        self._frame_queue = deque(maxlen=capacity)

    def reset(self, **kwargs):
        self._frame_queue.clear()
        obs, info = super().reset(**kwargs)
        frame = self.env.render()
        if frame is not None:
            self._frame_queue.append(frame)
        return obs, info

    def render(self):
        frame = self.env.render()
        if frame is None:
            return None

        self._frame_queue.append(frame)

        if len(self._frame_queue) == 1:
            return self._frame_queue[0]

        # pixel-wise max over all frames in the queue
        max_frame = np.maximum.reduce(list(self._frame_queue))
        return max_frame
