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
