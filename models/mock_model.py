import numpy as np

class MockModel:
    """
    A mock AI model for Asteroids. Ignores the observation and returns a fixed or random action.
    Output: np.ndarray of shape (5,) with binary values for [thrust, backward, left, right, shoot]
    """
    def __init__(self):
        pass

    def __call__(self, obs):
        # Example: always thrust, shoot, and turn left every frame
        # [thrust, backward, left, right, shoot]
        action = np.zeros(5, dtype=np.int32)
        action[0] = 1  # thrust
        action[2] = 1  # left
        action[4] = 1  # shoot
        return action
