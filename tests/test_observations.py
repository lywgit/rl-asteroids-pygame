import numpy as np
from asteroids.gym_env import AsteroidsEnv

def test_observation_format():
    env = AsteroidsEnv(render_mode=None)
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    assert obs.dtype == np.uint8, "Observation dtype should be uint8 (RGB image)"
    assert len(obs.shape) == 3, "Observation should be a 3D array (H, W, 3)"
    assert obs.shape[2] == 3, "Last dimension should be 3 (RGB)"
    print("Observation format test passed.")

if __name__ == "__main__":
    test_observation_format()
