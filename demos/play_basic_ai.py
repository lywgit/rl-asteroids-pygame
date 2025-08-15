import time
import gymnasium as gym
from asteroids.gym_env import AsteroidsEnv
from models.mock_model import MockModel

def play_game_with_model(env, model, render=True, max_steps=1000):
    obs, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    while not done and step_count < max_steps:
        action = model(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if render:
            env.render()
            # Removed time.sleep(0.016) - gym_env handles timing with clock.tick(60)
        step_count += 1
    print(f"Game finished. Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    env = AsteroidsEnv(render_mode="human")
    model = MockModel()
    play_game_with_model(env, model, render=True)
