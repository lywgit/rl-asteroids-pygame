import time
import gymnasium as gym
from game.asteroids_env import AsteroidsEnv
from models.torch_model import TorchMockModel, RandomTorchMockModel

def play_game_with_torch_model(env, model, render=True, max_steps=1000):
    obs, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    
    print(f"Starting game with {model.__class__.__name__}")
    
    while not done and step_count < max_steps:
        action = model(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if render:
            env.render()
            time.sleep(0.016)
        
        step_count += 1
        
        # Print progress every 100 steps
        if step_count % 100 == 0:
            print(f"Step {step_count}, Score: {info['score']}, Total Reward: {total_reward}")
    
    print(f"Game finished after {step_count} steps. Final score: {info['score']}, Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    # Test with the neural network model
    print("Testing with TorchMockModel (neural network)...")
    env = AsteroidsEnv(render_mode="human")
    model = TorchMockModel()
    play_game_with_torch_model(env, model, render=True, max_steps=500)
    
    time.sleep(2)  # Brief pause between games
    
    # Test with the random model
    print("\nTesting with RandomTorchMockModel...")
    env = AsteroidsEnv(render_mode="human")
    random_model = RandomTorchMockModel()
    play_game_with_torch_model(env, random_model, render=True, max_steps=500)
