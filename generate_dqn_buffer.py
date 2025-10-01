import numpy as np
from datetime import datetime
import argparse 
from pathlib import Path

from train_dqn import Agent
from shared.experience import Experience, ExperienceBuffer
from shared.models import AtariDQN, AtariDuelingDQN
from shared.environments import make_atari_env, make_py_asteroids_env, atari_name_id_map, py_asteroids_name_id_map

def main():
    parser = argparse.ArgumentParser(description='Generate DQN experience buffer for training')
    parser.add_argument('game', type=str, help='Game to generate buffer for: py-asteroids, beamrider, or asteroids')
    parser.add_argument('--size', type=int, default=100000, help='Size of the experience buffer')
    parser.add_argument('--clip-reward', action='store_true', help='Clip rewards to [-1, 1] (default: False)')
    args = parser.parse_args()

    game: str = args.game.lower()

    # parameters relevant to initial buffer creation
    dueling_dqn: bool = True # Doesn't matter much for buffer generation
    buffer_size: int = args.size  
    clip_reward: bool = args.clip_reward  # Whether to clip rewards to [-1, 1]
    initial_experience_epsilon: float = 0 # always set to 0 for initial buffer generation

    # Storage format options - use HDF5 for large buffers to avoid memory issues
    use_hdf5 = True  # Set to True for large buffers (>50k), False for small buffers

    print(f"🎮 Generating buffer for game: {game}")
    print(f"📊 Buffer size: {buffer_size}")
    print(f"🤖 Using {'Dueling ' if dueling_dqn else ''}DQN")
    print(f"🎲 Initial epsilon: {initial_experience_epsilon}")

    # Initialize environment (same logic as train_dqn.py)
    if game.startswith('py-asteroids'):
        config_version = py_asteroids_name_id_map.get(game, game) # ex: py-asteroids or py-asteroids-v1
        env = make_py_asteroids_env(action_mode="ale", config_version=config_version, clip_reward=clip_reward) 
    else: 
        env_id = atari_name_id_map.get(game, game)
        try:
            env = make_atari_env(env_id, clip_reward=clip_reward)
        except Exception as e:
            raise ValueError(f"Unsupported game: {game}. Error: {e}")

    if dueling_dqn:
        net = AtariDuelingDQN(env.observation_space.shape, env.action_space.n) # type: ignore
    else:
        net = AtariDQN(env.observation_space.shape, env.action_space.n) # type: ignore

    # Initialize experience buffer and play env to fill it
    buffer = ExperienceBuffer(buffer_size)
    agent = Agent(env, buffer)

    print(f"Filling initial experience buffer with epsilon {initial_experience_epsilon} ...")
    while len(buffer) < buffer_size:
        if len(buffer) % (buffer_size // 5) == 0:
            print(len(buffer))
        agent.play_step(net, epsilon=initial_experience_epsilon, device="cpu")
    agent.reset_env()
    print(len(buffer), "Experience buffer filled")

    current_time = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    buffer_id = f"{game}_{current_time}_{buffer_size}"
    if clip_reward:
        buffer_id += "_clip"

    save_dir = Path("buffers/")
    save_dir.mkdir(parents=True, exist_ok=True)

    if use_hdf5:
        save_path = save_dir / f'{buffer_id}.h5'
        print(f"Saving experience buffer to HDF5: {save_path} ...")
        buffer.save_buffer_to_hdf5(str(save_path))
    else:
        save_path = save_dir / f'buffer_{buffer_id}.npz'
        print(f"Saving experience buffer to NPZ: {save_path} ...")
        buffer.save_buffer_to_npz(str(save_path))
    
    print("Saved.")


if __name__ == "__main__":
    main()


