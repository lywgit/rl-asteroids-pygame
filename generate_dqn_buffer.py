from train_dqn import Agent, Experience, ExperienceBuffer
from shared.models import AtariDQN, AtariDuelingDQN
from shared.environments import make_atari_env, make_py_asteroids_env
import numpy as np
from datetime import datetime
import time 

# parameters relevant to initial buffer creation
game = "asteroids" # py-asteroids,  beamrider, asteroids
dueling_dqn = True
py_asteroids_action_mode = "combination" # "single" or "combination"
buffer_size = 100000 # 100000 
initial_experience_epsilon = 0

# Storage format options - use HDF5 for large buffers to avoid memory issues
use_hdf5 = True  # Set to True for large buffers (>50k), False for small buffers

# Initialize environment
if game == 'py-asteroids':
    env = make_py_asteroids_env(action_mode="combination", clip_reward=True) # "combination" or "single"
elif game == 'beamrider':
    env = make_atari_env("ALE/BeamRider-v5", grayscale_obs=True, max_episode_steps=100000)
elif game == 'asteroids':
    env = make_atari_env("ALE/Asteroids-v5", grayscale_obs=True, max_episode_steps=100000)
else:
    raise ValueError(f"Unsupported game: {game}")

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
buffer_id = f"{current_time}_{game}_{buffer_size}"

if use_hdf5:
    save_path = f'buffer_{buffer_id}.h5'
    print(f"Saving experience buffer to HDF5: {save_path} ...")
    buffer.save_buffer_to_hdf5(save_path)
    print("Saved to HDF5 format")
    
    # try loading data back (may OOM)  
    print("Loading experience buffer back from HDF5...")
    loaded_buffer = ExperienceBuffer(capacity=buffer_size)
    loaded_buffer.load_buffer_from_hdf5(save_path)
    print(f"loaded size = {len(loaded_buffer)}")
else:
    save_path = f'buffer_{buffer_id}.npz'
    print(f"Saving experience buffer to NPZ: {save_path} ...")
    buffer.save_buffer_to_npz(save_path)
    print("Saved to NPZ format")
    
    # try loading data back (may OOM)
    print("Loading experience buffer back from NPZ...")
    loaded_buffer = ExperienceBuffer(capacity=buffer_size)
    loaded_buffer.load_buffer_from_npz(save_path)
    print(f"loaded size = {len(loaded_buffer)}")

print(f"   📁 File saved as: {save_path}")


