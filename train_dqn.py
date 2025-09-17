"""
DQN training script for Atari and Asteroids environments.
"""
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import collections
import matplotlib.pyplot as plt
import yaml

import gymnasium as gym
import ale_py
from torch.utils.tensorboard import SummaryWriter
import argparse
import torch
import torch.nn as nn

# Import shared components
from shared.models import AtariDQN, AtariDuelingDQN
from shared.environments import make_atari_env, make_py_asteroids_env
from shared.utils import get_device


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config_copy(config: dict, checkpoint_dir: Path) -> None:
    """Save a copy of the configuration to the checkpoint directory"""
    config_path = checkpoint_dir / "config.yaml"
    
    # Preserve the order from get_default_config()
    default_config = get_default_config()
    ordered_config = {}
    
    # First add all keys in the default order
    for key in default_config.keys():
        if key in config:
            ordered_config[key] = config[key]
    
    # Then add any additional keys that might exist in config but not in default
    for key, value in config.items():
        if key not in ordered_config:
            ordered_config[key] = value
    
    with open(config_path, 'w') as f:
        yaml.dump(ordered_config, f, default_flow_style=False, indent=2, sort_keys=False)
    print(f"Configuration saved to: {config_path}")


def get_default_config() -> dict:
    """Get default configuration values"""
    return {
        'game': 'beamrider',
        'max_steps': 100000,
        'replay_buffer_size': 100000,
        'learning_rate': 0.0001,
        'batch_size': 32,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.1,
        'epsilon_decay_frames': 100000,
        'checkpoint_interval': 10000,
        'eval_episode_interval': 5,
        'load_model': None,
        'load_curl_checkpoint': None,  # Path to CURL pretrained encoder checkpoint
        'freeze_curl_encoder': False,  # Whether to freeze CURL encoder during training
        'load_replay_buffer': None,    # Path to pregenerated replay buffer (.h5 or .npz)
        'comment': '',
        'double_dqn': True,
        'dueling_dqn': True
    }
    

# Experience and replay buffer
# Note: use numpy arrays to store experiences (see Agent.play_step)
Experience = collections.namedtuple("Experience", ["obs", "action", "reward", "done", "next_obs"])

class ExperienceBuffer:
    def __init__(self, capacity) -> None:
        self.buffer = collections.deque(maxlen=capacity)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def append(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size:int, as_torch_tensor:bool = False, device='cpu'):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        out = (np.stack(x) for x in zip(*[self.buffer[idx] for idx in indices]))
        if as_torch_tensor:
            out = (torch.tensor(x, device=device) for x in out)
        return out
    
    def get_experience(self, index: int) -> Experience:
        """Get a single experience by index for memory-efficient access"""
        return self.buffer[index]
    
    def save_buffer_to_npz(self, file_path:str):
        obs, action, reward, done, next_obs = (np.stack(x) for x in zip(*list(self.buffer)))
        np.savez_compressed(file_path, obs=obs, action=action, reward=reward, done=done, next_obs=next_obs)
        print(f"Save buffer to {file_path}, size : {len(self.buffer)}")

    def load_buffer_from_npz(self, file_path:str):
        with np.load(file_path) as data:
            for exp in zip(data['obs'], data['action'], data['reward'], data['done'], data['next_obs']):
                self.buffer.append(Experience(*exp))
        print(f"Load buffer from {file_path}, size: {len(self.buffer)}")
    
    def save_buffer_to_hdf5(self, file_path: str):
        """
        Save buffer to HDF5 format for memory-efficient storage of large buffers.
        HDF5 is more memory-efficient than npz for large datasets.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 support. Install with: pip install h5py")
        
        print(f"Saving buffer to HDF5: {file_path}, size: {len(self.buffer)}")
        
        with h5py.File(file_path, 'w') as f:
            # Create datasets with compression
            buffer_size = len(self.buffer)
            
            if buffer_size == 0:
                print("Warning: Buffer is empty, creating minimal HDF5 file")
                return
            
            # Get sample to determine shapes and dtypes
            sample_exp = self.buffer[0]
            obs_shape = sample_exp.obs.shape
            obs_dtype = sample_exp.obs.dtype
            
            # Create datasets with chunking and compression for better performance
            chunk_size = min(1000, buffer_size)  # Reasonable chunk size
            
            obs_dataset = f.create_dataset('obs', 
                                         shape=(buffer_size,) + obs_shape, 
                                         dtype=obs_dtype,
                                         chunks=(chunk_size,) + obs_shape,
                                         compression='gzip', 
                                         compression_opts=6)
            
            action_dataset = f.create_dataset('action', 
                                            shape=(buffer_size,), 
                                            dtype=np.int64,
                                            chunks=(chunk_size,),
                                            compression='gzip',
                                            compression_opts=6)
            
            reward_dataset = f.create_dataset('reward', 
                                            shape=(buffer_size,), 
                                            dtype=np.float32,
                                            chunks=(chunk_size,),
                                            compression='gzip',
                                            compression_opts=6)
            
            done_dataset = f.create_dataset('done', 
                                          shape=(buffer_size,), 
                                          dtype=np.bool_,
                                          chunks=(chunk_size,),
                                          compression='gzip',
                                          compression_opts=6)
            
            next_obs_dataset = f.create_dataset('next_obs', 
                                              shape=(buffer_size,) + obs_shape, 
                                              dtype=obs_dtype,
                                              chunks=(chunk_size,) + obs_shape,
                                              compression='gzip',
                                              compression_opts=6)
            
            # Write data in chunks to avoid memory spikes
            print("Writing data in chunks...")
            for i in range(0, buffer_size, chunk_size):
                end_idx = min(i + chunk_size, buffer_size)
                chunk_experiences = [self.buffer[j] for j in range(i, end_idx)]
                
                # Stack the chunk data
                chunk_obs = np.stack([exp.obs for exp in chunk_experiences])
                chunk_actions = np.array([exp.action for exp in chunk_experiences])
                chunk_rewards = np.array([exp.reward for exp in chunk_experiences])
                chunk_dones = np.array([exp.done for exp in chunk_experiences])
                chunk_next_obs = np.stack([exp.next_obs for exp in chunk_experiences])
                
                # Write chunk to datasets
                obs_dataset[i:end_idx] = chunk_obs
                action_dataset[i:end_idx] = chunk_actions
                reward_dataset[i:end_idx] = chunk_rewards
                done_dataset[i:end_idx] = chunk_dones
                next_obs_dataset[i:end_idx] = chunk_next_obs
                
                if (i + chunk_size) % (chunk_size * 10) == 0:  # Progress every 10 chunks
                    print(f"  Progress: {min(end_idx, buffer_size)}/{buffer_size}")
            
            # Store metadata
            f.attrs['buffer_size'] = buffer_size
            f.attrs['obs_shape'] = obs_shape
            f.attrs['obs_dtype'] = str(obs_dtype)
            
        print(f"✅ Successfully saved buffer to HDF5: {file_path}")

    def load_buffer_from_hdf5(self, file_path: str):
        """
        Load buffer from HDF5 format with memory-efficient streaming.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 support. Install with: pip install h5py")
        
        # print(f"Loading buffer from HDF5: {file_path}")
        
        with h5py.File(file_path, 'r') as f:
            buffer_size = f.attrs['buffer_size']
            print(f"Buffer size in file: {buffer_size}")
            
            # Clear existing buffer
            self.buffer.clear()
            
            # Load data in chunks to avoid memory spikes
            chunk_size = min(1000, buffer_size)
            
            print("Loading data in chunks...")
            for i in range(0, buffer_size, chunk_size):
                end_idx = min(i + chunk_size, buffer_size)
                
                # Read chunk data
                chunk_obs = f['obs'][i:end_idx]
                chunk_actions = f['action'][i:end_idx]
                chunk_rewards = f['reward'][i:end_idx]
                chunk_dones = f['done'][i:end_idx]
                chunk_next_obs = f['next_obs'][i:end_idx]
                
                # Convert to experiences and append
                for j in range(len(chunk_obs)):
                    exp = Experience(
                        obs=chunk_obs[j],
                        action=chunk_actions[j],
                        reward=chunk_rewards[j],
                        done=chunk_dones[j],
                        next_obs=chunk_next_obs[j]
                    )
                    self.buffer.append(exp)
                
                if (i + chunk_size) % (chunk_size * 10) == 0:  # Progress every 10 chunks
                    print(f"  Progress: {min(end_idx, buffer_size)}/{buffer_size}")
        
        print(f"✅ Successfully loaded buffer from HDF5: {file_path}, final size: {len(self.buffer)}")


# Agent: handle interaction with environment and keep experience buffer for replay
class Agent:
    def __init__(self, env:gym.Env, buffer:ExperienceBuffer):
        self.env = env
        self.experience_buffer = buffer
        self.curr_obs = None
        self.episode_reward = 0.0
        self.reset_env()

    def reset_env(self):
        self.episode_reward = 0.0
        self.curr_obs, _ = self.env.reset()

    def play_step(self, net:nn.Module, epsilon:float = 0.0, device:str="cpu", update_buffer=True) -> None | float:
        episode_reward = None # return cumulated episode_reward when episode ends else None
        # choose an action
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                obs = torch.tensor(self.curr_obs, device=device, dtype=torch.float32).unsqueeze(0) # add batch dimension
                q_values = net(obs)
                action = q_values.argmax(dim=1).item()
        
        next_obs, reward, truncated, terminated, _ = self.env.step(action)
        is_done = truncated or terminated
        self.episode_reward += reward # type: ignore

        # update experience buffer
        # limit data type to np.int64, np.float32, np.bool for consistency
        if update_buffer and self.curr_obs is not None:
            exp = Experience(
                np.array(self.curr_obs, dtype=np.float32), 
                np.array(action, dtype=np.int64), 
                np.array(reward, dtype=np.float32), 
                np.array(is_done, dtype=np.bool), 
                np.array(next_obs, dtype=np.float32)
            )
            self.experience_buffer.append(exp)
        self.curr_obs = next_obs
        if is_done:
            episode_reward = self.episode_reward
            self.reset_env()

        return episode_reward


def train(env, config):
    double_dqn = True # experimental
    device = get_device()
    print("Using device:", device)

    game = config['game']
    buffer_size = config['replay_buffer_size']
    max_steps = config['max_steps']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    epsilon_start = config['epsilon_start']
    epsilon_end = config['epsilon_end']
    epsilon_decay_frames = config['epsilon_decay_frames']
    checkpoint_interval = config['checkpoint_interval']
    eval_episode_interval = config['eval_episode_interval']
    double_dqn = config['double_dqn']
    dueling_dqn = config.get('dueling_dqn', False)
    comment = config['comment']
    gamma = config['gamma']

    current_time = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    run_id = f"{current_time}_{game}"
    if comment:
        run_id = f"{run_id}_{comment}"
    print('Run ID:', run_id)

    checkpoint_dir = Path('./checkpoints') / run_id
    log_dir = Path('./runs') / run_id

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the configuration
    save_config_copy(config, checkpoint_dir)

    writer = SummaryWriter(log_dir=log_dir)

    hyperparameters = {
        "game": game,
        "buffer_size": buffer_size,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay_frames": epsilon_decay_frames,
        "gamma": gamma,
        "frames": 0,
        "double_dqn": double_dqn,
        "dueling_dqn": dueling_dqn
        }
    metrics = {
        "best_eval_reward": float("-inf")
    }


    buffer = ExperienceBuffer(capacity=buffer_size)
    agent = Agent(env, buffer) # when agent play step, it updates the buffer by default

    if dueling_dqn:
        print("Using Dueling DQN architecture")
        net = AtariDuelingDQN(env.observation_space.shape, env.action_space.n).to(device)
        tgt_net = AtariDuelingDQN(env.observation_space.shape, env.action_space.n).to(device)
    else:
        net = AtariDQN(env.observation_space.shape, env.action_space.n).to(device)
        tgt_net = AtariDQN(env.observation_space.shape, env.action_space.n).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Load saved model or curl pretrained encoder if specified
    if config.get('load_model'):
        try:
            print(f"Loading model from: {config['load_model']}")
            checkpoint = torch.load(config['load_model'], map_location=device)
            net.load_state_dict(checkpoint)
            tgt_net.load_state_dict(checkpoint)  # Start with same weights for target network
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(f"❌ Failed to load model from: {config['load_model']}")
            print("❌ Training stopped. Please check the model path and try again.")
            return
        
    if config.get('load_curl_checkpoint'):
        try:
            print(f"Loading CURL pretrained encoder from: {config['load_curl_checkpoint']}")
            curl_checkpoint = torch.load(config['load_curl_checkpoint'], map_location=device, weights_only=False)
            curl_state_dict = curl_checkpoint['encoder_state_dict']
            
            # Extract convolutional weights from CURL encoder
            conv_state_dict = {}
            for key, value in curl_state_dict.items():
                if key.startswith('conv.'):
                    # The key already matches DQN model structure (conv.0.weight, conv.2.weight, etc.)
                    conv_state_dict[key] = value
            
            # Load weights into DQN model's convolutional layers
            dqn_state_dict = net.state_dict()
            loaded_layers = []
            missed_layers = []
            
            for key in conv_state_dict:
                if key in dqn_state_dict:
                    if conv_state_dict[key].shape == dqn_state_dict[key].shape:
                        dqn_state_dict[key] = conv_state_dict[key]
                        loaded_layers.append(key)
                    else:
                        missed_layers.append(f"{key} (shape mismatch)")
                else:
                    missed_layers.append(f"{key} (not found)")
            
            # Load the updated state dict into the network
            net.load_state_dict(dqn_state_dict)
            
            # Also load into target network to maintain consistency
            tgt_net.load_state_dict(dqn_state_dict)
            
            # Print loading summary
            total_conv_params = sum(p.numel() for name, p in net.named_parameters() if name.startswith('conv.'))
            loaded_conv_params = sum(conv_state_dict[key].numel() for key in loaded_layers)
            
            print(f"✅ CURL encoder loaded successfully!")
            print(f"   Loaded layers: {len(loaded_layers)}")
            print(f"   Missed layers: {len(missed_layers)}")
            if missed_layers:
                print(f"   Missed: {missed_layers}")
            print(f"   Conv parameters loaded: {loaded_conv_params}/{total_conv_params}")
            print(f"   CURL epoch: {curl_checkpoint.get('epoch', 'unknown')}")
            
            # Optionally freeze encoder layers if specified
            if config.get('freeze_curl_encoder', False):
                frozen_params = 0
                for name, param in net.named_parameters():
                    if name.startswith('conv.'):
                        param.requires_grad = False
                        frozen_params += param.numel()
                
                # Also freeze target network parameters (though they don't have gradients anyway)
                for name, param in tgt_net.named_parameters():
                    if name.startswith('conv.'):
                        param.requires_grad = False
                
                print(f"🔒 Froze {frozen_params} encoder parameters")
            else:
                print("🔓 Encoder layers will be fine-tuned during training")
                
        except Exception as e:
            print(f"❌ Error loading CURL checkpoint: {e}")
            print(f"❌ Failed to load CURL checkpoint from: {config['load_curl_checkpoint']}")
            print("❌ Training stopped. Please check the CURL checkpoint path and try again.")
            return


    # fill (or load) buffer before start training
    if config.get('load_replay_buffer'):
        try:
            buffer_path = config['load_replay_buffer']
            print(f"Loading replay buffer from: {buffer_path}")
            if buffer_path.endswith('.h5') or buffer_path.endswith('.hdf5'):
                buffer.load_buffer_from_hdf5(buffer_path)
            elif buffer_path.endswith('.npz'):
                buffer.load_buffer_from_npz(buffer_path)
            else:
                raise ValueError("Unsupported buffer file format. Use .h5 or .npz")
            print("✅ Replay buffer loaded successfully")
        except Exception as e:
            print(f"❌ Error loading replay buffer: {e}")
            print(f"❌ Failed to load replay buffer from: {config['load_replay_buffer']}")
            print("❌ Training stopped. Please check the buffer path and try again.")
            return
    else:
        initial_experience_epsilon = config['epsilon_start']
        print(f"Filling initial experience buffer with epsilon {initial_experience_epsilon} ...")
        while len(buffer) < buffer_size:
            if len(buffer) % (buffer_size // 5) == 0:
                print(len(buffer))
            agent.play_step(net, epsilon=initial_experience_epsilon, device=device)
        agent.reset_env()
        print("Experience buffer filled", len(buffer))

    # start training
    frame_idx = 0
    episode_idx = 0
    best_eval_reward = float("-inf")
    best_eval_reward_frame_idx = 0
    try:
        while frame_idx < max_steps:
            frame_idx += 1
            epsilon = epsilon_start - (epsilon_start - epsilon_end) * (frame_idx / epsilon_decay_frames)
            epsilon = max(epsilon, epsilon_end)
            # play steps and update buffer until an episode is done
            episode_reward = agent.play_step(net, epsilon=epsilon, device=device)


            # sample from buffer 
            obs, actions, rewards, dones, next_obs = buffer.sample(batch_size=batch_size, as_torch_tensor=True, device=device)
            # train online network
            # 1. target/actual Q(s,a) = reward + gamma * max_a{tgt_net(next_state)}
            if double_dqn:
                next_actions = net(next_obs).argmax(dim=1, keepdim=True)
                Q_target = rewards + gamma * tgt_net(next_obs).gather(1, next_actions).squeeze(-1) * (~dones)
            else:
                Q_target = rewards + gamma * tgt_net(next_obs).max(dim=1)[0] * (~dones) # if done set to 0
            Q_target = Q_target.detach()
            # 2. current/predicted Q(s,a) = net(state)_action
            Q_pred = net(obs).gather(1, actions.unsqueeze(-1)).squeeze(-1)  # type: ignore
            # 3. loss
            loss = nn.MSELoss()(Q_pred, Q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # sync target network periodically
            if frame_idx % 1000 == 0:
                tgt_net.load_state_dict(net.state_dict())
                # print("Target network updated")

            if frame_idx % checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"dqn_{frame_idx}.pth"
                torch.save(net.state_dict(), str(checkpoint_path))

            # per episode operation 
            if episode_reward is not None:
                episode_idx += 1
                # logs
                print(f"frame {frame_idx}, (episode {episode_idx}), reward {episode_reward:.2f}, epsilon {epsilon:.2f}")
                writer.add_scalar("train/epsilon", epsilon, frame_idx)
                writer.add_scalar("train/reward", episode_reward, frame_idx)

                # evaluate deterministic (zero-epsilon) periodically
                if episode_idx % eval_episode_interval == 0:
                    print("Evaluating...")
                    rewards = []
                    for _ in range(5):
                        episode_reward = None
                        while episode_reward is None:
                            episode_reward = agent.play_step(net, epsilon=0, device=device, update_buffer=False)
                        rewards.append(episode_reward)
                    mean_eval_reward = sum(rewards) / len(rewards)
                    writer.add_scalar("eval/reward", mean_eval_reward, frame_idx)
                    print("Evaluation results:", mean_eval_reward, "best:", best_eval_reward)

                    if mean_eval_reward > best_eval_reward:
                        best_eval_reward = max(best_eval_reward, mean_eval_reward)
                        best_eval_reward_frame_idx = frame_idx
                        print("New best evaluation reward:", mean_eval_reward, "checkpoint saved")
                        torch.save(net.state_dict(), checkpoint_dir / f"dqn_best.pth")
                
                
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user. Current frame: {frame_idx}")
    finally:
        metrics["best_eval_reward"] = best_eval_reward
        writer.add_hparams(hyperparameters, metrics, global_step=best_eval_reward_frame_idx)
        writer.close()
        env.close()
        print("Training complete")


def main():
    parser = argparse.ArgumentParser(description='Train DQN agent on Beamrider or (not the standard) Asteroids game')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML configuration file')
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        print(f"✅ Loaded configuration from: {args.config}")
    except FileNotFoundError:
        print(f"❌ Config file not found: {args.config}")
        print("🔧 Creating default config.yaml...")
        config = get_default_config()
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
        print("✅ Default config.yaml created. Please modify it as needed and run again.")
        return
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Validate configuration
    if config['game'] not in ['py-asteroids', 'beamrider', 'asteroids']:
        print(f"❌ Invalid game: {config['game']}. Must be 'py-asteroids', 'beamrider' or 'asteroids'")
        return
    
    # Initialize environment
    if config['game'] == 'py-asteroids':
        env = make_py_asteroids_env(action_mode="combination", clip_reward=True) # "combination" or "single"
    elif config['game'] == 'beamrider':
        env = make_atari_env("ALE/BeamRider-v5", grayscale_obs=True, max_episode_steps=100000)
    elif config['game'] == 'asteroids':
        env = make_atari_env("ALE/Asteroids-v5", grayscale_obs=True, max_episode_steps=100000)
    else:
        raise ValueError(f"Unsupported game: {config['game']}")

    train(env, config)


if __name__ == "__main__":
    main()