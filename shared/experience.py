"""
Experience replay buffer for reinforcement learning.
Contains Experience namedtuple and ExperienceBuffer class for storing and sampling experiences.
"""
import collections
import numpy as np
import torch
from typing import NamedTuple


# Experience and replay buffer
# Note: use numpy arrays to store experiences (see Agent.play_step)
class Experience(NamedTuple):
    obs: np.ndarray
    action: np.ndarray  # int64 scalar array
    reward: np.ndarray  # float32 scalar array
    done: np.ndarray    # bool scalar array
    next_obs: np.ndarray


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
            chunk_size = min(1000, buffer_size)  # type: ignore  # Reasonable chunk size
            
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
            for i in range(0, buffer_size, chunk_size):  # type: ignore
                end_idx = min(i + chunk_size, buffer_size)  # type: ignore
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
                    print(f"  Progress: {min(end_idx, buffer_size)}/{buffer_size}")  # type: ignore
            
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
            chunk_size = min(1000, buffer_size)  # type: ignore
            
            print("Loading data in chunks...")
            for i in range(0, buffer_size, chunk_size):  # type: ignore
                end_idx = min(i + chunk_size, buffer_size)  # type: ignore
                
                # Read chunk data
                chunk_obs = f['obs'][i:end_idx]  # type: ignore
                chunk_actions = f['action'][i:end_idx]  # type: ignore
                chunk_rewards = f['reward'][i:end_idx]  # type: ignore
                chunk_dones = f['done'][i:end_idx]  # type: ignore
                chunk_next_obs = f['next_obs'][i:end_idx]  # type: ignore
                
                # Convert to experiences and append
                for j in range(len(chunk_obs)):  # type: ignore
                    exp = Experience(
                        obs=chunk_obs[j],  # type: ignore
                        action=chunk_actions[j],  # type: ignore
                        reward=chunk_rewards[j],  # type: ignore
                        done=chunk_dones[j],  # type: ignore
                        next_obs=chunk_next_obs[j]  # type: ignore
                    )
                    self.buffer.append(exp)
                
                if (i + chunk_size) % (chunk_size * 10) == 0:  # Progress every 10 chunks
                    print(f"  Progress: {min(end_idx, buffer_size)}/{buffer_size}")  # type: ignore
        
        print(f"✅ Successfully loaded buffer from HDF5: {file_path}, final size: {len(self.buffer)}")