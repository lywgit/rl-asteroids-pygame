"""
Prioritized Experience Replay implementation using Sum Tree data structure.
Based on the paper "Prioritized Experience Replay" by Schaul et al. (2016).
"""
import numpy as np
import torch
from typing import Tuple, List, Any, Optional, Union, Generator, cast
from .experience import Experience, ExperienceBuffer


class SumTree:
    """
    Sum Tree data structure for efficient prioritized sampling.
    
    The tree is stored as a binary tree in array format:
    - Index 0: root (contains total sum)
    - Indices [0, capacity-1): internal nodes (contain sums of subtrees)  
    - Indices [capacity-1, 2*capacity-1): leaf nodes (contain actual priorities)
    
    Example with capacity=4:
    Array: [sum_total, sum_left, sum_right, p0, p1, p2, p3]
    Tree:       0
              /   \\
             1     2
            / \\   / \\
           3   4 5   6
    """
    
    def __init__(self, capacity: int):
        """
        Initialize Sum Tree.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0
    
    def total(self) -> float:
        """Return total sum of all priorities (value at root)."""
        return self.tree[0]
    
    def add(self, priority: float, data: Any) -> None:
        """
        Add new data with given priority.
        
        Args:
            priority: Priority value for the data
            data: Data to store (typically an Experience)
        """
        # Calculate tree index for this data slot
        tree_idx = self.write_idx + self.capacity - 1
        
        # Store the data
        self.data[self.write_idx] = data
        
        # Update priority in tree
        self.update(tree_idx, priority)
        
        # Update write pointer
        self.write_idx = (self.write_idx + 1) % self.capacity
        
        # Track number of entries
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, tree_idx: int, priority: float) -> None:
        """
        Update priority at given tree index and propagate changes upward.
        
        Args:
            tree_idx: Index in the tree array
            priority: New priority value
        """
        # Calculate the change in priority
        change = priority - self.tree[tree_idx]
        
        # Update the leaf node
        self.tree[tree_idx] = priority
        
        # Propagate the change up the tree
        self._propagate(tree_idx, change)
    
    def _propagate(self, tree_idx: int, change: float) -> None:
        """
        Recursively propagate priority change up to the root.
        
        Args:
            tree_idx: Starting tree index
            change: Change in priority to propagate
        """
        # Calculate parent index
        parent_idx = (tree_idx - 1) // 2
        
        # Add change to parent
        self.tree[parent_idx] += change
        
        # Continue propagating if not at root
        if parent_idx != 0:
            self._propagate(parent_idx, change)
    
    def sample(self, value: float) -> Tuple[int, float, Any]:
        """
        Sample data point based on priority value.
        
        Args:
            value: Random value in [0, total()] to determine which data to sample
            
        Returns:
            Tuple of (tree_index, priority, data)
        """
        tree_idx = self._retrieve(0, value)
        data_idx = tree_idx - self.capacity + 1
        
        return tree_idx, self.tree[tree_idx], self.data[data_idx]
    
    def _retrieve(self, tree_idx: int, value: float) -> int:
        """
        Recursively retrieve leaf index corresponding to the given cumulative value.
        
        Args:
            tree_idx: Current tree index (start with 0 for root)
            value: Target cumulative value
            
        Returns:
            Tree index of the selected leaf node
        """
        # Calculate left and right child indices
        left_child_idx = 2 * tree_idx + 1
        right_child_idx = left_child_idx + 1
        
        # If we're at a leaf node, return it
        if left_child_idx >= len(self.tree):
            return tree_idx
        
        # Decide which subtree to search based on cumulative values
        left_sum = self.tree[left_child_idx]
        
        if value <= left_sum:
            # Search left subtree
            return self._retrieve(left_child_idx, value)
        else:
            # Search right subtree (subtract left sum from value)
            return self._retrieve(right_child_idx, value - left_sum)
    
    def get_priority(self, data_idx: int) -> float:
        """
        Get priority for data at given index.
        
        Args:
            data_idx: Data index (0 to capacity-1)
            
        Returns:
            Priority value
        """
        tree_idx = data_idx + self.capacity - 1
        return self.tree[tree_idx]
    
    def __len__(self) -> int:
        """Return number of entries in the tree."""
        return self.n_entries


class PrioritizedExperienceBuffer(ExperienceBuffer):
    """
    Prioritized Experience Replay Buffer using Sum Tree for efficient sampling.
    
    Samples experiences with probability proportional to their TD-error.
    Applies importance sampling correction to maintain unbiased learning.
    """
    
    def __init__(self, 
                 capacity: int, 
                 alpha: float = 0.6, 
                 beta: float = 0.4,
                 beta_increment: float = 0.0001,
                 epsilon: float = 1e-6):
        """
        Initialize prioritized experience buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0=uniform, 1=fully prioritized)
            beta: Importance sampling correction exponent
            beta_increment: Amount to increment beta per sample
            epsilon: Small constant to avoid zero priorities
        """
        # Don't call parent __init__ as we use different storage
        self.capacity = capacity
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.beta_increment = float(beta_increment)
        self.epsilon = float(epsilon)
        
        # Initialize sum tree
        self.tree = SumTree(capacity)
        
        # Track maximum priority seen so far
        self.max_priority: float = 1.0
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.tree)
    
    def append(self, experience: Experience, td_error: Optional[float] = None) -> None:
        """
        Add experience to buffer with priority based on TD-error.
        
        Args:
            experience: Experience tuple to add
            td_error: TD-error for priority calculation (None uses max priority)
        """
        priority = self._get_priority(td_error)
        self.tree.add(priority, experience)
    
    def sample(self, 
               batch_size: int, 
               as_torch_tensor: bool = False, 
               device: str = 'cpu') -> Any:
        """
        Sample batch of experiences with prioritized sampling.
        
        Args:
            batch_size: Number of experiences to sample
            as_torch_tensor: Whether to return tensors instead of numpy arrays
            device: Device for tensors if as_torch_tensor=True
            
        Returns:
            If priority sampling: (experiences, tree_indices, importance_weights)
            If uniform sampling: standard tuple of (obs, actions, rewards, dones, next_obs)
        """
        if len(self.tree) == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Sample experiences using prioritized sampling
        experiences = []
        tree_indices = []
        priorities = []
        
        # Divide priority range into segments for stratified sampling
        segment_size = self.tree.total() / batch_size
        
        for i in range(batch_size):
            # Sample uniformly within each segment
            min_val = segment_size * i
            max_val = segment_size * (i + 1)
            sample_value = np.random.uniform(min_val, max_val)
            
            # Get experience corresponding to this priority value
            tree_idx, priority, experience = self.tree.sample(sample_value)
            
            experiences.append(experience)
            tree_indices.append(tree_idx)
            priorities.append(priority)
        
        # Calculate importance sampling weights
        priorities = np.array(priorities)
        sampling_probabilities = priorities / self.tree.total()
        
        # IS weights = (N * P(i))^(-beta) / max_weight (for normalization)
        weights = np.power(len(self.tree) * sampling_probabilities, -self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Increment beta (anneal towards 1.0)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Return prioritized sampling format
        if as_torch_tensor:
            # Convert experiences to tensor format
            obs, actions, rewards, dones, next_obs = self._experiences_to_tensors(
                experiences, device)
            return (obs, actions, rewards, dones, next_obs), tree_indices, torch.tensor(
                weights, dtype=torch.float32, device=device)
        else:
            return experiences, tree_indices, weights
    
    def update_priorities(self, tree_indices: List[int], td_errors: np.ndarray) -> None:
        """
        Update priorities for experiences based on new TD-errors.
        
        Args:
            tree_indices: Tree indices of experiences to update
            td_errors: New TD-errors for priority calculation
        """
        for tree_idx, td_error in zip(tree_indices, td_errors):
            priority = self._get_priority(td_error)
            self.tree.update(tree_idx, priority)
            
            # Update max priority
            self.max_priority = max(self.max_priority, float(priority))
    
    def _get_priority(self, td_error: Optional[float]) -> float:
        """
        Convert TD-error to priority value.
        
        Args:
            td_error: TD-error magnitude
            
        Returns:
            Priority value (always > 0)
        """
        if td_error is None:
            return self.max_priority
        
        # Ensure td_error is a scalar float
        try:
            # Handle numpy scalars and arrays
            if hasattr(td_error, 'item') and callable(getattr(td_error, 'item')):
                td_error_val = float(td_error.item())  # type: ignore
            else:
                td_error_val = float(td_error)
        except (ValueError, TypeError) as e:
            # Fallback to max_priority if conversion fails
            print(f"Warning: Could not convert TD error {td_error} to float: {e}")
            return self.max_priority
        
        # Priority = (|TD-error| + epsilon)^alpha
        return (abs(td_error_val) + self.epsilon) ** self.alpha
    
    def _experiences_to_tensors(self, experiences: List[Experience], device: str):
        """Convert list of experiences to tensor format."""
        obs = torch.tensor(np.stack([exp.obs for exp in experiences]), 
                          dtype=torch.float32, device=device)
        actions = torch.tensor(np.array([exp.action for exp in experiences]), 
                              dtype=torch.int64, device=device)
        rewards = torch.tensor(np.array([exp.reward for exp in experiences]), 
                              dtype=torch.float32, device=device)
        dones = torch.tensor(np.array([exp.done for exp in experiences]), 
                            dtype=torch.bool, device=device)
        next_obs = torch.tensor(np.stack([exp.next_obs for exp in experiences]), 
                               dtype=torch.float32, device=device)
        
        return obs, actions, rewards, dones, next_obs
    
    # Implement methods for compatibility with base ExperienceBuffer
    def save_buffer_to_npz(self, file_path: str, chunk_size: int = 1000) -> None:
        """Save buffer to NPZ format with chunked processing to avoid memory spikes."""
        if len(self.tree) == 0:
            print("Warning: Cannot save empty buffer")
            return
        
        buffer_size = len(self.tree)
        
        # Pre-allocate arrays to avoid repeated memory allocations
        sample_exp = self.tree.data[0]
        obs_shape = (buffer_size,) + sample_exp.obs.shape
        next_obs_shape = (buffer_size,) + sample_exp.next_obs.shape
        
        obs = np.empty(obs_shape, dtype=sample_exp.obs.dtype)
        actions = np.empty(buffer_size, dtype=type(sample_exp.action))
        rewards = np.empty(buffer_size, dtype=type(sample_exp.reward))
        dones = np.empty(buffer_size, dtype=bool)
        next_obs = np.empty(next_obs_shape, dtype=sample_exp.next_obs.dtype)
        
        # Fill arrays in chunks to reduce peak memory usage
        for start_idx in range(0, buffer_size, chunk_size):
            end_idx = min(start_idx + chunk_size, buffer_size)
            chunk_experiences = [self.tree.data[i] for i in range(start_idx, end_idx)]
            
            # Extract and assign chunk data
            obs[start_idx:end_idx] = np.stack([exp.obs for exp in chunk_experiences])
            actions[start_idx:end_idx] = [exp.action for exp in chunk_experiences]
            rewards[start_idx:end_idx] = [exp.reward for exp in chunk_experiences]
            dones[start_idx:end_idx] = [exp.done for exp in chunk_experiences]
            next_obs[start_idx:end_idx] = np.stack([exp.next_obs for exp in chunk_experiences])
            
            # Clear chunk from memory
            del chunk_experiences
        
        np.savez_compressed(file_path, 
                           obs=obs, action=actions, reward=rewards, 
                           done=dones, next_obs=next_obs)
        print(f"Saved prioritized buffer to {file_path}, size: {buffer_size}")
    
    def load_buffer_from_npz(self, file_path: str) -> None:
        """Load buffer from NPZ format (all experiences get max priority)."""
        with np.load(file_path) as data:
            for exp_data in zip(data['obs'], data['action'], data['reward'], 
                               data['done'], data['next_obs']):
                experience = Experience(*exp_data)
                self.append(experience)  # Uses max priority
        print(f"Loaded buffer from {file_path} into prioritized buffer, size: {len(self.tree)}")
    
    def save_buffer_to_hdf5(self, file_path: str, chunk_size: int = 1000) -> None:
        """Save buffer to HDF5 format with chunked writing to avoid memory spikes."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 support")
        
        if len(self.tree) == 0:
            print("Warning: Cannot save empty buffer")
            return
        
        with h5py.File(file_path, 'w') as f:
            buffer_size = len(self.tree)
            
            # Get sample experience to determine shapes
            sample_exp = self.tree.data[0]
            obs_shape = (buffer_size,) + sample_exp.obs.shape
            next_obs_shape = (buffer_size,) + sample_exp.next_obs.shape
            
            # Create datasets with appropriate shapes and chunking
            obs_ds = f.create_dataset('obs', shape=obs_shape, dtype=np.float32, 
                                    compression='gzip', chunks=True)
            actions_ds = f.create_dataset('action', shape=(buffer_size,), dtype=np.int32,
                                        compression='gzip', chunks=True)
            rewards_ds = f.create_dataset('reward', shape=(buffer_size,), dtype=np.float32,
                                        compression='gzip', chunks=True)
            dones_ds = f.create_dataset('done', shape=(buffer_size,), dtype=bool,
                                      compression='gzip', chunks=True)
            next_obs_ds = f.create_dataset('next_obs', shape=next_obs_shape, dtype=np.float32,
                                         compression='gzip', chunks=True)
            priorities_ds = f.create_dataset('priorities', shape=(buffer_size,), dtype=np.float32,
                                           compression='gzip', chunks=True)
            
            # Write data in chunks to avoid memory spikes
            for start_idx in range(0, buffer_size, chunk_size):
                end_idx = min(start_idx + chunk_size, buffer_size)
                chunk_experiences = [self.tree.data[i] for i in range(start_idx, end_idx)]
                
                # Extract chunk data
                chunk_obs = np.stack([exp.obs for exp in chunk_experiences])
                chunk_actions = np.array([exp.action for exp in chunk_experiences])
                chunk_rewards = np.array([exp.reward for exp in chunk_experiences])
                chunk_dones = np.array([exp.done for exp in chunk_experiences])
                chunk_next_obs = np.stack([exp.next_obs for exp in chunk_experiences])
                chunk_priorities = np.array([self.tree.get_priority(i) for i in range(start_idx, end_idx)])
                
                # Write chunk to HDF5
                obs_ds[start_idx:end_idx] = chunk_obs
                actions_ds[start_idx:end_idx] = chunk_actions
                rewards_ds[start_idx:end_idx] = chunk_rewards
                dones_ds[start_idx:end_idx] = chunk_dones
                next_obs_ds[start_idx:end_idx] = chunk_next_obs
                priorities_ds[start_idx:end_idx] = chunk_priorities
                
                # Clear chunk data from memory
                del chunk_obs, chunk_actions, chunk_rewards, chunk_dones, chunk_next_obs, chunk_priorities
            
            # Save metadata
            f.attrs['buffer_size'] = buffer_size
            f.attrs['alpha'] = self.alpha
            f.attrs['beta'] = self.beta
            f.attrs['max_priority'] = self.max_priority
            
        print(f"Saved prioritized buffer to HDF5: {file_path}, size: {buffer_size}")
    
    def load_buffer_from_hdf5(self, file_path: str, chunk_size: int = 1000) -> None:
        """Load buffer from HDF5 format with chunked reading to avoid memory spikes."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 support")
        
        with h5py.File(file_path, 'r') as f:
            buffer_size = int(f.attrs['buffer_size'])  # type: ignore
            
            # Load data in chunks to avoid memory spikes
            for start_idx in range(0, buffer_size, chunk_size):
                end_idx = min(start_idx + chunk_size, buffer_size)
                
                # Read chunk data from HDF5 and convert to numpy arrays
                chunk_obs = np.array(f['obs'][start_idx:end_idx])  # type: ignore
                chunk_actions = np.array(f['action'][start_idx:end_idx])  # type: ignore
                chunk_rewards = np.array(f['reward'][start_idx:end_idx])  # type: ignore
                chunk_dones = np.array(f['done'][start_idx:end_idx])  # type: ignore
                chunk_next_obs = np.array(f['next_obs'][start_idx:end_idx])  # type: ignore
                
                # Load priorities if available, otherwise use max priority
                if 'priorities' in f:
                    chunk_priorities = np.array(f['priorities'][start_idx:end_idx])  # type: ignore
                else:
                    chunk_priorities = np.full(end_idx - start_idx, self.max_priority)
                
                # Add experiences to tree chunk by chunk
                for i in range(len(chunk_obs)):
                    experience = Experience(chunk_obs[i], chunk_actions[i], chunk_rewards[i], 
                                          chunk_dones[i], chunk_next_obs[i])
                    # Use stored priority
                    self.tree.add(float(chunk_priorities[i]), experience)
                
                # Clear chunk data from memory
                del chunk_obs, chunk_actions, chunk_rewards, chunk_dones, chunk_next_obs, chunk_priorities
            
            # Restore metadata if available
            if 'max_priority' in f.attrs:
                self.max_priority = float(f.attrs['max_priority'])  # type: ignore
            
        print(f"Loaded prioritized buffer from HDF5: {file_path}, size: {len(self.tree)}")