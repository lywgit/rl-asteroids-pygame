"""
DQN training script for Atari and Asteroids environments.
"""
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import yaml

import gymnasium as gym
import ale_py
from torch.utils.tensorboard import SummaryWriter
import argparse
import torch
import torch.nn as nn

# Import shared components
from shared.models import AtariDQN, AtariDuelingDQN, AtariDistributionalDQN, AtariDistributionalDuelingDQN
from shared.distributional_utils import get_value_range_for_game, categorical_projection, distributional_loss, kl_divergence
from shared.environments import make_atari_env, make_py_asteroids_env, atari_name_id_map
from shared.utils import get_device
from shared.experience import Experience, ExperienceBuffer
from shared.prioritized_experience import PrioritizedExperienceBuffer


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
        'max_steps': 500000,
        'replay_buffer_size': 100000,
        'load_replay_buffer': None,    # Path to pregenerated replay buffer (.h5 or .npz)
        'learning_rate': 0.0001,
        'batch_size': 32,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.1,
        'epsilon_decay_frames': 200000,
        'double_dqn': True,
        'dueling_dqn': True,
        'checkpoint_interval': 10000,
        'eval_episode_interval': 10,
        'load_model': None,
        'comment': '',
        # Prioritized Experience Replay settings
        'prioritized_replay': False,   # Enable prioritized experience replay
        'priority_alpha': 0.6,         # Priority exponent (0=uniform, 1=full priority)
        'priority_beta': 0.4,          # Importance sampling correction exponent
        'priority_beta_increment': 0.0001,  # Beta annealing rate per sample
        'priority_epsilon': 1e-6,      # Small constant to avoid zero priorities
        # Multi-step learning settings
        'n_step_learning': False,      # Enable n-step learning
        'n_steps': 3,                  # Number of steps for n-step returns
        # Distributional Q-learning settings (C51)
        'distributional_dqn': False,   # Enable distributional Q-learning
        'n_atoms': 51,                 # Number of atoms for categorical distribution
        'v_min': None,                 # Minimum value for support (auto-estimated if None)
        'v_max': None                  # Maximum value for support (auto-estimated if None)
    }


def load_model(config: dict, net: nn.Module, tgt_net: nn.Module, device: str) -> bool:
    """
    Load a pretrained model checkpoint into the networks.
    
    Args:
        config: Configuration dictionary
        net: Main network to load weights into
        tgt_net: Target network to load weights into
        device: Device to load the model on
        
    Returns:
        bool: True if successful, False if failed (should stop training)
    """
    if not config.get('load_model'):
        return True
        
    try:
        print(f"Loading model from: {config['load_model']}")
        checkpoint = torch.load(config['load_model'], map_location=device)
        net.load_state_dict(checkpoint)
        tgt_net.load_state_dict(checkpoint)  # Start with same weights for target network
        print("✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"❌ Failed to load model from: {config['load_model']}")
        print("❌ Training stopped. Please check the model path and try again.")
        return False


def load_replay_buffer(config: dict, buffer: 'ExperienceBuffer | PrioritizedExperienceBuffer', agent: 'Agent', net: nn.Module, 
                      buffer_size: int, device: str) -> bool:
    """
    Load or generate the replay buffer for training.
    
    Args:
        config: Configuration dictionary
        buffer: Experience buffer to fill
        agent: Agent for generating experiences
        net: Network for generating experiences (if needed)
        buffer_size: Size of the buffer to fill
        device: Device to run experience generation on
        
    Returns:
        bool: True if successful, False if failed (should stop training)
    """
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
            return True
        except Exception as e:
            print(f"❌ Error loading replay buffer: {e}")
            print(f"❌ Failed to load replay buffer from: {config['load_replay_buffer']}")
            print("❌ Training stopped. Please check the buffer path and try again.")
            return False
    else:
        initial_experience_epsilon = config['epsilon_start']
        print(f"Filling initial experience buffer with epsilon {initial_experience_epsilon} ...")
        while len(buffer) < buffer_size:
            if len(buffer) % (buffer_size // 5) == 0:
                print(len(buffer))
            agent.play_step(net, epsilon=initial_experience_epsilon, device=device)
        agent.reset_env()
        print("Experience buffer filled", len(buffer))
        return True


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
                
                # Handle both standard and distributional networks
                if hasattr(net, 'get_q_values'):
                    # Distributional network - get expected Q-values
                    q_values = net.get_q_values(obs)
                else:
                    # Standard network - direct Q-values
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
    log_dir = Path('./runs') / game /run_id

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the configuration
    save_config_copy(config, checkpoint_dir)

    writer = SummaryWriter(log_dir=log_dir)

    # Create appropriate buffer type based on configuration
    prioritized_replay = config.get('prioritized_replay', False)
    
    # N-step learning configuration
    n_step_learning = config.get('n_step_learning', False)
    n_steps = config.get('n_steps', 1)
    
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
        "dueling_dqn": dueling_dqn,
        "prioritized_replay": prioritized_replay,
        "priority_alpha": config.get('priority_alpha', 0.6) if prioritized_replay else 0.0,
        "priority_beta": config.get('priority_beta', 0.4) if prioritized_replay else 0.0,
        "priority_beta_increment": config.get('priority_beta_increment', 0.0001) if prioritized_replay else 0.0,
        "priority_epsilon": config.get('priority_epsilon', 1e-6) if prioritized_replay else 0.0,
        "n_step_learning": n_step_learning,
        "n_steps": n_steps
        }
    metrics = {
        "best_eval_reward": float("-inf")
    }
    # Initialize priority parameters (used even if prioritized_replay is False for logging)
    priority_alpha = float(config.get('priority_alpha', 0.6))
    priority_beta = float(config.get('priority_beta', 0.4))
    priority_beta_increment = float(config.get('priority_beta_increment', 0.0001))
    priority_epsilon = float(config.get('priority_epsilon', 1e-6))
    
    if prioritized_replay:
        print("Using Prioritized Experience Replay")
        buffer = PrioritizedExperienceBuffer(
            capacity=buffer_size,
            alpha=priority_alpha,
            beta=priority_beta,
            beta_increment=priority_beta_increment,
            epsilon=priority_epsilon
        )
    else:
        buffer = ExperienceBuffer(capacity=buffer_size)
    
    if n_step_learning:
        print(f"Using {n_steps}-step learning")
    else:
        print("Using standard 1-step learning")
    
    agent = Agent(env, buffer) # when agent play step, it updates the buffer by default

    # Distributional Q-learning configuration
    distributional_dqn = config.get('distributional_dqn', False)
    if distributional_dqn:
        n_atoms = config.get('n_atoms', 51)
        v_min = config.get('v_min', None) 
        v_max = config.get('v_max', None)
        
        # Auto-estimate value range if not provided
        if v_min is None or v_max is None:
            estimated_v_min, estimated_v_max = get_value_range_for_game(game, buffer)
            v_min = v_min if v_min is not None else estimated_v_min
            v_max = v_max if v_max is not None else estimated_v_max
        
        print(f"Using Distributional DQN (C51) with {n_atoms} atoms, support: [{v_min:.1f}, {v_max:.1f}]")
        
        if dueling_dqn:
            print("Using Distributional Dueling DQN architecture")
            net = AtariDistributionalDuelingDQN(env.observation_space.shape, env.action_space.n, 
                                               n_atoms=n_atoms, v_min=v_min, v_max=v_max).to(device)
            tgt_net = AtariDistributionalDuelingDQN(env.observation_space.shape, env.action_space.n,
                                                   n_atoms=n_atoms, v_min=v_min, v_max=v_max).to(device)
        else:
            print("Using Distributional DQN architecture")
            net = AtariDistributionalDQN(env.observation_space.shape, env.action_space.n,
                                        n_atoms=n_atoms, v_min=v_min, v_max=v_max).to(device)
            tgt_net = AtariDistributionalDQN(env.observation_space.shape, env.action_space.n,
                                            n_atoms=n_atoms, v_min=v_min, v_max=v_max).to(device)
    else:
        if dueling_dqn:
            print("Using Dueling DQN architecture")
            net = AtariDuelingDQN(env.observation_space.shape, env.action_space.n).to(device)
            tgt_net = AtariDuelingDQN(env.observation_space.shape, env.action_space.n).to(device)
        else:
            net = AtariDQN(env.observation_space.shape, env.action_space.n).to(device)
            tgt_net = AtariDQN(env.observation_space.shape, env.action_space.n).to(device)
            
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Update hyperparameters with distributional settings
    if distributional_dqn:
        hyperparameters.update({
            "distributional_dqn": True,
            "n_atoms": config.get('n_atoms', 51),
            "v_min": config.get('v_min', -10.0),
            "v_max": config.get('v_max', 10.0)
        })
    else:
        hyperparameters.update({
            "distributional_dqn": False,
            "n_atoms": 0,
            "v_min": 0.0,
            "v_max": 0.0
        })

    # Load saved model to continue training
    if not load_model(config, net, tgt_net, device):
        return
    
    # Load replay buffer or generate initial experiences
    if not load_replay_buffer(config, buffer, agent, net, buffer_size, device):
        return

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


            # Sample from buffer - handle both standard and prioritized replay with optional n-step
            if prioritized_replay:
                if n_step_learning:
                    # Prioritized n-step sampling
                    (obs, actions, rewards, dones, next_obs), tree_indices, importance_weights = buffer.sample_n_step(
                        batch_size=batch_size, n_steps=n_steps, gamma=gamma, as_torch_tensor=True, device=device)
                else:
                    # Standard prioritized sampling
                    (obs, actions, rewards, dones, next_obs), tree_indices, importance_weights = buffer.sample(
                        batch_size=batch_size, as_torch_tensor=True, device=device)
            else:
                if n_step_learning:
                    # Standard n-step sampling
                    obs, actions, rewards, dones, next_obs = buffer.sample_n_step(
                        batch_size=batch_size, n_steps=n_steps, gamma=gamma, as_torch_tensor=True, device=device)
                else:
                    # Standard uniform sampling
                    obs, actions, rewards, dones, next_obs = buffer.sample(
                        batch_size=batch_size, as_torch_tensor=True, device=device)
                importance_weights = torch.ones(batch_size, device=device)  # Uniform weights
                tree_indices = None  # Not needed for standard replay
            
            # Train online network
            # For n-step learning, use gamma^n_steps since rewards already contain n-step discounted sum
            discount_factor = gamma ** n_steps if n_step_learning else gamma
            
            if distributional_dqn:
                # === DISTRIBUTIONAL Q-LEARNING (C51) ===
                
                # 1. Get current state-action distributions
                current_dist = net(obs)  # [batch_size, n_actions, n_atoms]
                # Select distributions for taken actions
                current_action_dist = current_dist.gather(1, actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, net.n_atoms)).squeeze(1)
                
                # 2. Compute target distributions
                with torch.no_grad():
                    next_dist = tgt_net(next_obs)  # [batch_size, n_actions, n_atoms]
                    
                    if double_dqn:
                        # Use online network for action selection
                        if hasattr(net, 'get_q_values'):
                            next_q_values = net.get_q_values(next_obs)
                        else:
                            next_q_values = (net(next_obs) * net.support).sum(dim=2)
                        next_actions = next_q_values.argmax(dim=1)
                    else:
                        # Use target network for action selection  
                        if hasattr(tgt_net, 'get_q_values'):
                            next_q_values = tgt_net.get_q_values(next_obs)
                        else:
                            next_q_values = (next_dist * tgt_net.support).sum(dim=2)
                        next_actions = next_q_values.argmax(dim=1)
                    
                    # Select next action distributions
                    next_action_dist = next_dist.gather(1, next_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, tgt_net.n_atoms)).squeeze(1)
                    
                    # Categorical projection: project T_z onto support
                    target_dist = categorical_projection(
                        next_action_dist, rewards, dones, discount_factor,
                        tgt_net.support, tgt_net.n_atoms, tgt_net.v_min, tgt_net.v_max
                    )
                
                # 3. Compute distributional loss
                if prioritized_replay:
                    # === PRIORITIZED EXPERIENCE REPLAY ===
                    # Use KL divergence for priority updates (better for distributional)
                    priority_errors = kl_divergence(current_action_dist, target_dist, reduction='none').detach()
                    
                    # Cross-entropy loss with importance sampling
                    elementwise_loss = distributional_loss(current_action_dist, target_dist, reduction='none')
                    weighted_loss = elementwise_loss * importance_weights
                    loss = weighted_loss.mean()
                    
                    # Optimize network
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update experience priorities based on KL divergence
                    if tree_indices is not None:
                        if hasattr(tree_indices, 'tolist'):
                            tree_indices_list = tree_indices.tolist()
                        else:
                            tree_indices_list = list(tree_indices)
                        buffer.update_priorities(tree_indices_list, priority_errors.cpu().numpy())
                        
                else:
                    # === STANDARD EXPERIENCE REPLAY ===
                    loss = distributional_loss(current_action_dist, target_dist)
                    
                    # Optimize network
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
            else:
                # === STANDARD Q-LEARNING ===
                
                # 1. Compute target Q-values
                if double_dqn:
                    next_actions = net(next_obs).argmax(dim=1, keepdim=True)
                    Q_target = rewards + discount_factor * tgt_net(next_obs).gather(1, next_actions).squeeze(-1) * (~dones)
                else:
                    Q_target = rewards + discount_factor * tgt_net(next_obs).max(dim=1)[0] * (~dones)
                Q_target = Q_target.detach()
                
                # 2. Current Q-values for taken actions
                Q_pred = net(obs).gather(1, actions.unsqueeze(-1)).squeeze(-1)
                
                # 3. MSE loss calculation
                if prioritized_replay:
                    # === PRIORITIZED EXPERIENCE REPLAY ===
                    td_errors = torch.abs(Q_pred - Q_target).detach()
                    
                    elementwise_loss = nn.MSELoss(reduction='none')(Q_pred, Q_target)
                    weighted_loss = elementwise_loss * importance_weights
                    loss = weighted_loss.mean()
                    
                    # Optimize network
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update experience priorities based on TD errors
                    if tree_indices is not None:
                        if hasattr(tree_indices, 'tolist'):
                            tree_indices_list = tree_indices.tolist()
                        else:
                            tree_indices_list = list(tree_indices)
                        buffer.update_priorities(tree_indices_list, td_errors.cpu().numpy())
                        
                else:
                    # === STANDARD EXPERIENCE REPLAY ===
                    loss = nn.MSELoss()(Q_pred, Q_target)
                    
                    # Optimize network
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
                print(f"Frame {frame_idx}, (episode {episode_idx}): train reward {episode_reward:.2f}, epsilon {epsilon:.2f}")
                writer.add_scalar("train/epsilon", epsilon, frame_idx)
                writer.add_scalar("train/reward", episode_reward, frame_idx)
                
                # Log prioritized replay metrics
                if prioritized_replay:
                    writer.add_scalar("prioritized_replay/beta", buffer.beta, frame_idx)  # type: ignore
                    writer.add_scalar("prioritized_replay/max_priority", buffer.max_priority, frame_idx)  # type: ignore

                # evaluate deterministic (zero-epsilon) periodically
                if episode_idx % eval_episode_interval == 0:
                    print("Evaluating...")
                    rewards = []
                    for _ in range(10): # average over 10 episodes
                        episode_reward = None
                        while episode_reward is None:
                            episode_reward = agent.play_step(net, epsilon=0, device=device, update_buffer=False)
                        rewards.append(episode_reward)
                    mean_eval_reward = sum(rewards) / len(rewards)
                    writer.add_scalar("eval/reward", mean_eval_reward, frame_idx)
                    print(f"Frame {frame_idx}, (episode {episode_idx}): Mean eval reward {mean_eval_reward} (best: {best_eval_reward})")

                    if mean_eval_reward > best_eval_reward:
                        best_eval_reward = max(best_eval_reward, mean_eval_reward)
                        best_eval_reward_frame_idx = frame_idx
                        print("New best evaluation reward:", mean_eval_reward, "update dqn_best checkpoint")
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
       
    # Initialize environment
    game = str(config['game'].lower())
    if game == 'py-asteroids':
        env = make_py_asteroids_env(action_mode="combination") # "combination" or "single"
    else: 
        env_id = atari_name_id_map.get(game, game)
        try:
            env = make_atari_env(env_id)
        except Exception as e:
            raise ValueError(f"Unsupported game: {game}. Error: {e}")

    train(env, config)


if __name__ == "__main__":
    main()