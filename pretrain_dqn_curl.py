"""
CURL (Contrastive Unsupervised Representations for Reinforcement Learning) pretraining
for DQN encoder networks.

This script pretrains the convolutional encoder using contrastive learning on
experience buffer data before DQN training.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import random

from shared.models import AtariDQN, AtariDuelingDQN
from shared.utils import get_device
from shared.experience import ExperienceBuffer


def load_curl_pretrain_weight(config: dict, net: nn.Module, tgt_net: nn.Module, device: str) -> bool:
    """
    Load CURL pretrained encoder weights into the DQN networks.
    
    Args:
        config: Configuration dictionary
        net: Main network to load encoder weights into
        tgt_net: Target network to load encoder weights into
        device: Device to load the weights on
        
    Returns:
        bool: True if successful, False if failed (should stop training)
    """
    if not config.get('load_curl_checkpoint'):
        return True
        
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
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading CURL checkpoint: {e}")
        print(f"❌ Failed to load CURL checkpoint from: {config['load_curl_checkpoint']}")
        print("❌ Training stopped. Please check the CURL checkpoint path and try again.")
        return False


class CURLEncoder(nn.Module):
    """CURL encoder that wraps the convolutional part of DQN networks"""
    def __init__(self, input_shape, dueling_dqn=False):
        super().__init__()
        self.dueling_dqn = dueling_dqn
        
        # Create the base network to extract conv layers
        if dueling_dqn:
            base_net = AtariDuelingDQN(input_shape, n_action=1)  # n_action doesn't matter here
        else:
            base_net = AtariDQN(input_shape, n_action=1)
        
        # Extract only the convolutional layers
        self.conv = base_net.conv
        self.conv_out_size = base_net._get_conv_out(input_shape)
        
        # Add projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, x):
        """Forward pass through conv layers and projection head"""
        conv_features = self.conv(x).view(x.size(0), -1)
        projections = self.projection_head(conv_features)
        return conv_features, projections
    
    def get_conv_features(self, x):
        """Get only convolutional features (for transfer to DQN)"""
        return self.conv(x).view(x.size(0), -1)


class RandomShiftsAug(nn.Module):
    """Random shift data augmentation for CURL"""
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad,
                               device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2),
                             device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class CURLLoss(nn.Module):
    """CURL contrastive loss using InfoNCE"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_anchor, z_positive):
        """
        Compute CURL loss
        z_anchor: [batch_size, feature_dim] - anchor projections
        z_positive: [batch_size, feature_dim] - positive (consecutive frame) projections
        """
        batch_size = z_anchor.size(0)
        
        # Normalize features
        z_anchor = F.normalize(z_anchor, dim=1)
        z_positive = F.normalize(z_positive, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(z_anchor, z_positive.T) / self.temperature
        
        # Labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=z_anchor.device)
        
        # InfoNCE loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

def sample_temporal_pairs(buffer, batch_size, device='cpu'):
    """
    Sample temporal positive pairs from the experience buffer.
    
    Args:
        buffer: Experience buffer with sequential data
        batch_size: Number of pairs to sample
        device: Device to load tensors on
    
    Returns:
        tuple: (anchor_obs, positive_obs) where positive_obs are consecutive frames
    """
    # Cache valid indices to avoid recomputing every time
    if not hasattr(buffer, '_valid_temporal_indices'):
        print("Computing valid temporal pairs...")
        valid_indices = []
        buffer_size = len(buffer)
        
        # Efficiently check done flags by directly accessing buffer
        for i in range(buffer_size - 1):  # -1 because we need i+1 to exist
            # Access done flag directly from buffer
            _, _, _, done, _ = buffer.buffer[i]
            if not done:  # If done[i] is False, then i+1 is a consecutive frame
                valid_indices.append(i)
        
        buffer._valid_temporal_indices = np.array(valid_indices)
        print(f"Found {len(valid_indices)} valid temporal pairs out of {buffer_size} experiences ({len(valid_indices)/buffer_size*100:.1f}%)")
    
    valid_indices = buffer._valid_temporal_indices
    
    if len(valid_indices) < batch_size:
        raise ValueError(f"Not enough valid temporal pairs. Found {len(valid_indices)}, need {batch_size}")
    
    # Randomly sample valid anchor indices
    sampled_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
    
    # Get anchor observations and consecutive positive observations
    anchor_obs = []
    positive_obs = []
    
    for idx in sampled_indices:
        # Get anchor observation (current frame)
        anchor_exp = buffer.buffer[idx]
        anchor_obs.append(anchor_exp.obs)
        
        # Get positive observation (next frame)
        positive_exp = buffer.buffer[idx + 1]
        positive_obs.append(positive_exp.obs)
    
    # Convert to tensors more efficiently
    anchor_obs = torch.from_numpy(np.stack(anchor_obs)).to(device)
    positive_obs = torch.from_numpy(np.stack(positive_obs)).to(device)
    
    # Convert to float and normalize if needed
    if anchor_obs.dtype == torch.uint8:
        anchor_obs = anchor_obs.float() / 255.0
        positive_obs = positive_obs.float() / 255.0
    
    return anchor_obs, positive_obs


def sample_augmentation_pairs(buffer, batch_size, device='cpu', aug_transform=None):
    """
    Sample pairs where positives are augmented versions of the same frame.
    This is the original CURL approach.
    """
    # Sample random observations
    obs_batch, _, _, _, _ = buffer.sample(batch_size, as_torch_tensor=True, device=device)
    
    if obs_batch.dtype == torch.uint8:
        obs_batch = obs_batch.float() / 255.0
    
    # Create augmented positive pairs
    if aug_transform is None:
        aug_transform = RandomShiftsAug(pad=4)
    
    anchor_obs = obs_batch
    positive_obs = aug_transform(obs_batch.clone())
    
    return anchor_obs, positive_obs


def sample_temporal_augmented_pairs(buffer, batch_size, device='cpu', augment_prob=0.3, aug_transform=None):
    """
    Sample temporal pairs and optionally apply augmentation for stronger positives.
    
    Args:
        buffer: Experience buffer with sequential data
        batch_size: Number of pairs to sample
        device: Device to load tensors on
        augment_prob: Probability of applying augmentation to positive pairs
        aug_transform: Augmentation transform to apply
    
    Returns:
        tuple: (anchor_obs, positive_obs) where positives are consecutive frames + optional augmentation
    """
    # Get temporal pairs first
    anchor_obs, positive_obs = sample_temporal_pairs(buffer, batch_size, device)
    
    # Apply augmentation to some positive pairs
    if augment_prob > 0:
        if aug_transform is None:
            aug_transform = RandomShiftsAug(pad=4)
        
        # Randomly decide which samples to augment
        augment_mask = torch.rand(batch_size, device=device) < augment_prob
        
        if augment_mask.any():
            # Apply augmentation to selected positive samples
            positive_obs[augment_mask] = aug_transform(positive_obs[augment_mask])
    
    return anchor_obs, positive_obs


def sample_mixed_pairs(buffer, batch_size, device='cpu', temporal_ratio=0.7, aug_transform=None):
    """
    Mix temporal and augmentation-based positive pairs in one batch.
    
    Args:
        buffer: Experience buffer
        batch_size: Total batch size
        device: Device to load tensors on
        temporal_ratio: Fraction of batch to use temporal pairs (rest uses augmentation)
        aug_transform: Augmentation transform to apply
    """
    temporal_size = int(batch_size * temporal_ratio)
    aug_size = batch_size - temporal_size
    
    pairs = []
    
    # Get temporal pairs
    if temporal_size > 0:
        temporal_anchor, temporal_positive = sample_temporal_pairs(buffer, temporal_size, device)
        pairs.append((temporal_anchor, temporal_positive))
    
    # Get augmentation pairs
    if aug_size > 0:
        aug_anchor, aug_positive = sample_augmentation_pairs(buffer, aug_size, device, aug_transform)
        pairs.append((aug_anchor, aug_positive))
    
    # Concatenate all pairs
    if len(pairs) == 2:
        anchor_obs = torch.cat([pairs[0][0], pairs[1][0]], dim=0)
        positive_obs = torch.cat([pairs[0][1], pairs[1][1]], dim=0)
    else:
        anchor_obs, positive_obs = pairs[0]
    
    # Shuffle the batch to mix temporal and augmentation pairs
    perm = torch.randperm(batch_size, device=device)
    anchor_obs = anchor_obs[perm]
    positive_obs = positive_obs[perm]
    
    return anchor_obs, positive_obs


def get_default_curl_config():
    """Get default CURL pretraining configuration"""
    return {
        'buffer_path': 'buffer_2025-09-15-111752_asteroids_10000.npz',
        'dueling_dqn': True,
        'batch_size': 256,
        'learning_rate': 0.0001,
        'temperature': 0.1,
        'num_epochs': 100,
        'save_interval': 10,
        'log_interval': 10,
        'device': 'auto',
        'comment': 'curl_temporal_pretrain',
        
        # Augmentation settings
        'augmentation_strategy': 'temporal',  # 'temporal', 'augmentation', 'mixed', 'temporal_aug'
        'temporal_ratio': 0.7,  # For mixed strategy: fraction of temporal pairs
        'augment_prob': 0.3,    # For temporal_aug: probability of augmenting temporal pairs
        'aug_pad': 4,           # Padding for random shifts
    }


def evaluate_representations(encoder, buffer, device, num_samples=1000):
    """
    Evaluate the quality of learned representations using various metrics
    """
    encoder.eval()
    
    # Use cached valid indices if available
    if hasattr(buffer, '_valid_temporal_indices'):
        valid_indices = buffer._valid_temporal_indices
        eval_samples = min(num_samples, len(valid_indices))
        
        # Sample temporal pairs for evaluation
        obs_batch, next_obs_batch = sample_temporal_pairs(buffer, eval_samples, device=device)
    else:
        # Fallback to random sampling if temporal indices not computed yet
        eval_samples = min(num_samples, len(buffer))
        obs_batch, _, _, done_batch, next_obs_batch = buffer.sample(
            eval_samples, as_torch_tensor=True, device=device
        )
        # Convert to float if needed
        if obs_batch.dtype == torch.uint8:
            obs_batch = obs_batch.float() / 255.0
        if next_obs_batch.dtype == torch.uint8:
            next_obs_batch = next_obs_batch.float() / 255.0
    
    with torch.no_grad():
        # Get features for current and next observations
        curr_features, _ = encoder(obs_batch)
        next_features, _ = encoder(next_obs_batch)
        
        # Compute feature similarity
        curr_norm = F.normalize(curr_features, dim=1)
        next_norm = F.normalize(next_features, dim=1)
        
        # Temporal consistency: features should be similar for consecutive frames
        temporal_similarity = torch.sum(curr_norm * next_norm, dim=1)
        
        # For temporal pairs, all should be non-terminal by construction
        # So we'll compute statistics differently
        
        # Feature variance (higher is better for representation diversity)
        feature_std = curr_features.std(dim=0).mean()
        
        # Compute some random negative pairs for comparison
        # Shuffle next_features to create random pairs
        shuffled_next = next_features[torch.randperm(len(next_features))]
        shuffled_next_norm = F.normalize(shuffled_next, dim=1)
        random_similarity = torch.sum(curr_norm * shuffled_next_norm, dim=1)
        
        metrics = {
            'temporal_consistency': temporal_similarity.mean().item(),
            'random_pair_similarity': random_similarity.mean().item(),
            'temporal_vs_random_gap': (temporal_similarity.mean() - random_similarity.mean()).item(),
            'feature_diversity': feature_std.item(),
            'feature_norm': curr_features.norm(dim=1).mean().item()
        }
    
    encoder.train()
    return metrics


def pretrain_curl(config):
    """Main CURL pretraining function"""
    device = get_device() if config['device'] == 'auto' else config['device']
    print(f"Using device: {device}")
    
    # Create directories
    current_time = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    run_id = f"{current_time}_curl_pretrain"
    if config['comment']:
        run_id = f"{run_id}_{config['comment']}"
    
    checkpoint_dir = Path('./checkpoints') / run_id
    log_dir = Path('./runs') / run_id
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(checkpoint_dir / 'curl_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    writer = SummaryWriter(log_dir=log_dir)
    
    # Load experience buffer
    buffer_size = 100000
    buffer = ExperienceBuffer(capacity=buffer_size)
    buffer.load_buffer_from_hdf5(config['buffer_path']) 

    # Determine input shape from buffer data (assuming first obs exists)
    sample_obs, _, _, _, _ = buffer.sample(1, as_torch_tensor=True, device='cpu')
    input_shape = sample_obs.shape[1:]  # Remove batch dimension
    print(f"Input shape: {input_shape}")
    
    # Initialize encoder
    encoder = CURLEncoder(input_shape, dueling_dqn=config['dueling_dqn']).to(device)
    
    # Initialize loss and optimizer
    curl_loss = CURLLoss(temperature=config['temperature'])
    optimizer = optim.Adam(encoder.parameters(), lr=config['learning_rate'])
    
    print(f"Starting CURL pretraining for {config['num_epochs']} epochs")
    print(f"Batch size: {config['batch_size']}")
    print(f"Buffer size: {len(buffer)}")
    
    # Get augmentation strategy
    sampling_strategy = config.get('augmentation_strategy', 'temporal')
    aug_transform = RandomShiftsAug(pad=config.get('aug_pad', 4)) if 'aug' in sampling_strategy else None
    
    print(f"Using sampling strategy: {sampling_strategy}")
    if sampling_strategy == 'mixed':
        print(f"  Temporal ratio: {config.get('temporal_ratio', 0.7)}")
    elif sampling_strategy == 'temporal_aug':
        print(f"  Augment probability: {config.get('augment_prob', 0.3)}")
    
    # Training loop
    step = 0
    avg_loss = 0.0  # Initialize avg_loss
    
    for epoch in range(config['num_epochs']):
        epoch_losses = []
        
        # Calculate number of batches per epoch
        batches_per_epoch = len(buffer) // config['batch_size']
        
        for batch_idx in range(batches_per_epoch):
            # Sample pairs based on strategy
            if sampling_strategy == 'temporal':
                obs_anchor, obs_positive = sample_temporal_pairs(
                    buffer, config['batch_size'], device=device
                )
            elif sampling_strategy == 'augmentation':
                obs_anchor, obs_positive = sample_augmentation_pairs(
                    buffer, config['batch_size'], device=device, aug_transform=aug_transform
                )
            elif sampling_strategy == 'mixed':
                obs_anchor, obs_positive = sample_mixed_pairs(
                    buffer, config['batch_size'], device=device,
                    temporal_ratio=config.get('temporal_ratio', 0.7),
                    aug_transform=aug_transform
                )
            elif sampling_strategy == 'temporal_aug':
                obs_anchor, obs_positive = sample_temporal_augmented_pairs(
                    buffer, config['batch_size'], device=device,
                    augment_prob=config.get('augment_prob', 0.3),
                    aug_transform=aug_transform
                )
            else:
                raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
            
            # Forward pass
            _, z_anchor = encoder(obs_anchor)
            _, z_positive = encoder(obs_positive)
            
            # Compute loss
            loss = curl_loss(z_anchor, z_positive)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Logging
            if step % config['log_interval'] == 0:
                writer.add_scalar('Loss/CURL', loss.item(), step)
                
            step += 1
        
        # Epoch logging
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Average Loss: {avg_loss:.6f}")
        writer.add_scalar('Loss/Epoch', avg_loss, epoch)
        
        # Evaluate representations
        if (epoch + 1) % config['log_interval'] == 0:
            metrics = evaluate_representations(encoder, buffer, device)
            for metric_name, metric_value in metrics.items():
                writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
            print(f"Representation metrics: {metrics}")
        
        # Save checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint_path = checkpoint_dir / f'curl_encoder_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config,
                'input_shape': input_shape
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_checkpoint_path = checkpoint_dir / 'curl_encoder_final.pth'
    torch.save({
        'epoch': config['num_epochs'],
        'encoder_state_dict': encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'config': config,
        'input_shape': input_shape
    }, final_checkpoint_path)
    
    print(f"Final model saved: {final_checkpoint_path}")
    print("CURL pretraining completed!")
    
    writer.close()
    return encoder, final_checkpoint_path


def main():
    parser = argparse.ArgumentParser(description='CURL pretraining for DQN encoder')
    parser.add_argument('--config', type=str, default='curl_config.yaml', 
                       help='Path to CURL configuration file')
    parser.add_argument('--buffer', type=str, 
                       help='Path to experience buffer .npz file')
    
    args = parser.parse_args()
    
    # Load or create config
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded CURL configuration from: {args.config}")
    except FileNotFoundError:
        print(f"❌ Config file not found: {args.config}")
        print("🔧 Creating default curl_config.yaml...")
        config = get_default_curl_config()
        with open('curl_config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print("✅ Default curl_config.yaml created. Please modify it as needed and run again.")
        return
    
    # Override buffer path if provided
    if args.buffer:
        config['buffer_path'] = args.buffer
    
    # Validate buffer path exists
    if not os.path.exists(config['buffer_path']):
        print(f"❌ Buffer file not found: {config['buffer_path']}")
        return
    
    print("CURL Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Start pretraining
    pretrain_curl(config)


if __name__ == "__main__":
    main()
