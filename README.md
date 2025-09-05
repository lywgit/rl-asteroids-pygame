# Asteroids AI - Deep Q-Network Training

A reinforcement learning project that trains DQN agents to play a custom Asteroid shooting game.

The algorithm currently works for the Atari [Beamrider](https://ale.farama.org/environments/beam_rider/) game, but not for the asteroids game (yet).

## Background 

The goal of this project is straightforward: to combine my two previous projects, [rlbook-hands-on](https://github.com/lywgit/rlbook-hands-on) and [asteroids pygame](https://github.com/lywgit/bootdev-asteroids-pygame), and train an AI that can play the game!

It turns out this wasn't easy and I couldn't get the DQN model to learn the asteroids game. So I decided to take a step back and target the classic Atari BeamRider game first, which is presumably simpler. Fortunately, the algorithm actually works and I am able to train a descent DQN model that can score around 4000 points on average. 

I am still exploring ways to make it work on the custom asteroid game at the moment.


## 🎮 Features

- **DQN Training**: Deep Q-Network implementation for both games  
- **Config-based Training**: YAML configuration system for reproducible experiments
- **Model Inference**: Play trained models with visualization and video recording
- **Custom Asteroids Environment**: Built with Pygame (full control) and Gymnasium integration

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
git clone <repository-url>
cd asteroids-ai
uv sync
```

### Training a Model

1. **Configure training parameters** in `config.yaml`:
```yaml
game: asteroids  # or 'beamrider'
max_steps: 200000
batch_size: 128
learning_rate: 0.0001
# ... other parameters
```

2. **Start training**:
```bash
uv run train_dqn.py
```

3. **Monitor progress**:
```bash
# View tensorboard logs
tensorboard --logdir runs/
```

### Playing with Trained Models

```bash
# Play asteroids with a trained model
uv run play_dqn_model.py asteroids --model checkpoints/your-model/dqn_best.pth

# Play beamrider 
uv run play_dqn_model.py beamrider --model path/to/beamrider_model.pth

# Record gameplay videos
uv run play_dqn_model.py asteroids --model your-model.pth --record-video

# Play multiple episodes
uv run play_dqn_model.py asteroids --model your-model.pth --episodes 5
```

## 📁 Project Structure

```
asteroids-ai/
├── train_dqn.py              # Main training script
├── play_dqn_model.py         # Model inference and gameplay
├── config.yaml               # Training configuration
├── shared/                   # Shared components
│   ├── models.py             # AtariDQN network architecture
│   ├── environments.py       # Environment creation functions
│   ├── wrappers.py           # Gymnasium wrappers
│   └── utils.py              # Utility functions
├── asteroids/                # Custom Asteroids game
│   ├── gym_env.py           # Gymnasium environment wrapper
│   ├── game.py              # Core game logic
│   └── entities/            # Game entities (player, asteroids, etc.)
├── checkpoints/             # Saved model weights
├── runs/                    # Tensorboard logs
├── videos/                  # Recorded gameplay videos
└── play_asteroids_human.py  # Human-playable version
```

## 🛠 Main Scripts

### `train_dqn.py`

Configuration-based DQN training script supporting both Asteroids and BeamRider.

**Key Features:**
- YAML configuration system
- Automatic hyperparameter logging
- Periodic evaluation and model checkpointing
- Tensorboard integration
- Experience replay buffer
- Target network updates

**Usage:**
```bash
# Use default config.yaml
uv run train_dqn.py

# Use custom config
uv run train_dqn.py --config my_config.yaml
```

### `play_dqn_model.py`

Model inference script for playing games with trained DQN agents.

**Key Features:**
- Visual gameplay rendering
- Video recording capability
- Performance statistics
- Multiple episode evaluation
- Support for both game environments

**Usage:**
```bash
uv run play_dqn_model.py {asteroids|beamrider} --model MODEL_PATH [options]

Options:
  --episodes N          Number of episodes to play (default: 1)
  --delay SECONDS       Delay between steps for visualization
  --record-video        Record gameplay videos
  --no-render          Run without display for evaluation
```

## ⚙️ Configuration

Training parameters are specified in YAML files. Key parameters:

```yaml
# Game selection
game: "asteroids"  # or "beamrider"

# Training parameters
max_steps: 200000
replay_buffer_size: 100000
learning_rate: 0.0001
batch_size: 128
gamma: 0.99

# Exploration parameters
epsilon_start: 1.0
epsilon_end: 0.1
epsilon_decay_frames: 100000

# Checkpointing
checkpoint_interval: 10000
eval_episode_interval: 5
load_model: null  # Path to continue training from existing model

# Metadata
comment: "experiment_description"
```

## 🎯 Game Environments

### Asteroids
- **Custom implementation** with Pygame
- **Multi-action space**: 5 discrete actions (thrust, rotate left/right, shoot, hyperspace)
- **Action modes**: Single actions or action combinations
- **Scoring**: Points for destroying asteroids + survival bonus
- **Observation**: 84x84 grayscale frames, 4-frame stack

### BeamRider (Atari)
- **Classic Atari game** via ALE (Arcade Learning Environment)
- **Preprocessing**: Standard Atari preprocessing pipeline
- **Observation**: 84x84 grayscale frames, 4-frame stack

## 📊 Training Results

- **Checkpoints**: Saved every 10,000 steps to `checkpoints/TIMESTAMP_GAME_COMMENT/`
- **Best models**: `dqn_best.pth` saved when evaluation improves
- **Logs**: Tensorboard logs in `runs/` directory
- **Config preservation**: Training configuration saved with each run

## 🔧 Architecture

### DQN Network
- **Input**: 4-stacked grayscale frames (4, 84, 84)
- **Convolutional layers**: 3 layers with ReLU activation
- **Fully connected**: 512 hidden units → action values
- **Output**: Q-values for each possible action

### Shared Components
The project uses a modular architecture with shared components:
- **Models**: Neural network architectures
- **Environments**: Game environment creation
- **Wrappers**: Action space and observation preprocessing
- **Utils**: Device detection and common utilities

## 🎮 Human Play

```bash
# Play Asteroids as a human
uv run play_asteroids_human.py

# Controls: Arrow keys to move, Space to shoot, Esc to exit
```

## 📈 Monitoring Training

Monitor training progress with Tensorboard:

```bash
tensorboard --logdir runs/
```

Metrics tracked:
- Training episode rewards
- Evaluation episode rewards  
- Exploration epsilon values
- Hyperparameter configurations


## Author's Note

- The development of this project was assisted by AI tools, specifically GitHub Copilot Agent + Claude Sonnet 4.