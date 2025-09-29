# Asteroids AI Research and Implementation Roadmap

## Goal

Develop and train a DQN Agent that can learn to play the custom Py-Asteroids game.

## Phase 1: Reinforcement Learning Integration 

- [x] 1.1 Adapt Custom Py-Asteroids Game to a Standard Gymnasium Environment
  - [x] 1.1.1 Refactor games core logic (game state, step, reset, render, etc.) from the main loop and input handling.
  - [x] 1.1.2 Create Environment Class that inherits from `gymnasium.Env`.  
- [x] 1.2 Human Playability
- [x] 1.3 Agent Playability: evaluate scores and record video 

## Phase 2: Basic DQN implementation and exploration

- [x] 2.1 Implement DQN Agent with Experience Replay Buffer
  - [x] 2.1.1 Double DQN 
  - [x] 2.2.2 Dueling DQN
- [r] 2.2 Train Py-Asteroids agent 
    > Not successful, should verify algorithm on other Atari games first.  
- [x] 2.3 Broaden the scope to include more Atari games for better over view of agent capability
- [x] 2.4 Explore hyperparameter on multiple Atari games especially small batch size + long epoch + large replay buffer 
- [r] Train Py-Asteroids agent
    > Still no luck
- [x] 2.5 Wrap up code as v1.0 

## Phase 3: Advance Rainbow DQN technique Implementation 

- [x] 3.1 Implement Rainbow DQN techniques
  - [x] 3.1.1 Prioritized Experience Replay (PER)
  - [x] 3.1.2 Multi-step Learning
  - [x] 3.1.3 Distributional Q-learning
  - [x] 3.1.4 Noisy Network
- [x] 3.2 Train Py-Asteroids with full feature
    > Seem to work!
- [ ] 3.3 Wrap up code as v2.0
  - [x] 3.3.1 Improve model integrity by using model_config
  - [ ] 3.3.2 Add versioning for Py-Asteroids game setup  
  - [ ] 3.3.3 map actions to ale game actions
- [ ] 3.4 Benchmark on Atari games and Py-Asteroids

## Phase 4: Further work
- [ ] 4.1 Review and Double check each implementation
- [ ] 4.2 Run Ablation studies 
- [ ] 4.3 Design more env version for research. Ex filled color, slow bullet-speed, instant time survival etc


## Reminder

ROADMAP is about: 
- WHAT and WHY
- Epic and Feature level
- Historical and strategic document