#!/usr/bin/env python3
"""
Play BeamRider manually with keyboard controls using gymnasium.utils.play

Controls:
- A/D: Left/Right
- SPACE: Fire
- ESC: Quit game
"""

import sys
import os
import gymnasium as gym
import ale_py
from gymnasium.utils.play import play
import pygame
from shared.wrappers import MaxRender
from shared.environments import atari_name_id_map

def play_atari_human(game:str):
    """Play BeamRider or Asteroids with human keyboard controls using official gymnasium.utils.play"""
    
    # Register ALE environments
    gym.register_envs(ale_py) # "import ale_py" is actually enough

    # Create environment with rgb_array mode for gymnasium.utils.play
    env_id = atari_name_id_map.get(game, game)
    print(f"🎮 Creating environment for game: {game} (env_id: {env_id})")
    try:
        env = gym.make(env_id, render_mode='rgb_array', frameskip=1)
    except Exception as e:
        raise ValueError(f"Unsupported game: {game}")
    env = MaxRender(env) # Smooth rendering for Atari games (particularly Asteroids) to avoid flickering
    # Use official gymnasium.utils.play function
  
    
    # Define key mappings (key -> action_id)
    # You can use characters, pygame constants, or tuples
    # manually defined for BeamRider, Asteroids
    custom_keys_to_action = {
        # Character keys
        "a": 4,              # LEFT
        "d": 3,              # RIGHT  
        " ": 1,              # FIRE (space)
        
        # Arrow keys using pygame constants
        pygame.K_LEFT: 4,    # LEFT
        pygame.K_RIGHT: 3,   # RIGHT
        pygame.K_UP: 2,      # UP (if needed)
        
        # Combination keys (tuple format)
        (pygame.K_LEFT, ord(' ')): 8,   # LEFT + FIRE
        (pygame.K_RIGHT, ord(' ')): 7,  # RIGHT + FIRE
    }

    # Start the official human play mode
    keys_to_action = custom_keys_to_action if game.lower() in ('beamrider', 'asteroids') else None
    play(env, zoom=3, fps=60, keys_to_action=keys_to_action)
    
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Play Atari (BeamRider, Asteroids manually with keyboard controls')
    parser.add_argument('game', type=str,
                        help='Game to play (py-asteroids, or atari games: beamrider, asteroids, etc)')
    args = parser.parse_args()

    try:
        play_atari_human(args.game.lower())
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("✅ Game closed. Thanks for playing!")
