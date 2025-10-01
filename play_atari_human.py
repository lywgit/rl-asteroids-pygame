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

 
# Define key mappings (key -> action_id)
# You can use characters, pygame constants, or tuples
# use arrow keys for convenience
CUSTOM_KEYS_TO_ACTION_DICT = {
    "centipede": {
        # 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN, 6: UPRIGHT, 7: UPLEFT, 8: DOWNRIGHT, 9: DOWNLEFT,
        # 10: UPFIRE, 11: RIGHTFIRE, 12: LEFTFIRE, 13: DOWNFIRE, 14: UPRIGHTFIRE, 15: UPLEFTFIRE, 16: DOWNRIGHTFIRE, 17: DOWNLEFTFIRE
        pygame.K_SPACE: 1,
        pygame.K_UP: 2,
        pygame.K_RIGHT: 3,
        pygame.K_LEFT: 4,
        pygame.K_DOWN: 5,
        # Combination keys (tuple format)
        (pygame.K_UP, pygame.K_RIGHT): 6,
        (pygame.K_UP, pygame.K_LEFT): 7,
        (pygame.K_DOWN, pygame.K_RIGHT): 8,
        (pygame.K_DOWN, pygame.K_LEFT): 9,
        (pygame.K_UP, pygame.K_SPACE): 10,
        (pygame.K_RIGHT, pygame.K_SPACE): 11,
        (pygame.K_LEFT, pygame.K_SPACE): 12,
        (pygame.K_DOWN, pygame.K_SPACE): 13,
        (pygame.K_UP, pygame.K_RIGHT, pygame.K_SPACE): 14,
        (pygame.K_UP, pygame.K_LEFT, pygame.K_SPACE): 15,
        (pygame.K_DOWN, pygame.K_RIGHT, pygame.K_SPACE): 16,
        (pygame.K_DOWN, pygame.K_LEFT, pygame.K_SPACE): 17
    },
    "asteroids": { # DISABLE DOWN KEY! cause ship to disappear
        # 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN, 6: UPRIGHT, 7: UPLEFT,
        # 8: UPFIRE, 9: RIGHTFIRE, 10: RIGHTFIRE, 11: DOWNFIRE 12: UPRIGHTFIRE, 13: UPLEFTFIRE
        pygame.K_SPACE: 1,  
        pygame.K_UP: 2,       
        pygame.K_RIGHT: 3,   
        pygame.K_LEFT:  4,    
        # pygame.K_DOWN: 5,     
        
        # Combination keys (tuple format)
        (pygame.K_UP, pygame.K_RIGHT): 6,
        (pygame.K_UP, pygame.K_LEFT): 7,
        (pygame.K_UP, pygame.K_SPACE): 8,
        (pygame.K_RIGHT, pygame.K_SPACE): 9,
        (pygame.K_LEFT, pygame.K_SPACE): 10,
        # (pygame.K_DOWN, pygame.K_SPACE): 11, 
        (pygame.K_UP, pygame.K_RIGHT, pygame.K_SPACE): 12,
        (pygame.K_UP, pygame.K_LEFT, pygame.K_SPACE): 13
    },
    "beamrider": {
        # 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: UPRIGHT, 6: UPLEFT,
        # 7: RIGHTFIRE, 8: LEFTFIRE
        pygame.K_SPACE: 1,
        pygame.K_UP: 2,
        pygame.K_RIGHT: 3,
        pygame.K_LEFT: 4,
        # Combination keys (tuple format)
        (pygame.K_UP, pygame.K_RIGHT): 5,
        (pygame.K_UP, pygame.K_LEFT): 6,
        (pygame.K_RIGHT, pygame.K_SPACE): 7,
        (pygame.K_LEFT, pygame.K_SPACE): 8
    },
    "enduro": {
        # 1: FIRE, 2: RIGHT, 3: LEFT, 4: DOWN, 5: DOWNRIGHT, 6: DOWNLEFT,
        # 7: RIGHTFIRE, 8: LEFTFIRE
        pygame.K_SPACE: 1,
        pygame.K_RIGHT: 2,
        pygame.K_LEFT: 3,
        pygame.K_DOWN: 4,
        # Combination keys (tuple format)
        (pygame.K_DOWN, pygame.K_RIGHT): 5,
        (pygame.K_DOWN, pygame.K_LEFT): 6,
        (pygame.K_RIGHT, pygame.K_SPACE): 7,
        (pygame.K_LEFT, pygame.K_SPACE): 8
    },
    "pong":{
        # 1: FIRE, 2: RIGHT, 3: LEFT, 
        # 4: RIGHTFIRE, 5: LEFTFIRE
        pygame.K_SPACE: 1,
        pygame.K_RIGHT: 2,
        pygame.K_LEFT: 3,
        # Combination keys (tuple format)
        (pygame.K_RIGHT, pygame.K_SPACE): 4,
        (pygame.K_LEFT, pygame.K_SPACE): 5
    },
    "spaceinvaders":{
        # 1: FIRE, 2: RIGHT, 3: LEFT, 
        # 4: RIGHTFIRE, 5: LEFTFIRE
        pygame.K_SPACE: 1,
        pygame.K_RIGHT: 2,
        pygame.K_LEFT: 3,
        # Combination keys (tuple format)
        (pygame.K_RIGHT, pygame.K_SPACE): 4,
        (pygame.K_LEFT, pygame.K_SPACE): 5
    }
}


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
    if game in {'asteroids'}:
        print('Enabling smooth rendering')
        env = MaxRender(env) # Smooth rendering for Atari games (particularly Asteroids) to avoid flickering
    
    # Score tracking variables
    current_score = 0
    episode_count = 0
    step_count = 0
    
    def score_callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
        """Callback to track and display scores when episodes end"""
        nonlocal current_score, episode_count, step_count
        
        # Accumulate reward for current episode
        current_score += rew
        step_count += 1
        
        # Show current score every 300 steps (roughly every 5 seconds at 60 fps)
        if step_count % 300 == 0 and current_score > 0:
            print(f"💫 Current Score: {current_score:.0f}")  # Will appear in terminal, not game window
        
        # Check if episode is done (terminated or truncated)
        if terminated or truncated:
            episode_count += 1
            end_reason = "Game Over" if terminated else "Time Limit"
            print(f"\n🏆 Episode {episode_count} finished! ({end_reason})")
            print(f"📊 Final Score: {current_score:.0f}")
            
            # Show additional info if available (some Atari games provide lives info)
            if 'lives' in info:
                print(f"💀 Lives remaining: {info['lives']}")
            
            print("=" * 40)
            
            # Reset score for next episode
            current_score = 0
            step_count = 0
    
    # Use official gymnasium.utils.play function
  
   
    # Start the official human play mode
    keys_to_action = CUSTOM_KEYS_TO_ACTION_DICT.get(game, None)
    print("🎯 Score tracking enabled! Scores will be displayed when episodes end.")
    play(env, zoom=3, fps=60, keys_to_action=keys_to_action, callback=score_callback)
    
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
