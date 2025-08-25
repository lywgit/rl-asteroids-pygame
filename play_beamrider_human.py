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

# Add the project root to the path
sys.path.append(os.path.abspath('.'))


def play_beamrider_human():
    """Play BeamRider with human keyboard controls using official gymnasium.utils.play"""
    
    print("🎮 BeamRider - Human Player Mode (Official)")
    print("=" * 50)
    print("Controls:")
    print("  A / ←: Move Left")
    print("  D / →: Move Right")
    print("  ↑: Move Up")
    print("  SPACE: Fire")
    print("  ← + SPACE: Left + Fire")
    print("  → + SPACE: Right + Fire")
    print("  ESC: Quit game")
    print("=" * 50)
    
    # Register ALE environments
    gym.register_envs(ale_py)
    
    # Create environment with rgb_array mode for gymnasium.utils.play
    env = gym.make("ALE/BeamRider-v5", render_mode='rgb_array', frameskip=1)
    
    # Use official gymnasium.utils.play function
    from gymnasium.utils.play import play
    import pygame
    
    # Define key mappings (key -> action_id)
    # You can use characters, pygame constants, or tuples
    keys_to_action = {
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
    play(env, zoom=3, fps=60, keys_to_action=keys_to_action)
    
    env.close()


if __name__ == "__main__":
    print("🕹️  BeamRider Human Player (Official)")
    
    try:
        play_beamrider_human()
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("✅ Game closed. Thanks for playing!")
    print("🕹️  BeamRider Human Player")
    print("Make sure you have pygame installed: pip install pygame")
    
    try:
        play_beamrider_human()
    except ImportError as e:
        if "pygame" in str(e):
            print("❌ pygame not found. Install it with:")
            print("   uv add pygame")
            print("   # or")
            print("   pip install pygame")
        else:
            print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
