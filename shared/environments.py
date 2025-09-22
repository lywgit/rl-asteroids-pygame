"""
Environment creation functions for Atari and Asteroids games.
"""

import gymnasium as gym
import ale_py
from gymnasium.wrappers import (
    AtariPreprocessing, 
    FrameStackObservation, 
    MaxAndSkipObservation, 
    ResizeObservation,
    GrayscaleObservation,
    ClipReward
)
from asteroids.gym_env import AsteroidsEnv
from .wrappers import MultiBinaryToSingleDiscreteAction, MultiBinaryToDiscreteCombinationWrapper, ScaleObservation


atari_name_id_map = {
    'pong': 'ALE/Pong-v5',
    'beamrider': 'ALE/BeamRider-v5',
    'enduro': 'ALE/Enduro-v5',
    'spaceinvaders': 'ALE/SpaceInvaders-v5',
    'asteroids': 'ALE/Asteroids-v5',
    'centipede': 'ALE/Centipede-v5'
}

def make_atari_env(env_id: str, render_mode: str = "rgb_array", max_episode_steps: int = 100000, 
                   screen_size=(84, 84), frame_stack: int = 4, scale_obs: bool = True, 
                   clip_reward: bool = True,
                   grayscale_obs=True,
                   **kwargs):
    """
    Create Atari environment with standard preprocessing.
    
    Args:
        env_id: Gymnasium environment ID (e.g., "ALE/BeamRider-v5")
        render_mode: Rendering mode ("human", "rgb_array", etc.)
        max_episode_steps: Maximum steps per episode, default 1e5
        frame_stack: Number of frames to stack, default 4 (-> FrameStackObservation)
        clip_reward: Whether to clip rewards to [-1, 1], default True (-> ClipReward)
        screen_size: Tuple of (height, width) for observation resizing, default (84, 84) (-> AtariPreprocessing)
        scale_obs: Whether to scale observations to [0,1], default True (-> AtariPreprocessing)
        grayscale_obs: Make observation gray scale, default True (-> AtariPreprocessing)
        **kwargs: Additional arguments passed to AtariPreprocessing
        
    Returns:
        Configured Atari environment
    """
    env = gym.make(env_id, render_mode=render_mode, max_episode_steps=max_episode_steps, frameskip=1) 
    # disable initial frame skipping because AtariPreprocessing does that, too
    env = AtariPreprocessing(env, screen_size=screen_size, scale_obs=scale_obs, grayscale_obs=grayscale_obs, **kwargs)
    if frame_stack > 1:
        env = FrameStackObservation(env, frame_stack)
    if clip_reward:
        env = ClipReward(env, -1, 1)
    return env


def make_py_asteroids_env(render_mode: str = "rgb_array", screen_size=(84, 84), grayscale_obs: bool = True, 
                       scale_obs: bool = True, frame_stack: int = 4, clip_reward: bool = True,
                       action_mode: str = "single"):
    """
    Create Asteroids environment with preprocessing.
    
    Args:
        render_mode: Rendering mode ("human", "rgb_array", etc.)
        screen_size: Tuple of (height, width) for observation resizing, default (84, 84)
        grayscale_obs: Whether to convert to grayscale, default True
        scale_obs: Whether to scale observations to [0,1], default True
        frame_stack: Number of frames to stack, default 4
        clip_reward: Whether to clip rewards to [-1, 1], default True
        action_mode: Action space mode - "single" for single discrete actions, 
                    "combination" for discrete combination of all binary actions
    
    Returns:
        Configured Asteroids environment
    """
    env = AsteroidsEnv(render_mode=render_mode)
    env = MaxAndSkipObservation(env, skip=4)
    env = ResizeObservation(env, shape=screen_size)
    if grayscale_obs:
        env = GrayscaleObservation(env)
    if scale_obs:
        env = ScaleObservation(env)
    if frame_stack > 1:
        env = FrameStackObservation(env, frame_stack)
    if clip_reward:
        env = ClipReward(env, -1, 1)
    
    # Action space conversion
    if action_mode == "single":
        env = MultiBinaryToSingleDiscreteAction(env)
    elif action_mode == "combination":
        env = MultiBinaryToDiscreteCombinationWrapper(env)
    else:
        raise ValueError(f"Unknown action_mode: {action_mode}. Use 'single' or 'combination'")
    
    return env
