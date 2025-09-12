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


def make_atari_env(env_id: str, render_mode: str = "rgb_array", max_episode_steps: int = 10000, 
                   screen_size=(84, 84), frame_stack: int = 4, scale_obs: bool = True, 
                   **kwargs):
    """
    Create Atari environment with standard preprocessing.
    
    Args:
        env_id: Gymnasium environment ID (e.g., "ALE/BeamRider-v5")
        render_mode: Rendering mode ("human", "rgb_array", etc.)
        max_episode_steps: Maximum steps per episode
        screen_size: Tuple of (height, width) for observation resizing
        frame_stack: Number of frames to stack
        scale_obs: Whether to scale observations to [0,1]
        **kwargs: Additional arguments passed to AtariPreprocessing (e.g., grayscale_obs=True)
    
    Returns:
        Configured Atari environment
    """
    env = gym.make(env_id, render_mode=render_mode, max_episode_steps=max_episode_steps, frameskip=1) 
    # disable initial frame skipping because AtariPreprocessing does that, too
    env = AtariPreprocessing(env, screen_size=screen_size, scale_obs=scale_obs, **kwargs)
    if frame_stack > 1:
        env = FrameStackObservation(env, frame_stack)
    return env


def make_asteroids_env(render_mode: str = "rgb_array", screen_size=(84, 84), grayscale_obs: bool = True, 
                       scale_obs: bool = True, frame_stack: int = 4, clip_reward: bool = False,
                       action_mode: str = "single"):
    """
    Create Asteroids environment with preprocessing.
    
    Args:
        render_mode: Rendering mode ("human", "rgb_array", etc.)
        screen_size: Tuple of (height, width) for observation resizing
        grayscale_obs: Whether to convert to grayscale
        scale_obs: Whether to scale observations to [0,1]
        frame_stack: Number of frames to stack
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
