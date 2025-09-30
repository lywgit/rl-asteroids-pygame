import time
import argparse
import gymnasium as gym
import pygame
from asteroids.gym_env import AsteroidsEnv


def main():
    parser = argparse.ArgumentParser(description='Play py-asteroids manually with keyboard controls')
    parser.add_argument('-v', '--version', type=str, default='py-asteroids-v1',
                        help='Game version to play (default: py-asteroids-v1)')
    args = parser.parse_args()
    
    print(f"🎮 Starting py-asteroids game - Version: {args.version}")
    
    try:
        env = AsteroidsEnv(render_mode="human", config_version=args.version)
        obs, info = env.reset()
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    done = False
    running = True
    print(f"🎯 Game Config: {args.version}")
    print("Controls: Arrow keys to move, Space to shoot, Esc to exit. Close window to exit.")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif done and event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
        keys = pygame.key.get_pressed()
        # Build MultiBinary(5) action: [thrust, backward, left, right, shoot]
        action = [0, 0, 0, 0, 0]
        if keys[pygame.K_UP]:
            action[0] = 1  # thrust
        if keys[pygame.K_DOWN]:
            action[1] = 1  # backward
        if keys[pygame.K_LEFT]:
            action[2] = 1  # left
        if keys[pygame.K_RIGHT]:
            action[3] = 1  # right
        if keys[pygame.K_SPACE]:
            action[4] = 1  # shoot
        if not done:
            obs, reward, done, truncated, info = env.step(action)
            # Removed time.sleep(0.016) - gym_env handles timing with clock.tick(60)
        else:
            print(f"Game over!")
            print(f"  Score (asteroids): {info['score']}")
            print(f"  Survival bonus: {info['survival_reward']:.1f}")
            print(f"  Total reward: {info['total_reward']:.1f}")
            print(f"  Level reached: {info['level']}")
            print(f"  Time survived: {int(info['game_time']//60):02d}:{int(info['game_time']%60):02d}")
            print("Press R to restart or Esc to close the window and exit.")
            time.sleep(0.1)
    print("Game session ended.")
    env.close()

if __name__ == "__main__":
    main()
