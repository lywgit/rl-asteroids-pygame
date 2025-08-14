
import time
import gymnasium as gym
import pygame
from asteroids_env import AsteroidsEnv


def main():
    env = AsteroidsEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    running = True
    print("Controls: Arrow keys to move, Space to shoot. Close window to exit.")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        # Multiple key support: call step for each action
        # 0: do nothing, 1: left, 2: right, 3: thrust, 4: shoot, 5: backward
        actions = []
        if keys[pygame.K_LEFT]:
            actions.append(1)
        if keys[pygame.K_RIGHT]:
            actions.append(2)
        if keys[pygame.K_UP]:
            actions.append(3)
        if keys[pygame.K_DOWN]:
            actions.append(5)
        if keys[pygame.K_SPACE]:
            actions.append(4)
        if not actions:
            actions = [0]
        for action in actions:
            obs, reward, done, truncated, info = env.step(action)
        time.sleep(0.016)
        if done:
            print(f"Game over! Score: {obs[-1] if hasattr(obs, '__getitem__') else 'unknown'}")
            obs, info = env.reset()
    env.close()

if __name__ == "__main__":
    main()
