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
            if done and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
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
            time.sleep(0.016)
        else:
            print(f"Game over! Score: {info['score']}")
            print("Press R to restart or close the window to exit.")
            time.sleep(0.1)
    env.close()

if __name__ == "__main__":
    main()
