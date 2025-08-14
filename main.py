def main():

    import pygame
    from asteroids_game import AsteroidsGame

    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("Asteroids AI")
    clock = pygame.time.Clock()
    game = AsteroidsGame()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Placeholder: always do nothing (action=0)
        game.step(0)
        game.render(screen)
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
