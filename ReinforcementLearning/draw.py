import pygame
import os


class DrawGrid:
    def __init__(self, obstacles, rewards, width, height, arrows_direction):
        # Define some colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0,255,0)
        # Define the dimensions of the grid
        self.GRID_WIDTH = width
        self.GRID_HEIGHT = height
        # Define the size of each box in the grid
        self.BOX_SIZE = 40

        # Set the dimensions of the Pygame window
        self.WINDOW_WIDTH = self.BOX_SIZE*width
        self.WINDOW_HEIGHT = self.BOX_SIZE*height
        
        self.obstacles = obstacles
        self.rewards = rewards
        self.arrowDirections = arrows_direction
        # Used to fit arrow images within grid boxes
        self.constant1 = 2
        self.constant2 = 3
        self.load()
        
    def load(self):
        # Initialize Pygame
        # os.environ['SDL_VIDEO_CENTERED'] = '1'
        pygame.init()
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        # Set the background to white
        self.screen.fill(self.WHITE)  

        # Load images of all 4 arrows and resize it to fit within the box size of the grid boxes
        self.uparrow = pygame.image.load('uparrow.png')
        self.uparrow = pygame.transform.scale(self.uparrow, (self.BOX_SIZE- self.constant2, self.BOX_SIZE- self.constant2))
        self.downarrow = pygame.image.load('downarrow.png')
        self.downarrow = pygame.transform.scale(self.downarrow, (self.BOX_SIZE- self.constant2, self.BOX_SIZE- self.constant2))
        self.rightarrow = pygame.image.load('rightarrow.png')
        self.rightarrow = pygame.transform.scale(self.rightarrow, (self.BOX_SIZE- self.constant2, self.BOX_SIZE- self.constant2))
        self.leftarrow = pygame.image.load('leftarrow.png')
        self.leftarrow = pygame.transform.scale(self.leftarrow, (self.BOX_SIZE- self.constant2, self.BOX_SIZE- self.constant2))
        
        self.drawGrid()
    
    def drawArrows(self, row,col, arrowtype):
        # Draw the uparrow on top of the grid in the box at position (2, 2)
        x = col * self.BOX_SIZE
        y = row * self.BOX_SIZE

        if arrowtype == "UP":
            self.screen.blit(self.uparrow, (x +self.constant1, y+self.constant1))
        elif arrowtype == "DOWN":
            self.screen.blit(self.downarrow, (x +self.constant1, y+self.constant1))
        elif arrowtype == "LEFT":
            self.screen.blit(self.leftarrow, (x +self.constant1, y+self.constant1))
        elif arrowtype == "RIGHT":
            self.screen.blit(self.rightarrow, (x +self.constant1, y+self.constant1))    

    def drawGrid(self):
        # Draw the grid of boxes on top of the uparrow
        for row in range(self.GRID_HEIGHT):
            for col in range(self.GRID_WIDTH):
                # print(row,col)
                x = col * self.BOX_SIZE
                y = row * self.BOX_SIZE
                pygame.draw.rect(self.screen, self.BLACK, [x, y, self.BOX_SIZE, self.BOX_SIZE], 1)
                if (row, col) in self.obstacles:
                    pygame.draw.rect(self.screen, self.RED, [x+1, y+1, self.BOX_SIZE-2, self.BOX_SIZE-2])
                elif (row,col) in  self.rewards:
                    pygame.draw.rect(self.screen, self.GREEN, [x+1, y+1, self.BOX_SIZE-2, self.BOX_SIZE-2])
                else:
                    # print(self.arrowDirections[(10*row)+col])
                    self.drawArrows(row, col, self.arrowDirections[(self.GRID_HEIGHT *row)+col])
               
        self.others()

    def others(self):
        # Set the title of the Pygame window
        pygame.display.set_caption("Reinforcement Learning via Episodic Learning")

        # Update the display
        pygame.display.flip()

        # Run the Pygame event loop
        clock = pygame.time.Clock()
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            clock.tick(60)

        # Quit Pygame
        pygame.quit()








