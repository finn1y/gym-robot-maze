#!/usr/bin/env python3

import pygame
import sys

from gym_robot_maze.envs.maze import Maze

class MazeRender:
    """
        Class to render a maze using pygame library
    """
    def __init__(self, maze: Maze, size=(600, 600), n_agents=1):
        self.__maze = maze
        self._agent_pos = [self.__maze.get_start() for i in range(n_agents)]
        #Calculate scaler value to map maze coordinates onto pixels
        self.__scaler = (size[0] / self.__maze.get_size()[0], size[1] / self.__maze.get_size()[1])
        self.n_agents = n_agents

        pygame.init()
        self.__screen = pygame.display.set_mode(size)

    def update(self, agent_pos):
        """
            function to update maze render variables

            agent_pos is an array of the [x, y] coordinates of the agent
        """
        for i in range(self.n_agents):
            self._agent_pos[i] = agent_pos[i]

    def draw(self):
        """
            function to draw the maze to the screen
        """
        self.__screen.fill((30, 30, 30))
        WHITE = (225, 225, 225)
        RED = (225, 0, 0)
        BLUE = (0, 0, 225)

        for wall in self.__maze.get_wall():
            pygame.draw.rect(self.__screen, WHITE, ((wall[0] * self.__scaler[0], wall[1] * self.__scaler[1]), (self.__scaler)))

        pygame.draw.rect(self.__screen, RED, ((self.__maze.get_goal() * self.__scaler), (self.__scaler)))
        #Radius of agent should be 1/3 of the scaler value in the smaller dimension
        r = self.__scaler[0] / 3 if self.__scaler[0] < self.__scaler[1] else self.__scaler[1] / 3

        for i in range(self.n_agents):
            pygame.draw.circle(self.__screen, BLUE, ((self._agent_pos[i][0] + 0.5) * self.__scaler[0], (self._agent_pos[i][1] + 0.5) * self.__scaler[1]), r)

        pygame.display.flip()

    def quit(self):
        pygame.quit()

#Testing
if __name__ == '__main__':
    m1 = Maze()
    render = MazeRender(m1)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
        
        render.draw()

