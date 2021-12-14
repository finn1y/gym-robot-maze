#!/usr/bin/env python3

import pygame
import sys

from gym_robot_maze.envs.maze import Maze

class MazeRender:
    """
        Class to render a maze using pygame library
    """
    def __init__(self, maze: Maze, size=(600, 600)):
        self.__maze = maze
        self._agent_pos = self.__maze.get_start()
        #Calculate scalar value to map maze coordinates onto pixels
        self.__scalar = (size[0] / self.__maze.get_size()[0], size[1] / self.__maze.get_size()[1])

        pygame.init()
        self.__screen = pygame.display.set_mode(size)

    def update(self, agent_pos):
        """
            function to update maze render variables

            agent_pos is an array of the [x, y] coordinates of the agent
        """
        self._agent_pos = agent_pos

    def draw(self):
        """
            function to draw the maze to the screen
        """
        self.__screen.fill((30, 30, 30))
        WHITE = (225, 225, 225)
        RED = (225, 0, 0)
        BLUE = (0, 0, 225)

        for wall in self.__maze.get_wall():
            pygame.draw.rect(self.__screen, WHITE, ((wall[0] * self.__scalar[0], wall[1] * self.__scalar[1]), (self.__scalar)))

        pygame.draw.rect(self.__screen, RED, ((self.__maze.get_goal() * self.__scalar), (self.__scalar)))
        #Radius of agent should be 1/3 of the scalar value in the smaller dimension
        r = self.__scalar[0] / 3 if self.__scalar[0] < self.__scalar[1] else self.__scalar[1] / 3
        pygame.draw.circle(self.__screen, BLUE, ((self._agent_pos[0] + 0.5) * self.__scalar[0], (self._agent_pos[1] + 0.5) * self.__scalar[1]), r)

        pygame.display.flip()

#Testing
if __name__ == '__main__':
    m1 = Maze()
    render = MazeRender(m1)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
        
        render.draw()

