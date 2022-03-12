#!/usr/bin/env python3

import pygame
import sys
import numpy as np

from gym_robot_maze.envs.maze import Maze

class MazeRender:
    """
        Class to render a maze using pygame library
    """
    def __init__(self, maze: Maze, size=(600, 600), n_agents: int=1):
        self._maze = maze
        self._n_agents = n_agents

        if self.n_agents > 1:
            self.agent_pos = [self.maze.start for i in range(self.n_agents)]
        else:
            self.agent_pos = self.maze.start
        
        #Calculate scaler value to map maze coordinates onto pixels
        self._scaler = (size[0] / self.maze.size[0], size[1] / self.maze.size[1])

        pygame.init()
        self._screen = pygame.display.set_mode(size)

    #-----------------------------------------------------------------------------------
    # Properties
    #-----------------------------------------------------------------------------------

    @property
    def maze(self):
        return self._maze

    @property
    def n_agents(self):
        return self._n_agents

    @property
    def agent_pos(self):
        return self._agent_pos

    @agent_pos.setter
    def agent_pos(self, val: np.ndarray):
        if not isinstance(val, np.ndarray):
            raise TypeError("Agent position must be an array")
        self._agent_pos = val

    @property
    def scaler(self):
        return self._scaler

    @property
    def screen(self):
        return self._screen

    #-----------------------------------------------------------------------------------
    # Methods
    #-----------------------------------------------------------------------------------

    def update(self, pos: np.ndarray, index: int=None):
        """
            function to update maze render variables

            pos is an array of the [x, y] coordinates of the agent

            index is the index of the agent if there are multiple
        """
        if index == None:
            self.agent_pos = pos
        else:
            self.agent_pos[agent_i] = pos

    def draw(self):
        """
            function to draw the maze to the screen
        """
        self.screen.fill((30, 30, 30))
        WHITE = (225, 225, 225)
        RED = (225, 0, 0)
        BLUE = (0, 0, 225)

        for wall in self.maze.walls:
            pygame.draw.rect(self.screen, WHITE, ((wall[0] * self.scaler[0], wall[1] * self.scaler[1]), (self.scaler)))

        pygame.draw.rect(self.screen, RED, ((self.maze.goal * self.scaler), (self.scaler)))
        #Radius of agent should be 1/3 of the scaler value in the smaller dimension
        r = self.scaler[0] / 3 if self.scaler[0] < self.scaler[1] else self.scaler[1] / 3

        if self.n_agents > 1:
            for i in range(self.n_agents):
                pygame.draw.circle(self.screen, BLUE, ((self.agent_pos[i][0] + 0.5) * self.scaler[0], (self.agent_pos[i][1] + 0.5) * self.scaler[1]), r)
        else:
            pygame.draw.circle(self.screen, BLUE, ((self.agent_pos[0] + 0.5) * self.scaler[0], (self.agent_pos[1] + 0.5) * self.scaler[1]), r)
        
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

