#!/usr/bin/env python3

import numpy as np
import gym
import pygame
import sys
import time

from gym import Env, spaces
from gym_robot_maze.envs.maze import Maze
from gym_robot_maze.envs.maze_render import MazeRender

class MazeEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, is_render: bool=True):
        super(MazeEnv, self).__init__()
        self.maze = Maze()
        self.is_render = is_render

        if self.is_render:
            self.maze_render = MazeRender(self.maze)

        self.observation_space = spaces.Box(low=0, high=np.finfo(np.float32).max, shape=(1, 3), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

        self.agent = Agent(pos=self.maze.get_start(), facing=180)
        self.state = self.get_state() 

    def step(self, action):
        """
            function to step the environment given an action

            action is an int corresponding to the agents action 

            returns a tuple of (reward: int, done: bool) where reward is the reward 
            for the agent and done is whether or not the goal state is reached
        """
        assert self.action_space.contains(action), "Invalid Action"
        done = False

        if action == 0:
            R = -100
            #NOOP
            
        elif action == 1:
            R = -1
            vel = self.agent.move()

            #Check for collision with walls if move carried out
            for wall in self.maze.get_wall():
                if self.agent.pos[0] + vel[0] == wall[0] and self.agent.pos[1] + vel[1] == wall[1]:
                    R = -50
                    vel = (0, 0)
                    break

            #Update agents position
            self.agent.pos[0] += vel[0]
            self.agent.pos[1] += vel[1]
            self.maze_render.update(self.agent.pos)

        elif action == 2:
            R = -1
            self.agent.rotate(90)
        elif action == 3:
            R = -1
            self.agent.rotate(180)
        elif action == 4:
            R = -1
            self.agent.rotate(270)
    
        #Check for goal state
        if self.agent.pos[0] == self.maze.get_goal()[0] and self.agent.pos[1] == self.maze.get_goal()[1]:
            done = True
            R = 500

        return self.get_state(), R, done

    def reset(self):
        self.agent.pos = self.maze.get_start()
        self.agent.facing = 180
        self.done = False

    def render(self, mode='human'):
        """
            function to draw objects to screen and exit program on window close
        """
        if self.is_render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()    

            self.maze_render.draw()

    def close(self):
        None

    def get_state(self):
        """
            function to get the current state of the agent within the environment

            returns an array of [foward_dist, left_dist, right_dist] where each is the distance to the nearest wall in that direction
        """
        #initiliase dists in each direction to the maximum possible value
        dists = [self.maze.get_size()[1], self.maze.get_size()[0], self.maze.get_size()[1], self.maze.get_size()[0]]

        for wall in self.maze.get_wall():
            #dists of 0 are impossible as would be on top of agent
            dist_y = 0
            dist_x = 0

            #calculate distance of each wall in x and y direction
            if self.agent.pos[0] == wall[0]:
                dist_y = wall[1] - self.agent.pos[1]
            if self.agent.pos[1] == wall[1]:
                dist_x = wall[0] - self.agent.pos[0]
            
            #if distance to wall is shorter than previous calculated then replace distance
            if dist_y > 0 and dist_y < dists[2]:
                dists[2] = dist_y 
            elif dist_y < 0 and -dist_y < dists[0]:
                dists[0] = -dist_y 

            if dist_x > 0 and dist_x < dists[1]:
                dists[1] = dist_x 
            elif dist_x < 0 and -dist_x < dists[3]:
                dists[3] = -dist_x 

        forward = dists[int(self.agent.facing / 90)]
        left = dists[(int(self.agent.facing / 90) + 3) % 4]
        right = dists[(int(self.agent.facing / 90) + 1) % 4]
        
        return np.array([forward, left, right], dtype=np.float32)

class Agent():
    """
        class to contain all agent variables
    """
    def __init__(self, pos: np.ndarray=np.zeros(2), facing=0):
        self.pos = pos
        self.facing = facing

    def rotate(self, angle):
        self.facing = (self.facing + angle) % 360

    def move(self):
        """
            function to move the agent forward in the current facing direction

            returns the velocity of the agent as a tuple (x, y)
        """
        if self.facing == 0 or self.facing == 180:
            x = 0
            y = 1 if self.facing == 180 else -1
        elif self.facing == 90 or self.facing == 270:
            x = 1 if self.facing == 90 else -1
            y = 0
        elif self.facing > 0 and self.facing < 180:
            x = 1
            y = 1 if self.facing > 90 else -1
        elif self.facing > 180 and self.facing < 360:
            x = -1
            y = 1 if self.facing < 270 else -1

        return x, y

if __name__ == '__main__':
    env = MazeEnv()
    m1 = env.maze
    size = m1.get_size()

    print(f'{size[0]} x {size[1]} maze, start at {m1.get_start()}, goal at {m1.get_goal()}')
    
    env.reset()
    for _ in range(10000):
        observation, reward, done = env.step(env.action_space.sample())
        print(observation)
        env.render()
        
        if done == True:
            print("Done")
            break

