#!/usr/bin/env python3

import numpy as np
import gym
import pygame
import pickle
import sys, os
import time

from gym import Env, spaces
from gym_robot_maze.envs.maze import Maze
from gym_robot_maze.envs.maze_render import MazeRender

class Agent():
    """
        class to contain all agent variables
    """
    def __init__(self, pos: np.ndarray=np.zeros(2), facing=0.0):
        self.pos = pos
        self.facing = facing

    #-----------------------------------------------------------------------------------
    # Properties
    #-----------------------------------------------------------------------------------

    @property
    def pos(self) -> np.ndarray:
        return self._pos

    @pos.setter
    def pos(self, val: np.ndarray):
        if not isinstance(val, np.ndarray):
            raise TypeError("Pos (position) must be a numpy array of x and y coordinates.")
        self._pos = val

    @property
    def facing(self):
        return self._facing

    @facing.setter
    def facing(self, val: float):
        if val < 0.0 or val > 360.0:
            raise ValueError("Facing must have a value between 0 and 360 degrees.")
        if not isinstance(val, float):
            raise TypeError("Facing must be a float.")
        self._facing = val

    #-----------------------------------------------------------------------------------
    # Methods
    #-----------------------------------------------------------------------------------

    def rotate(self, angle):
        self.facing = (self.facing + angle) % 360.0

    def move(self):
        """
            function to move the agent forward in the current facing direction

            returns the velocity of the agent as a tuple (x, y)
        """
        if self.facing == 0.0 or self.facing == 180.0:
            x = 0
            y = 1 if self.facing == 180.0 else -1
        elif self.facing == 90.0 or self.facing == 270.0:
            x = 1 if self.facing == 90.0 else -1
            y = 0
        elif self.facing > 0.0 and self.facing < 180.0:
            x = 1
            y = 1 if self.facing > 90.0 else -1
        elif self.facing > 180.0 and self.facing < 360.0:
            x = -1
            y = 1 if self.facing < 270.0 else -1

        return x, y

class MazeEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, is_render: bool=False, n_agents: int=1, save_robot_path: bool=False, save_maze_path: str=None, load_maze_path: str=None):
        """
            function to initialise Maze Environmnt class

            is_render is a bool to determine whether to render the maze using PyGame

            n_agents is an int of the number of agents within the maze

            save_robot_path is a bool to determine whether to return the agents' path through the maze once reaching the goal state

            save_maze_path is a path to save the maze file to

            load_maze_path is a path to a maze file to be loaded
        """
        super(MazeEnv, self).__init__()
        
        #load maze from path if provided else generate random maze
        if load_maze_path:
            self._maze = self.load_maze(load_maze_path)
        else:
            self._maze = Maze()

        #if save maze path provided save maze to path
        if save_maze_path:
            self.save_maze(save_maze_path)

        self._is_render = is_render
        self._n_agents = n_agents
        self._save_robot_path = save_robot_path

        if self.is_render:
            self._maze_render = MazeRender(self.maze, n_agents=self.n_agents)
        
        #observation space has three dimenstion [forward_dist, left_dist, right_dist] where each is the distance to the nearest wall in the direction
        self.observation_space = spaces.Box(low=np.zeros(3, dtype=np.float32), high=np.array([max(self.maze.size[0], self.maze.size[1]) for i in range(3)], dtype=np.float32), dtype=np.float32)
        #action space has 4 discrete actions: move forward (0), turn right (1), turn around 180 (2), turn left (3) 
        self.action_space = spaces.Discrete(4)

        if self.n_agents > 1:
            self._agents = [Agent(pos=self.maze.start.copy(), facing=180.0) for i in range(self.n_agents)]
        else:
            self._agents = Agent(pos=self.maze.start.copy(), facing=180.0)

        self.state = self.get_state() 
        self.done = False

        if self.save_robot_path:
            #array to record the path of the agents through the maze
            if self.n_agents > 1:
                self.robot_path = [[self.agents[i].pos.copy()] for i in range(self.n_agents)]
            else:
                self.robot_path = [self.agents.pos.copy()]

    #-----------------------------------------------------------------------------------
    # Properties
    #-----------------------------------------------------------------------------------

    @property
    def maze(self):
        return self._maze

    @property
    def is_render(self):
        return self._is_render

    @property
    def n_agents(self):
        return self._n_agents

    @property
    def save_robot_path(self):
        return self._save_robot_path

    @property
    def agents(self):
        return self._agents

    @property
    def maze_render(self):
        return self._maze_render

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, val: np.ndarray):
        if not isinstance(val, np.ndarray):
            raise TypeError("State must be an array.")
        self._state = val

    @property
    def done(self):
        return self._done

    @done.setter
    def done(self, val: bool):
        if not isinstance(val, bool):
            TypeError("Done must be a boolean value.")
        self._done = val

    @property
    def robot_path(self):
        return self._robot_path

    @robot_path.setter
    def robot_path(self, val: list):
        if not isinstance(val, list):
            raise TypeError("Robot path must be a list of states.")
        self._robot_path = val

    #-----------------------------------------------------------------------------------
    # Methods
    #-----------------------------------------------------------------------------------

    def step(self, actions):
        """
            function to step the environment given an action

            action is an int corresponding to the agents action 

            returns a tuple of (reward: int, done: bool) where reward is the reward 
            for the agent and done is whether or not the goal state is reached
        """
        info = {}
        Rs = []

        if self.n_agents > 1:
            #multi agent
            for i in range(self.n_agents):
                #check action is within action space (a valid action)
                assert self.action_space.contains(actions[i]), "Invalid Action"

                #init reward to 0
                R = 0

                if actions[i] == 0:
                    R = -1
                    vel = self.agents[i].move()

                    #Check for collision with walls if move carried out
                    for wall in self.maze.walls:
                        if self.agents[i].pos[0] + vel[0] == wall[0] and self.agents[i].pos[1] + vel[1] == wall[1]:
                            R = -50
                            vel = (0, 0)
                            break

                    #Update agents position
                    self.agents[i].pos[0] += vel[0]
                    self.agents[i].pos[1] += vel[1]

                    if self.is_render:
                        self.maze_render.update(self.agents[i].pos, i)

                    if self.save_robot_path:
                        self.robot_path[i].append(self.agents[i].pos.copy())

                elif actions[i] == 1:
                    R = -1
                    self.agents[i].rotate(90)
                elif actions[i] == 2:
                    R = -1
                    self.agents[i].rotate(180)
                elif actions[i] == 3:
                    R = -1
                    self.agents[i].rotate(270)
        
                #Check for goal state
                if self.agents[i].pos[0] == self.maze.goal[0] and self.agents[i].pos[1] == self.maze.goal[1]:
                    self.done = True
                    R = 500
            
                Rs.append(R)
        else:
            #single agent
            #check action is within action space (a valid action)
            assert self.action_space.contains(actions), "Invalid Action"

            #init reward to 0
            R = 0

            if actions == 0:
                R = -1
                vel = self.agents.move()

                #Check for collision with walls if move carried out
                for wall in self.maze.walls:
                    if self.agents.pos[0] + vel[0] == wall[0] and self.agents.pos[1] + vel[1] == wall[1]:
                        R = -50
                        vel = (0, 0)
                        break

                    #Update agents position
                self.agents.pos[0] += vel[0]
                self.agents.pos[1] += vel[1]

                if self.is_render:
                    self.maze_render.update(self.agents.pos)

                if self.save_robot_path:
                    self.robot_path.append(self.agents.pos.copy())

            elif actions == 1:
                R = -1
                self.agents.rotate(90)
            elif actions == 2:
                R = -1
                self.agents.rotate(180)
            elif actions == 3:
                R = -1
                self.agents.rotate(270)
        
            #Check for goal state
            if self.agents.pos[0] == self.maze.goal[0] and self.agents.pos[1] == self.maze.goal[1]:
                self.done = True
                R = 500
            
            Rs.append(R)
    
        if self.save_robot_path:
            info["robot_path"] = self.robot_path

        return self.get_state(), Rs, self.done, info

    def reset(self):
        """
            function to reset env to initial state

            returns the state
        """
        #reset agent(s)
        if self.n_agents > 1:
            for i in range(self.n_agents):
                self.agents[i].pos = self.maze.start.copy()
                self.agents[i].facing = 180.0
        else:
            self.agents.pos = self.maze.start.copy()
            self.agents.facing = 180.0
        
        #reset done
        self.done = False

        #reset render
        if self.is_render:
            if self.n_agents > 1:
                for i in range(self.n_agents):
                    self.maze_render.update(self.agents[i].pos, i)
            else:
                self.maze_render.update(self.agents.pos)

            self.maze_render.draw()

        #reset robot saved path through maze
        if self.n_agents > 1:
            if self.save_robot_path:
                self.robot_path = [[self.agents[i].pos.copy()] for i in range(self.n_agents)]
        else:
            self.robot_path = [self.agents.pos.copy()]

        return self.get_state()

    def render(self, mode='human'):
        """
            function to draw objects to screen and exit program on window close
        """
        if self.is_render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    self.maze_render.quit()
                    sys.exit()

            self.maze_render.draw()

    def close(self):
        if self.is_render:
            self.maze_render.quit()

    def get_state(self):
        """
            function to get the current state of the agent within the environment

            returns an array of [foward_dist, left_dist, right_dist] where each is the distance to the nearest wall in that direction
        """
        states = []
        
        if self.n_agents > 1:
            for i in range(self.n_agents):
                dists = self._get_wall_dists(self.agents[i])
                states.append(np.array([forward, left, right], dtype=np.float32))
        else:
            states = self._get_wall_dists(self.agents)
            
        return states

    def _get_wall_dists(self, agent: Agent):
        """
            function to get the distance to the nearest walls from an agent

            agent is the agent whose distance from walls is being calculated
        """
        #initiliase dists in each direction to the maximum possible value
        dists = [self.maze.size[1], self.maze.size[0], self.maze.size[1], self.maze.size[0]]

        for wall in self.maze.walls:
            #dists of 0 are impossible as would be on top of agent
            dist_y = 0
            dist_x = 0

            #calculate distance of each wall in x and y direction
            if agent.pos[0] == wall[0]:
                dist_y = wall[1] - agent.pos[1]
            if agent.pos[1] == wall[1]:
                dist_x = wall[0] - agent.pos[0]
            
            #if distance to wall is shorter than previous calculated then replace distance
            if dist_y > 0 and dist_y < dists[2]:
                dists[2] = dist_y 
            elif dist_y < 0 and -dist_y < dists[0]:
                dists[0] = -dist_y 

            if dist_x > 0 and dist_x < dists[1]:
                dists[1] = dist_x 
            elif dist_x < 0 and -dist_x < dists[3]:
                dists[3] = -dist_x 

        forward = dists[int(agent.facing / 90)]
        left = dists[(int(agent.facing / 90) + 3) % 4]
        right = dists[(int(agent.facing / 90) + 1) % 4]

        return np.array([forward, left, right], dtype=np.float32)

    def save_maze(self, path):
        """
            function to save maze to a pickel file

            path is the path to the directory where the file will be saved
        """
        #make directory if not already exists
        if not os.path.isdir(path):
            os.makedirs(path)

        with open(f'{path}/maze.pkl', "wb") as handle:
            pickle.dump(self.maze, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_maze(self, path):
        """
            function to load a maze into the environment from a picklefile

            path is the path to the directory where the file to be loaded is

            returns the Maze object of the loaded maze
        """
        if os.path.isfile(f'{path}/maze.pkl'):
            with open(f'{path}/maze.pkl', "rb") as handle:
                maze = pickle.load(handle)
        else:
            raise FileNotFoundError("Maze file not found at given path.")

        return maze

#Testing
if __name__ == '__main__':
    env = MazeEnv(is_render=True, save_robot_path=True, save_maze_path="./test", load_maze_path="./test")
    m1 = env.maze
    size = m1.size

    print(f'{size[0]} x {size[1]} maze, start at {m1.start}, goal at {m1.goal}')
    
    env.reset()
    for _ in range(10000):
        observation, reward, done, info = env.step(env.action_space.sample())
        env.render()
        
        if done == True:
            print("Done")
            print(info["robot_path"])
            break

    env.close()


