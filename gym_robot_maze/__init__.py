#!/usr/bin/env python3

from gym_robot_maze.envs.maze import Maze
from gym.envs.registration import register

register(
    id="RobotMaze-v1",
    entry_point="gym_robot_maze.envs:MazeEnv",
)
