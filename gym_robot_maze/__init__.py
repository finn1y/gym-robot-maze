#!/usr/bin/env python3

from gym.envs.registration import register

register(
    id="RobotMaze-v1",
    entry_point="gym_robot_maze.envs:MazeEnv",
)
