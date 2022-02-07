# gym-robot-maze

A maze environment designed for use with [open-ai gym](https://gym.openai.com/), as first suggested by Brockman et al in [[1]](#1).

## Environment

#### Action  Space

The agent may choose one of 4 actions: 
  0. move forward
  1. rotate right
  2. rotate 180
  3. rotate left

#### Observation Space

The agent recieves an observation as a tupe of (forward_dist, left_dist, right_dist) where each value is the distance to the nearest wall in that direction. This simulates a laser or ultrasonic sensor which can detect the distance to an obstacle in a direction.

#### Reward

The agent receives a reward of -1 for every action taken. If the action results in a collision with a wall then the reward is reduced to -50 to deter the agent from colliding with walls. If the agent reaches the goal state then the reward is increased to 500 to stimulate the agent in reaching the goal

#### Goal State

The goal state is the end point of the maze, this is the bottom right most point in the maze.

## Maze Generation

The mazes generated use the recursive division algorithm, described by Buck in [[2]](#2). The size is chosen randomly from a 10x10 
grid up to a given max size (defaults to 20x20).

## Install

1. Clone the repo
```
git clone https://github.com/finn1y/gym-robot-maze 
```
2. Install depedencies
```
pip install -r requirements.txt
```
3. Install gym robot maze
```
pip install -e ./gym-robot-maze
```

## References

<a id="1">[1]</a>
G. Brockman, V. Cheung, L. Pettersson et al, "OpenAI Gym", *arXiv:1606.01540v1 [cs.LG]*, 2016. Available: [link](https://arxiv.org/abs/1606.01540) [Accessed 2 Feb 2022]

<a id="2">[2]</a>
J. Buck, “Maze generation: Recursive division”, *weblog.jamisbuck.org*, 2011. Available: [link](http://weblog.jamisbuck.org/2011/1/12/maze-generation-recursive-division-algorithm) [Accessed 2 Feb 2022]


