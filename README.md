# gym-robot-maze

A maze environment designed for use with [open-ai gym](https://gym.openai.com/), as first suggested by Brockman et al in [[1]](#1).

## Environment

#### Action  Space

The agent can take one of four discrete actions, summarised in the table below.

Action number | Action
--------------|--------
0             | move forward
1             | rotate right
2             | rotate 180
3             | rotate left

#### Observation Space

The agent recieves an observation as a tupe of (forward_dist, left_dist, right_dist) where each value is the distance to the nearest wall in that direction. This simulates a laser or ultrasonic sensor which can detect the distance to an obstacle in a direction.

Observation numnber | Observation       | Min | Max
--------------------|-------------------|-----|------
0                   | Forward distance  | 0   | inf
1                   | Left distance     | 0   | inf
2                   | Right distance    | 0   | inf

Note: although the max value is technically infinite it is limited to the size of the maze in that direction, as this is the largest value that observation can take within that maze instantiation. E.g. for a 10x10 maze the max value for all ditances is 10.

#### Reward

The agent receives a reward of -1 for every action taken. If the action results in a collision with a wall then the reward is reduced to -50 to deter the agent from colliding with walls. If the agent reaches the goal state then the reward is increased to 500 to stimulate the agent in reaching the goal. This can be summed up in the following equation:

$$R = \begin{cases}
500 & (if s' = s_{goal})\\
-50 & (if s' = s)\\
-1 & (otherwise)
\end{cases}$$

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


