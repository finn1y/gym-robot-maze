# gym-robot-maze

A maze environment designed for use with [open-ai gym](https://gym.openai.com/), as first suggested by Brockman et al in [[1]](#1)

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


