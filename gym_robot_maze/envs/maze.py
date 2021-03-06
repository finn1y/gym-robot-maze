#!/usr/bin/env python3

import numpy as np

class Maze():
    """
        Maze class generates a random maze with minimum size 20x20 using recursive division algorithm
    """
    def __init__(self, max_size=[20, 20]):
        self._width = np.random.randint(10, high=max_size[0])
        self._height = np.random.randint(10, high=max_size[1])
        self.generate_walls()
        self._start = np.array([1, 1])
        self._goal = np.array([self.width - 2, self.height - 2])

    #-----------------------------------------------------------------------------------
    # Properties
    #-----------------------------------------------------------------------------------

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def start(self):
        return self._start

    @property
    def goal(self):
        return self._goal

    @property
    def walls(self):
        return self._walls

    @walls.setter
    def walls(self, val: np.ndarray):
        if not isinstance(val, np.ndarray):
            raise TypeError("Walls must be an array of wall coordinates.")
        self._walls = val

    @property
    def holes(self):
        return self._holes

    @holes.setter
    def holes(self, val: np.ndarray):
        if not isinstance(val, np.ndarray):
            raise TypeError("Holes must be an array of hole coordinates.")
        self._holes = val

    @property
    def size(self):
        return np.array([self.width, self.height])

    #-----------------------------------------------------------------------------------
    # Methods
    #-----------------------------------------------------------------------------------

    def generate_walls(self):
        """
            function to generate walls within the maze
            uses recursive division algorithm
        """
        self.walls = np.zeros((1, 2))
        self.holes = np.zeros((1, 2))

        #Place walls around the edge of the maze
        for i in range(self.width):
            self.walls = np.append(self.walls, np.array([[i, 0]]), axis=0)
            self.walls = np.append(self.walls, np.array([[i, self.height - 1]]), axis=0)

        for i in range(1, self.height - 1):
            self.walls = np.append(self.walls, np.array([[0, i]]), axis=0)
            self.walls = np.append(self.walls, np.array([[self.width - 1, i]]), axis=0)

        cells = np.array([[[1, 1], [self.width - 2, self.height - 2]]])
        
        while np.shape(cells)[0] > 0:
            div_cells = self.div_cell(cells[0][0], cells[0][1])
            for j in range(2):
                #Do not divide cells which have one dimension less than 3
                #This gives a resolution of 1, i.e. most passages will be 1 wide or tall
                if div_cells[j][1][0] > 2 or div_cells[j][1][1] > 2:
                    cells = np.append(cells, [div_cells[j]], axis=0)
            
            cells = np.delete(cells, 0, axis=0)


    #algorithm based on the recursive division algorithm as described by Buck in:
    #https://weblog.jamisbuck.org/2011/1/12/maze-generation-recursive-division-algorithm
    def div_cell(self, coord, size):
        """
            function to divide a cell into two smaller subcells by placing a wall randomly within it
            and joining the subcells with a hole randomly placed in the wall

            coord is an array of the [x, y] coordinates of the cell within the maze

            size is an array of the [width, height] of the cell

            returns a tuple of the generated subcells each subcell is a 2D array of [[coordinates], [size]]
        """
        #Place the wall vertically in the cell if it is wider than it is tall
        if size[1] < size[0]:
            VERTICAL = 1 #True
        elif size[1] > size[0]:
            VERTICAL = 0 #False -> HORIZONTAL
        else:
            #If the cell is a square place the wall in a random orientation
            VERTICAL = np.random.randint(0, high=2)

        wall_x = np.random.randint(coord[0] + 1, high=(coord[0] + size[0] - 1)) if VERTICAL else coord[0]
        wall_y = coord[1] if VERTICAL else np.random.randint(coord[1] + 1, high=(coord[1] + size[1] - 1))

        hole_x = wall_x if VERTICAL else np.random.randint(wall_x, high=wall_x + size[0])
        hole_y = np.random.randint(wall_y, high=(wall_y + size[1])) if VERTICAL else wall_y

        if VERTICAL:
            wall_coords = np.array([[wall_x, wall_y + i] for i in range(size[1] + 1) if (wall_y + i) != hole_y])
            subcell_0 = np.array([coord, [wall_x - coord[0], size[1]]])
            subcell_1 = np.array([[wall_x + 1, wall_y], [(coord[0] + size[0] - 1) - wall_x, size[1]]])
        else:
            wall_coords = np.array([[wall_x + i, wall_y] for i in range(size[0] + 1) if (wall_x + i) != hole_x])
            subcell_0 = np.array([coord, [size[0], wall_y - coord[1]]])
            subcell_1 = np.array([[wall_x, wall_y + 1], [size[0], (coord[1] + size[1] - 1) - wall_y]])

        for wall_coord in wall_coords:
            place_wall = True
            
            #Check if wall to be placed will block an old hole
            for hole in self.holes:
                if np.linalg.norm(wall_coord - hole) <= 1:
                    place_wall = False
                    
            #Only place walls that will not block old holes
            if place_wall:
                self.walls = np.append(self.walls, [wall_coord], axis=0)

        self.holes = np.append(self.holes, [[hole_x, hole_y]], axis=0)

        return (subcell_0, subcell_1)

#Testing
if __name__ == "__main__":
    m1 = Maze()
    size = m1.size

    #Array to print to consol for visual representation of maze
    array = np.zeros((size[1], size[0]), dtype=int)
    array[m1.start[1]][m1.start[0]] = 2
    array[m1.goal[1]][m1.goal[0]] = 3
    for wall in m1.walls:
        array[int(wall[1])][int(wall[0])] = 1

    print(f'{size[0]} x {size[1]} maze, start at {m1.start}, goal at {m1.goal}')
    print(array)
