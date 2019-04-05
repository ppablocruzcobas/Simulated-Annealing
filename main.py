
__authors__ = ["Melissa", "Ronaldo", "Pedro Pablo"]

import numpy as np
from solver import Heuristic, Problem


if __name__ == "__main__":
    # defines the problem
    C = np.matrix([[2, 1, 3], [1, -1, 0], [1, 3, 2]])
    p = np.atrix([1, 0, -1])
    problem = Problem(C, p)
    # defines the euristic to be used to solve the problem
    h = Heuristic(50, 1000, 50)
    # testing
    h.solve(problem)
