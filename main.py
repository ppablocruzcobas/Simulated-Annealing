
__authors__ = ["Melissa", "Ronaldo", "Pedro Pablo"]

import numpy as np

from heuristic import *


def obj_func(x):
    # Q = np.matrix([[2.0, 1.0, 3.0], [1.0, -1.0, 0.0], [1.0, 3.0, 2.0]])
    # p = np.matrix([1.0, 0.0, -1.0])
    # return (np.dot(np.dot(x.T, Q), x) + np.dot(p, x.T))[0, 0]
    return np.power(x, 2)

def const1(x):
    return x

if __name__ == "__main__":
    # Defines the problem
    problem = Problem(obj_func, [Constrain(const1, 0.2, 0.8)], obj='max')
    # Defines the parameters of the heuristic to be used to solve the problem
    h = Heuristic(50, 1000000, 50)
    # Testing
    h.solve(problem)
