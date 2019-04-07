
__authors__ = ["Melissa", "Ronaldo", "Pedro Pablo"]

from heuristic import *

import numpy as np


Q = np.matrix([[2.0, 1.0, 3.0], [1.0, -1.0, 0.0], [1.0, 3.0, 2.0]])
p = np.matrix([1.0, 0.0, -1.0])

def obj_func(x):
    # return (np.dot(np.dot(x, Q), x.T) + np.dot(p, x.T))[0, 0]
    return np.power(x, 2)
    # return x

def const1(x):
    return x

if __name__ == "__main__":
    # Defines the problem
    problem = Problem(obj_func, [Constrain(
        const1, 1, 5)], dim=1, obj='min')
    # Defines the parameters of the heuristic to be used to solve the problem
    h = Heuristic(50, 100, 50)
    # Testing
    h.solve(problem, graph=False, verbose=True)
    # x = np.random.random_sample(3)
    # print(obj_func(x))
