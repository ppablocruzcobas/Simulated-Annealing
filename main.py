
__authors__ = ["Melissa", "Ronaldo", "Pedro Pablo"]

from heuristic import *

import numpy as np


def obj_func(x):
    return (np.dot(np.dot(x, Q), x.T) + np.dot(p, x.T))[0, 0]
    # return np.power(x, 2)
    # return x

def const1(x):
    return x

if __name__ == "__main__":
    DIM = 50
    Q = np.random.random_sample((DIM, DIM))
    p = np.random.random_sample(DIM)
    # Defines the problem
    problem = Problem(obj_func, [Constrain(const1, 0, 1)],
                      dim=DIM, obj='min')
    # Defines the parameters of the heuristic to be used to solve the problem
    h = Heuristic(50, 1000, 50)
    # Testing
    h.solve(problem, graph=True, verbose=False)
    print('x =', problem.get_solution())
