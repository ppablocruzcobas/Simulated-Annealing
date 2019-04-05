
__authors__ = ["Melissa", "Ronaldo", "Pedro Pablo"]

import numpy as np


"""
`t0` = initial temperature
`max_iters` = maximun number of iterations (stop criteria)
`t_iters` = number of iterations to wait before lower temperature
"""
class Heuristic:
    def __init__(self, t0, max_iters, t_iters):
        self.t0 = t0
        self.max_iters = max_iters
        self.t_iters = t_iters

    def solve(self, problem):
        self.p = problem
        # generate new 'x', evaluate and compare with current one
        # until not improvement is appreciated
        # that's all
        value = 100000
        for i in range(1000000):
            v = self.p(np.random.random_sample(3))
            if self.p(v) < value:
                value = self.p(v)
                self.p.setX(v)
                print(value)
        print(self.p.getX())
