
__authors__ = ["Melissa", "Ronaldo", "Pedro Pablo"]

import numpy as np


class Heuristic:
    """
    `t0` = initial temperature \n
    `max_iters` = maximun number of iterations (stop criteria) \n
    `t_iters` = number of iterations to wait before lower temperature
    """
    def __init__(self, t0, max_iters, t_iters):
        self.t0 = t0
        self.max_iters = max_iters
        self.t_iters = t_iters

    def neightboor(self, x):
        return x

    def solve(self, problem):
        # generate new 'x', evaluate and compare with current one
        # until not improvement is appreciated
        # that's all
        if problem.get_obj() == 'min':
            value = np.inf
        else:
            value = -np.inf
            
        constrains = problem.get_constrains()

        for i in range(self.max_iters):
            satisfied = True
            s = np.random.random_sample(problem.get_dim())
            for constrain in constrains:
                satisfied = satisfied and constrain(s)
            if satisfied == True:
                v = problem(s)
                if problem.get_obj() == 'min':
                    if problem(v) < value:
                        value = problem(v)
                        problem.set_solution(s)
                        print(s)
                else:
                    if problem(v) > value:
                        value = problem(v)
                        problem.set_solution(s)
                        print(s)
        print('x = ', problem.get_solution())
        print('obj = ', problem(problem.get_solution()))
