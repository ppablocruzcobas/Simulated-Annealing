
__authors__ = ["Melissa", "Ronaldo", "Pedro Pablo"]

from heuristic.graphic import Application

import numpy as np
import sys


class Heuristic:
    """
    `t0` = initial temperature \n
    `max_iters` = maximun number of iterations (stop criteria) \n
    `t_iters` = number of iterations to wait before lower temperature
    """
    def __init__(self, t0=50, max_iters=1000, t_iters=50, delta=1e-2):
        self._t0 = t0
        self._max_iters = max_iters
        self._t_iters = t_iters
        self._delta = delta
        self._iters = 0

        self._app = Application(sys.argv, self._update)

    def _neighbour(self, x):
        return x + self._delta * (2
                * np.random.random_sample(self._problem.get_dim()) - 1)

    def _T(self):
        return 1

    def _E(self, state):
        pass

    def _P(self, i, j):
        return (self._E(i) - self._E(j)) / self._T()

    def _update(self):
        if self._iters == self._max_iters:
            sys.exit(0)
        self._iters += 1
        
        self._s = self._neighbour(self._s)

        self._app.append(self._T(), self._problem(self._s))
        
        satisfied = True
        for constrain in self._constrains:
            satisfied = satisfied and constrain(self._s)
            
        if satisfied == True:
            v = self._problem(self._s)
            if self._problem.get_obj() == 'min':
                if v < self._value:
                    self._value = v
                    self._problem.set_solution(self._s)
                    if self._verbose:
                        print(self._s)
            elif self._problem.get_obj() == 'max':
                if v > self._value:
                    self._value = v
                    self._problem.set_solution(self._s)
                    if self._verbose:
                        print(self._s)

    def solve(self, problem, graph=True, verbose=True):
        self._problem = problem
        self._verbose = verbose
        
        if problem.get_obj() == 'min':
            self._value = np.inf
        elif problem.get_obj() == 'max':
            self._value = -np.inf

        self._constrains = problem.get_constrains()
        a = []
        b = []
        for constrain in self._constrains:
            if constrain.has_lower():
                a.append(constrain.get_lower())
            else:
                a.append(-5)
            if constrain.has_upper():
                b.append(constrain.get_upper())
            else:
                b.append(5)

        a = np.matrix(a)
        b = np.matrix(b)
        self._s = a + (b - a) * np.random.random_sample(problem.get_dim())

        if graph:
            self._app.show_graph()
        self._app.exec_()
