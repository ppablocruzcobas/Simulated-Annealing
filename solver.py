
import numpy as np


class Heuristic:
    def __init__(self, problem):
        self.p = problem

    def solve(self):
        # generate new 'x', evaluate and compare with current one
        # that's all
        self.p.setX([])


class Problem:
    def __init__(self, C, p):
        self.C = C
        self.p = p
        self.x = np.zeros((p, 1))

    def __call__(self):
        return self.x.T * self.C * self.x + self.p.T * self.x

    def setX(self, x):
        self.x = x

    def getX(self):
        return self.x