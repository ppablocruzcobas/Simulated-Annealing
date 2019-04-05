
__authors__ = ["Melissa", "Ronaldo", "Pedro Pablo"]

import numpy as np


class Problem:
    def __init__(self, C, p):
        self.C = C
        self.p = p
        self.x = np.zeros(len(p))

    def __call__(self, x):
        return (np.dot(np.dot(x, self.C), x.T) + np.dot(self.p, x.T))[0, 0]

    def setX(self, x):
        self.x = x

    def getX(self):
        return self.x
