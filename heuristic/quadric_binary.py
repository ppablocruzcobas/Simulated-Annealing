
__authors__ = ["Melissa", "Ronaldo", "Pedro Pablo"]

from heuristic.anneal import Annealer
import random as r
import numpy as np


class QuadricProblemBinary(Annealer):
    """
    Class to solve the quadric problem
    min xTQx + pTx s.t. a <= x <= b; a, b \in R^{n}:
    `Q`: a square matrix
    `p`: a vector
    `l_limit`: the vector of lower limits
    `u_limit`: the vector of upper limits
    """
    def __init__(self, Q, p, l_limit=None, u_limit=None):
        self.l_limit = np.array(l_limit)
        self.u_limit = np.array(u_limit)
        self.Q = Q
        self.p = p
        # Call the constructor of the `Anneal` class with only
        # `initial_state` (mandatory); all others parameters have default values
        super(QuadricProblemBinary, self).__init__(
            initial_state=list((self.l_limit + self.u_limit) / 2))

    def neighbour(self, size=1, delta=None):
        """
        Generate a set of length `size` elements in the square centered
        at current `state` and with side length `delta`
        and returns only one element randomly choiced
        """
        state = self.get_state()
        p = np.random.randint(0, len(state))
        state[p] = 1 - state[p]
        return state
    
    def temperature(self, step, steps, t_max, t_min):
        # alpha = np.power(self.t_min / self.t_max, 1 / self.max_iters)
        # return t_max * alpha ** step
        k = -np.log(t_max / t_min)
        return t_max * np.exp(k * step / steps)

    def energy(self):
        state = self.get_state_as_array()
        return np.dot(np.dot(state, self.Q), state.T) + np.dot(self.p, state.T)
