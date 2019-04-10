
__authors__ = ["Melissa", "Ronaldo", "Pedro Pablo"]

from heuristic.anneal import Annealer
import random as r
import numpy as np


class QuadricProblem(Annealer):
    def __init__(self, Q, p, l_limit=None, u_limit=None):
        self.l_limit = np.array(l_limit)
        self.u_limit = np.array(u_limit)
        self.Q = Q
        self.p = p
        
        super(QuadricProblem, self).__init__(
            initial_state=list((self.l_limit + self.u_limit) / 2))

    def neighbour(self, size=25, delta=1e-2):
        """
        Generate a set of length `size` elements in the square centered
        at current `state` and with side length `delta`
        """
        state = self.get_state_as_array()
        result = []
        for i in range(size):
            nb = state + delta * (-1 + 2
                               * np.random.random_sample(len(state)))
            for j in range(len(nb)):
                if nb[j] < self.l_limit[j]:
                    nb[j] = self.l_limit[j] + delta * \
                        np.random.random_sample()
                elif nb[j] > self.u_limit[j]:
                    nb[j] = self.u_limit[j] - delta * \
                        np.random.random_sample()
                result.append(nb)
        return r.choice(result)

    def temperature(self, step, steps, t_max, t_min):
        # alpha = np.power(self.t_min / self.t_max, 1 / self.max_iters)
        # return t_max * alpha ** step
        k = -np.log(t_max / t_min)
        return t_max * np.exp(k * step / steps)

    def energy(self):
        state = self.get_state_as_array()
        return np.dot(np.dot(state, self.Q), state.T) + np.dot(self.p, state.T)
