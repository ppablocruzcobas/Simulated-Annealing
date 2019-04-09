
__authors__ = ["Melissa", "Ronaldo", "Pedro Pablo"]

from heuristic import *
import numpy as np
import random as r


class QuadricProblem(Annealer):
    def __init__(self, dim, Q, p, l_limit, u_limit):
        self.l_limit = np.array(l_limit)
        self.u_limit = np.array(u_limit)
        
        super(QuadricProblem, self).__init__(
            initial_state=list((self.l_limit + self.u_limit) / 2),
            t_max=10000, t_min=2.5)

        self.Q = np.random.random_sample((dim, dim))
        self.p = np.random.random_sample(dim)

    def neighbour(self):
        delta = 1e-2
        state = self.get_state_as_array()
        result = []
        for i in range(50):
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

    def energy(self):
        state = self.get_state_as_array()
        return np.dot(np.dot(state, self.Q), state.T) + np.dot(self.p, state.T)


def create_problem(dim):
    Q = np.random.random_sample((dim, dim))
    p = np.random.random_sample(dim)
    a = np.random.randint(-5, 5, dim)
    b = np.random.randint(5, 15, dim)
    
    print()
    print("-------------------------------------------------------------------")
    print()
    print("Problem of dimension", dim)
    print()
    print("Q =", Q.tolist())
    print()
    print("p =", p.tolist())
    print()
    print("a =", a.tolist())
    print()
    print("b =", b.tolist())
    print()
    
    q_problem = QuadricProblem(dim, Q, p, list(a), list(b))
    return q_problem

    
if __name__ == "__main__":
    q_problem = create_problem(5)
    f_state, f_energy = q_problem.anneal()
    print("Best Energy =", f_energy)

    q_problem = create_problem(10)
    f_state, f_energy = q_problem.anneal()
    print("Best Energy =", f_energy)

    q_problem = create_problem(50)
    f_state, f_energy = q_problem.anneal()
    print("Best Energy =", f_energy)
