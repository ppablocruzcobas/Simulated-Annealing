
__authors__ = ["Melissa", "Ronaldo", "Pedro Pablo"]

from heuristic import *
import numpy as np

def create_params(dim):
    Q = np.random.random_sample((dim, dim))
    p = np.random.random_sample(dim)
    a = -1 + 2 * np.random.random_sample(dim)
    b = 4 + 3 * np.random.random_sample(dim)
    
    print()
    print("-------------------------------------------------------------------")
    print()
    print("Dimension", dim)
    print()
    np.savetxt("Q" + str(dim) + ".txt", np.matrix(Q), fmt='%1.4f')
    np.savetxt("p" + str(dim) + ".txt", np.matrix(p), fmt='%1.4f')
    np.savetxt("a" + str(dim) + ".txt", np.matrix(a), fmt='%1.4f')
    np.savetxt("b" + str(dim) + ".txt", np.matrix(b), fmt='%1.4f')

    return Q, p, a, b

    
if __name__ == "__main__":
    Q, p, a, b = create_params(5)
    energies = []

    for i in range(10):
        q_problem = QuadricIntegerProblem(Q, p, list(a), list(b))
        f_state, f_energy = q_problem.anneal()
        energies.append(f_energy)
        np.savetxt("state" + str(len(p)) + ".txt",
                   np.matrix(f_state), fmt='%1.4f')
        
    print()
    print("-------------------------------------------------------------------")
    print("Energies =", energies)

