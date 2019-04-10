
__authors__ = ["Melissa", "Ronaldo", "Pedro Pablo"]

from heuristic import *
import numpy as np

def create_params(dim):
    Q = 3 + 2 * np.random.random_sample((dim, dim))
    p = 1 + 3 * np.random.random_sample(dim)
    a = np.random.randint(5, 7, dim)
    b = np.random.randint(9, 15, dim)
    
    print()
    print("-------------------------------------------------------------------")
    print()
    print("Dimension", dim)
    print()
    np.savetxt("Q" + str(dim) + ".txt", Q)
    np.savetxt("p" + str(dim) + ".txt", p)
    np.savetxt("a" + str(dim) + ".txt", np.matrix(a))
    np.savetxt("b" + str(dim) + ".txt", np.matrix(b))

    return Q, p, a, b

    
if __name__ == "__main__":
    Q, p, a, b = create_params(5)
    energies = []

    for i in range(10):
        q_problem = QuadricProblem01(Q, p, list(a), list(b))
        f_state, f_energy = q_problem.anneal()
        energies.append(f_energy)
        np.savetxt("state" + str(len(p)) + ".txt", np.array(f_state))
        
    print()
    print("-------------------------------------------------------------------")
    print("Energies =", energies)

    # q_problem = create_problem(10)
    # f_state, f_energy = q_problem.anneal()
    # np.savetxt("state_dim10.txt", np.array(f_state))
    # print()
    # print("Best Energy =", f_energy)

    # q_problem = create_problem(50)
    # f_state, f_energy = q_problem.anneal()
    # np.savetxt("state_dim50.txt", np.array(f_state))
    # print()
    # print("Best Energy =", f_energy)
