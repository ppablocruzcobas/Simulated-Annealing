
__authors__ = ["Melissa", "Ronaldo", "Pedro Pablo"]

from heuristic import QuadricBinaryProblem, QuadricIntegerProblem
import numpy as np
import pylab as plt
import os


class BaseTest:
    """
    Base Class that implements basic methods for testing
    `dims` = list of dimensions
    `tests` = how many times solve each problem
    `t_type` = type of problem ('binary' or 'integer')
    """
    def __init__(self, dims, tests, t_type):
        self.dims = dims
        self.tests = tests
        self.t_type = t_type

        # If folders doesn't exists, then create them
        for dim in dims:
            os.makedirs(self.path(dim), exist_ok=True)

    def path(self, dim):
        """
        According to `dim`, it returns the path where the data will be saved.
        Example:
        The data of a binary problem of dimension N will be saved in
        `data/N/binary/`
        """
        return "data/" + str(dim) + "/" + self.t_type + "/"

    def create_params(self, dim):
        """
        Returns a tuple of parameters for the problem according to `dim`
        (dimension)
        `Q` = a matrix of shape `dim` x `dim`
                values between -5 and 5 randomly generated
        `p` = a vector of length `dim`
                values between -5 and 5 randomly generated
        `a` = lower limit (only used in the integer problem)
                values between -5 and 5 randomly generated
        `b` = upper limit (only used in the integer problem)
                values between 10 and 20 randomly generated
        """
        Q = -5 + 10 * np.random.random_sample((dim, dim))
        p = -5 + 10 * np.random.random_sample(dim)
        if self.t_type == 'integer':
            a = -5 + 10 * np.random.random_sample(dim)
            b = 10 + 10 * np.random.random_sample(dim)

        print()
        print("-------------------------------------------------------------------")
        print()
        print("Dimension", dim)
        print()
        np.savetxt(self.path(dim) + "Q.txt", np.matrix(Q), fmt='%1.4f')
        np.savetxt(self.path(dim) + "p.txt", np.matrix(p), fmt='%1.4f')
        if self.t_type == 'integer':
            np.savetxt(self.path(dim) + "a.txt", np.matrix(a), fmt='%1.4f')
            np.savetxt(self.path(dim) + "b.txt", np.matrix(b), fmt='%1.4f')

        if self.t_type == 'integer':
            return Q, p, a, b
        elif self.t_type == 'binary':
            return Q, p

    def run(self):
        """
        Execute the tests and save all results (including image) in corresping
        path
        """
        for dim in self.dims:
            if self.t_type == 'integer':
                Q, p, a, b = self.create_params(dim)
                init = np.floor(b)
                q_problem = QuadricIntegerProblem(Q, p, list(a), list(b),
                                                  initial_state=init)
            elif self.t_type == 'binary':
                Q, p = self.create_params(dim)
                init = list(np.zeros(dim))
                q_problem = QuadricBinaryProblem(Q, p,
                                                 initial_state=init)
            t_max, t_min, init = q_problem.find_best_parameters()
            
            states = []
            energies = []

            for i in range(self.tests):
                best_state, best_energy = q_problem.anneal(
                    initial_state=init)
                states.append(best_state)
                energies.append(best_energy)
                
            np.savetxt(self.path(dim) + "states.txt",
                           np.matrix(states), fmt='%2.0f')
            np.savetxt(self.path(dim) + "energies.txt",
                           np.matrix(energies))

            st_dev = np.sqrt(np.var(energies))
            mean = np.mean(energies)
            median = np.median(energies)

            np.savetxt(self.path(dim) + "results.txt",
                       [st_dev, mean, median])
            
            # Plot and save graphic
            x = range(1, 1 + self.tests)
            plt.figure("results (dim = %i)" % dim)
            plt.axhline(y=mean, color='r', linestyle='-', label='mean')
            plt.scatter(x, energies, marker='*', c='b', label='values')
            plt.legend()
            plt.savefig(self.path(dim) + "results.png", dpi=300)

            print()
            print("St. Dev.: %f      Mean: %f      Median: %f" %
                  (st_dev, mean, median))
            print("-------------------------------------------------------------------")


class BinaryTest(BaseTest):
    """
    Class to create and execute binary tests.
    """
    def __init__(self, dims, tests):
        super(BinaryTest, self).__init__(dims, tests, 'binary')


class IntegerTest(BaseTest):
    """
    Class to create and execute integer tests.
    """
    def __init__(self, dims, tests):
        super(IntegerTest, self).__init__(dims, tests, 'integer')
