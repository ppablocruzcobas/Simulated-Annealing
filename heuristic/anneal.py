from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


__authors__ = ["Melissa", "Ronaldo", "Pedro Pablo"]


import abc
import copy
import sys
import time
import numpy as np


class Annealer(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, initial_state=None,
                 params_auto=True, t_max=1e+10, t_min=1e-4,
                 max_iters=10000, updates=1000):
        self.params_auto = params_auto
        self.t_max = t_max
        self.t_min = t_min
        self.max_iters = max_iters
        self.updates = updates

        # Defaults
        self.copy_strategy = 'deepcopy'

        # Placeholders
        self.best_state = None
        self.best_energy = None

        if initial_state is not None:
            self.state = self.copy_state(initial_state)
        else:
            raise ValueError('No valid value supplied for `initial_state`')

    def round_figures(self, x, n):
        """
        Returns x rounded to n significant figures.
        """
        return round(x, n)

    def time_string(self, seconds):
        """
        Returns time in seconds as a string formatted HHHH:MM:SS.
        """
        s = int(round(seconds))  # round to nearest second
        h, s = divmod(s, 3600)   # get hours and remainder
        m, s = divmod(s, 60)     # split remainder into minutes and seconds
        return '%4i:%02i:%02i' % (h, m, s)

    def copy_state(self, state):
        """
        Returns an exact copy of the provided state
        Implemented according to self.copy_strategy, one of

        * deepcopy: use copy.deepcopy (slow but reliable)
        * slice: use list slices (faster but only works if state is list-like)
        * method: use the state's copy() method
        """
        if self.copy_strategy == 'deepcopy':
            return copy.deepcopy(state)
        elif self.copy_strategy == 'slice':
            return state[:]
        elif self.copy_strategy == 'method':
            return state.copy()
        else:
            return None

    def get_state(self):
        """
        Returns the current state of the system.
        """
        return self.state

    def get_state_as_array(self):
        """
        Returns the current state of the system as an array.
        to .
        """
        return np.array(self.state)

    def update(self, step, T, E, accepts, improves):
        """
        Default update, outputs to stderr.

        Prints the current temperature, energy, acceptance rate,
        improvement rate, elapsed time.

        The acceptance rate indicates the percentage of moves since the last
        update that were accepted by the Metropolis algorithm.  It includes
        moves that decreased the energy, moves that left the energy
        unchanged, and moves that increased the energy yet were reached by
        thermal excitation.

        The improvement rate indicates the percentage of moves since the
        last update that strictly decreased the energy. At high
        temperatures it will include both moves that improved the overall
        state and moves that simply undid previously accepted moves that
        increased the energy by thermal excititation.  At low temperatures
        it will tend toward zero as the moves that can decrease the energy
        are exhausted and moves that would increase the energy are no longer
        thermally accessible.
        """
        elapsed = time.time() - self.start
        if step == 0:
            print(' Temperature        Energy        Accept        Improve      Elapsed',
                  file=sys.stderr)
        print('\r%12.4f  %12.4f %12.0f%%  %12.0f%%   %s' %
              (T, E, 100 * accepts, 100 * improves, self.time_string(elapsed)),
              file=sys.stderr, end="\r")
        sys.stderr.flush()

    def find_best_parameters(self, steps=1000):
        """
        Explores the annealing landscape and
        estimates optimal temperature settings.
        """
        def simulate(T, steps):
            """
            Anneals a system at constant temperature and returns the 
            rate of acceptance and the rate of improvement.
            """
            E = self.energy()
            prev_energy = E
            accepts, improves = 0, 0
            for _ in range(steps):
                self.state = self.neighbour()
                E = self.energy()
                dE = prev_energy - E
                if self.probability(dE, T) >= np.random.random():
                    accepts += 1
                    if dE > .0:
                        improves += 1                    
                    prev_energy = E
                else:
                    E = prev_energy
            return float(accepts) / steps, float(improves) / steps

        T = .0
        E = self.energy()

        while T <= 1e-4:
            self.state = self.neighbour()
            T = abs(self.energy() - E)

        # Search for t_max - a temperature that gives 99% acceptance
        acceptance, improvement = simulate(T, steps)
        while acceptance < .99 and T < 1e+10:
            T *= 1.15
            acceptance, improvement = simulate(T, steps)
        self.t_max = T

        # Search for t_min - a temperature that gives 0% improvement
        acceptance, improvement = simulate(T, steps)
        while improvement > .0 and T > 1e-4:
            T /= 2
            acceptance, improvement = simulate(T, steps)
        self.t_min = T

        return self.t_max, self.t_min

    def anneal(self):
        """
        Minimizes the energy of a system by Simulated Annealing.

        Returns
        (state, energy): the best state and energy found.
        """
        step = 0
        self.start = time.time()

        print("-------------------------------------------------------------------")
        
        if self.params_auto is True:
            print()
            print("Calculating optimal parameters...")
            self.find_best_parameters()

        print()
        print("Tmax = %10.4f" % (self.t_max))
        print("Tmin = %10.4f" % (self.t_min))
        print("Iterations =", (self.max_iters))
        print()
            
        E = self.energy()

        prev_state = self.copy_state(self.state)
        prev_energy = E

        self.best_state = self.copy_state(self.state)
        self.best_energy = E

        accepts, improves = 0, 0

        if self.updates > 0:
            update_every = self.max_iters / self.updates
            self.update(step, self.t_max, E, 1, 1)

        # Attempt moves to new states
        while step < self.max_iters:
            step += 1
            T = self.temperature(step, self.max_iters,
                                 self.t_max, self.t_min)

            for i in range(100):
                self.state = self.neighbour()
                E = self.energy()
                dE = prev_energy - E
                if self.probability(dE, T) >= np.random.random():
                    # Accept new state and compare to best state
                    accepts += 1
                    prev_state = self.copy_state(self.state)
                    prev_energy = E
                    if dE > 0:
                        improves += 1
                    if E < self.best_energy:
                        self.best_state = self.copy_state(self.state)
                        self.best_energy = E
                else:
                    # Restore previous state
                    self.state = self.copy_state(prev_state)
                    E = prev_energy
            if self.updates > 1:
                if (step // update_every) > ((step - 1) // update_every):
                    self.update(step, T, E,
                                float(accepts / 100) / update_every,
                                float(improves / 100) / update_every)
                    accepts, improves = 0, 0

        print()
        print()
        print("Best Energy =", self.best_energy)
        print()
        print("-------------------------------------------------------------------")

        # Return best state and energy
        return self.best_state, self.best_energy


    def probability(self, dE, T):
        """
        Returns the probability of accept 
        """
        if dE >= 0:
            return 1.0
        else:
            return np.exp(dE / T)

    @abc.abstractmethod
    def neighbour(self):
        """
        Create a state change.
        Has to be implemented in the problem.
        """
        pass

    @abc.abstractmethod
    def temperature(self, step, steps, t_max, t_min):
        """
        Calculate the temperature of the system.
        Has to be implemented in the problem.
        """
        pass

    @abc.abstractmethod
    def energy(self):
        """
        Calculate state's energy.
        Has to be implemented in the problem.
        """
        pass


