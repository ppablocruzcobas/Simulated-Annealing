from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


__authors__ = ["Melissa", "Ronaldo", "Pedro Pablo"]


import abc
import copy
import datetime
import math
import pickle
import random
import signal
import sys
import time
import numpy as np


class Annealer(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, initial_state=None, t_max=2500, t_min=1.5,
                 max_iters=50000, updates=500):
        self.t_max = t_max
        self.t_min = t_min
        self.max_iters = max_iters
        self.updates = updates

        # Defaults
        self.copy_strategy = 'deepcopy'
        self.save_state_on_exit = True

        # Placeholders
        self.best_state = None
        self.best_energy = None

        if initial_state is not None:
            self.state = self.copy_state(initial_state)
        else:
            raise ValueError('No valid value supplied for `initial_state`')

    def time_string(self, seconds):
        """Returns time in seconds as a string formatted HHHH:MM:SS."""
        s = int(round(seconds))  # round to nearest second
        h, s = divmod(s, 3600)   # get hours and remainder
        m, s = divmod(s, 60)     # split remainder into minutes and seconds
        return '%4i:%02i:%02i' % (h, m, s)

    def copy_state(self, state):
        """Returns an exact copy of the provided state
        Implemented according to self.copy_strategy, one of

        * deepcopy : use copy.deepcopy (slow but reliable)
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
            raise RuntimeError('No implementation found for ' +
                               'the self.copy_strategy "%s"' %
                               self.copy_strategy)

    def get_state(self):
        return self.state

    def get_state_as_array(self):
        return np.array(self.state)

    def save_state(self, f_name='state.txt'):
        """Saves state to file"""
        try:
            np.savetxt(f_name, self.state)
            print()
            print('***********************************************************')
            print("State saved to file %s" % f_name)
        except:
            raise RuntimeWarning("Cannot save state to file %s" % f_name)

    def update(self, *args, **kwargs):
        """Wrapper for internal update.

        If you override the self.update method,
        you can chose to call the self.default_update method
        from your own Annealer.
        """
        self.default_update(*args, **kwargs)

    def default_update(self, step, T, E):
        """Default update, outputs to stderr.

        Prints the current temperature, energy, acceptance rate,
        improvement rate, elapsed time, and remaining time.

        The acceptance rate indicates the percentage of moves since the last
        update that were accepted by the Metropolis algorithm.  It includes
        moves that decreased the energy, moves that left the energy
        unchanged, and moves that increased the energy yet were reached by
        thermal excitation.

        The improvement rate indicates the percentage of moves since the
        last update that strictly decreased the energy.  At high
        temperatures it will include both moves that improved the overall
        state and moves that simply undid previously accepted moves that
        increased the energy by thermal excititation.  At low temperatures
        it will tend toward zero as the moves that can decrease the energy
        are exhausted and moves that would increase the energy are no longer
        thermally accessible."""

        elapsed = time.time() - self.start
        if step == 0:
            print(' Temperature        Energy       Elapsed',
                  file=sys.stderr)
        print('\r%12.5f  %12.2f    %s' %
              (T, E, self.time_string(elapsed)), file=sys.stderr, end="\r")
        sys.stderr.flush()

    def anneal(self):
        """Minimizes the energy of a system by simulated annealing.

        Parameters
        state : an initial arrangement of the system

        Returns
        (state, energy): the best state and energy found.
        """
        step = 0
        self.start = time.time()

        # Precompute factor for exponential cooling from Tmax to Tmin
        if self.t_min <= 0.0:
            raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')
        t_factor = -math.log(self.t_max / self.t_min)

        # Note initial state
        T = self.t_max
        E = self.energy()

        prev_state = self.copy_state(self.state)
        prev_energy = E

        self.best_state = self.copy_state(self.state)
        self.best_energy = E

        if self.updates > 0:
            update_every = self.max_iters / self.updates
            self.update(step, T, E)

        # Attempt moves to new states
        while step < self.max_iters:
            step += 1
            T = self.t_max * math.exp(t_factor * step / self.max_iters)
            self.state = self.neighbour()
            E = self.energy()
            dE = E - prev_energy
            if dE > 0.0 and 100 * math.exp(-dE / T) / self.max_iters < random.random():
                # Restore previous state
                self.state = self.copy_state(prev_state)
                E = prev_energy
            else:
                # Accept new state and compare to best state
                prev_state = self.copy_state(self.state)
                prev_energy = E
                if E < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = E
            if self.updates > 1:
                if (step // update_every) > ((step - 1) // update_every):
                    self.update(step, T, E)

        self.state = self.copy_state(self.best_state)
        if self.save_state_on_exit:
            self.save_state()

        # Return best state and energy
        return self.best_state, self.best_energy

    @abc.abstractmethod
    def neighbour(self):
        """Create a state change"""
        pass

    @abc.abstractmethod
    def energy(self):
        """Calculate state's energy"""
        pass
