
__authors__ = ["Melissa", "Ronaldo", "Pedro Pablo"]

import numpy as np


class Constrain:
    """
    Generic class that can be used to modelate any constrain of the form \
    `a <= g(x) <= b`, with a, b, g(x) \in R^{n}. \
    `a` and `b` can be also real values \n
    `func` = g(x) \n
    `lower` = a \n
    `upper` = b \n
    `strict_lower` = if `True` then `lower` not included (a < g(x)). \
                     Default is False (a <= g(x)) \n
    `strict_upper` = if `True` then `upper` not included (g(x) < b). \
                     Default is False (g(x) <= b) \n
    """
    def __init__(self, func, lower=None, upper=None,
                 strict_lower=False, strict_upper=False):
        self._func = func
        self._lower = lower
        self._upper = upper
        self._strict_lower = strict_lower
        self._strict_upper = strict_upper

    def __call__(self, x):
        """
        Returns `True` if the constrain is satisfied, `False` otherwise 
        """
        result = True
        if self._lower is not None:
            if self._strict_lower is True:
                result = result and (self._func(x) >
                                     self._lower).all() == True
            else:
                result = result and (self._func(x) >=
                                     self._lower).all() == True
        if self._upper is not None:
            if self._strict_upper is True:
                result = result and (self._func(x) <
                                     self._upper).all() == True
            else:
                result = result and (self._func(x) <=
                                     self._upper).all() == True
        return result

    def has_lower(self):
        return (self._lower is not None)

    def has_upper(self):
        return (self._upper is not None)

    def get_lower(self):
        return self._lower

    def get_upper(self):
        return self._upper


# ############################################################################ #


class Problem:
    """
    Generic class that can be used to modelate any problem \n
    `obj_func` = objective function to be optimised \n
    `constrains` = a list containing all constrains concerning to the problem \n
    `dim` = dimension of the problem \n
    `obj` = what we want to do with `obj_func` \n
        'min' for minimum
        'max' for maximum
    """
    def __init__(self, obj_func, constrains=None, dim=1, obj='min'):
        self._obj_func = obj_func
        self._constrains = constrains
        self._dim = dim
        self._obj = obj
        self._solution = None

    def __call__(self, x):
        """
        Returns the evaluation of the `obf_func` in the vector/real value x
        """
        return self._obj_func(x)

    def get_constrains(self):
        """
        Returns a list with all the constrains of the problem
        """
        return self._constrains

    def get_dim(self):
        """
        Returns the dimension of the problem
        """
        return self._dim

    def get_obj(self):
        """
        Returns what should be done with `obj_min` ('min' or 'max')
        """
        return self._obj

    def get_solution(self):
        """
        Returns the best solution obtained at the moment of the calling
        """
        return self._solution

    def set_solution(self, solution):
        """
        Update the solution in case of improvement.
        Should be use only by the `Heuristic`  class
        """
        self._solution = solution