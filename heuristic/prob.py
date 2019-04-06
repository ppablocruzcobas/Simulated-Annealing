
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
        self.func = func
        self.lower = lower
        self.upper = upper
        self.strict_lower = strict_lower
        self.strict_upper = strict_upper

    def __call__(self, x):
        """
        Returns `True` if the constrain is satisfied, `False` otherwise 
        """
        result = True
        if self.lower is not None:
            if self.strict_lower is True:
                result = result and (self.func(x) > self.lower).all() == True
            else:
                result = result and (self.func(x) >=
                                     self.lower).all() == True
        if self.upper is not None:
            if self.strict_upper is True:
                result = result and (self.func(x) < self.upper).all() == True
            else:
                result = result and (self.func(x) <=
                                     self.upper).all() == True
        return result


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
        self.obj_func = obj_func
        self.constrains = constrains
        self.dim = dim
        self.obj = obj
        self.solution = None

    def __call__(self, x):
        """
        Returns the evaluation of the `obf_func` in the vector/real value x
        """
        return self.obj_func(x)

    def get_constrains(self):
        """
        Returns a list with all the constrains of the problem
        """
        return self.constrains

    def get_dim(self):
        """
        Returns the dimension of the problem
        """
        return self.dim

    def get_obj(self):
        """
        Returns what should be done with `obj_min` ('min' or 'max')
        """
        return self.obj

    def get_solution(self):
        """
        Returns the best solution obtained at the moment of the calling
        """
        return self.solution

    def set_solution(self, solution):
        """
        Update the solution in case of improvement.
        Should be use only by the `Heuristic`  class
        """
        self.solution = solution