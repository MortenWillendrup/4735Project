# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 19:40:58 2018

@author: andy
"""

from scipy.misc import derivative
import numpy as np
import functools

def f(x,y):
	return x**2 - np.sin(y)
	
def partial_derivative(func, var=0, point=[], n=1, dx=1e-4):
    """
	func: a callable function
	var: integer, with which argument to take partial derivative 
		e.g. if var = 0, then take partial derivative w.r.t first argument
			if var = 1, then take partial derivative w.r.t second argument
	point: point at which the partial derivative is estimated
	n: order of derivative
	dx: spacing, float
    """
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return derivative(wraps, point[var], dx = dx, n=n, order=7)
				
if __name__ == "__main__":
	print(partial_derivative(f, 0, [3,2], 2))
	
	ff = functools.partial(f, 3)
	print(derivative(ff, 2, dx=1e-6, n =1))