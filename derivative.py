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
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return derivative(wraps, point[var], dx = dx, n=n, order=7)
				
if __name__ == "__main__":
	print(partial_derivative(f, 0, [3,2], 2))
	
	ff = functools.partial(f, 3)
	print(derivative(ff, 2, dx=1e-6, n =1))