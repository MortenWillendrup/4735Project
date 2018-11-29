# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 00:30:41 2018

@author: andy
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.stats import norm

from InterestRate_USD import Ho_Lee_USD, Hull_White_USD
from Stock import Stock_drift_calibrate, Stock_vol_calibrate

def const_vol(vol_arg):
	return vol_arg['sigma']

def GBM_vol(vol_arg):
	return vol_arg['sigma'] * vol_arg['x']
	
def local_GBM_vol(vol_arg):
	x = vol_arg['x']
	t = vol_arg['t']
	return vol_arg['x'] * vol_arg['sigma'](x, t)
	
def const_drift(drift_arg):
	return drift_arg['mu']

def GBM_drift(drift_arg):
	return drift_arg['mu'] * drift_arg['x']
	
def Ho_Lee_drift(drift_arg):
	theta = drift_arg['theta']
	return theta(drift_arg['t'])
	
def sto_GBM_drift(drift_arg):
	drift = drift_arg['drift']
	return drift(drift_arg['t'], drift_arg['x']) * drift_arg['x']
	
	

def RungeKutta_simulator(x0, T, drift_func, vol_func, T_init, delta_t, ensure_positive, **kwargs):
	eps = 1e-8
	if 'drift_arg' in kwargs:
		drift_arg = deepcopy(kwargs['drift_arg'])
	else:
		drift_arg = {}
		
	if 'vol_arg' in kwargs:
		vol_arg = deepcopy(kwargs['vol_arg'])
	else:
		vol_arg = {}
	#T = int(T+0.5)
	N = int((T - T_init) / delta_t)
	x = np.zeros(N+1)
	x[0] = x0
	for i in range(1,N+1):
		_t = T_init + delta_t * i
		drift_arg['t'] = _t
		drift_arg['x'] = x[i-1]
		vol_arg['t'] = _t
		vol_arg['x'] = x[i-1]
		drift = drift_func(drift_arg)
		#print(drift)
		vol = vol_func(vol_arg)
		x_tilde = x[i-1] + drift * delta_t + vol * math.sqrt(delta_t)
		vol_arg['x'] = x_tilde
		vol_tilde = vol_func(vol_arg=vol_arg)
		Z = np.random.normal()
		x[i] = x[i-1] + drift * delta_t + vol * math.sqrt(delta_t) * Z \
			+ 0.5 * (vol_tilde - vol) * math.sqrt(delta_t) * (Z**2 - 1)
		if(ensure_positive and x[i] < 0):
			x[i] = eps
		
	return x
	
def get_Payoffs(num_iter):
	for _it in range(num_iter):

		r_USD = RungeKutta_simulator(r0_USD, T1, Ho_Lee_drift, const_vol, 0.0 ,delta_t, False, drift_arg = drift_arg_USD, vol_arg = vol_arg_USD)
		
		r_Euro = r0_Euro

		Libor = libor_simulator(T1,r_USD[-1])
		if (Libor > L):
			payoffs[_it] = 0
			#print("Knocked out")
			continue
		
		#USD per Euro 
		vol_FX = 0.07

		
		#Stock in Euro, in Euro Measure, so we don't need to simulate FX
		drift_arg_Stock = Stock_drift_calibrate(r_Euro, q, rho, vol_arg_Stock, vol_FX, T)
		S = RungeKutta_simulator(S0, T, sto_GBM_drift, local_GBM_vol, 0.0, delta_t, True, drift_arg=drift_arg_Stock,vol_arg=vol_arg_Stock)

		payoffs[_it] = max(0, S[-1]-K)
	return payoffs
	
	
if __name__ == "__main__":
	num_iter = 1000
	delta_t = 1.0/12
	payoffs = np.zeros(num_iter)
	K = 3100
	L = 0.026
	T1 = 0.5
	T = 1.0
	
	
	
	# calibrate
	r0_USD = 0.022
	#drift_arg_USD, vol_arg_USD, libor_simulator = Ho_Lee_USD(T)
	drift_arg_USD, vol_arg_USD, libor_simulator = Hull_White_USD(T)

	r0_Euro = -0.37/100
	q = 0.0269
	
	Q0 = 1.13
	S0 = 3168.49
	vol_arg_Stock = Stock_vol_calibrate(T)
	
	rho = -0.37
	
	payoffs = get_Payoffs(num_iter)
	print("mean:" ,np.mean(payoffs))
	print("std of mean:", np.std(payoffs)/np.sqrt(num_iter))
	
#==============================================================================
# 	BS_sig = 0.14
# 	d1 = (np.log(S0/K) + (r0_Euro - q +0.5*BS_sig**2)*T)/(BS_sig*np.sqrt(T))
# 	d2 = d1 - np.sqrt(T)*BS_sig
# 	Nd1 = norm.cdf(d1)
# 	Nd2 = norm.cdf(d2)
# 	BS_call = S0*np.exp(-q*T)*Nd1 - K * np.exp(-r0_Euro*T)* Nd2
# 	print("Black Scholes :", BS_call)
# 
#==============================================================================
