# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:29:55 2018

@author: andy
"""

import numpy as np
from scipy.interpolate import interp1d
from EUR_shortrate_cali import HoLee_calibrate_EUR, HullWhite_calibrate_EUR
import math
import matplotlib.pyplot as plt

# Ho-Lee
def Ho_Lee_EUR(TMax):
	drift_args = {}
	vol_args = {}
	#theta = np.ones(100) * 0.0
	#theta = np.ones(100) * 0.001
	#x = np.linspace(0,TMax,num=100,endpoint=True)
	#theta = interp1d(x, theta)
	#sigma = 0.015
	#sigma = 0.0
	
	clb = HoLee_calibrate_EUR()
	sigma = clb.sigma
	theta = clb.theta
	
	drift_args['theta'] = theta
	vol_args['sigma'] = sigma
	return drift_args, vol_args

# Hull-White
def Hull_White_EUR(TMax):
	drift_args = {}
	vol_args = {}
	#theta = np.ones(100) * 0.0
	#theta = np.ones(100) * 0.001
	#x = np.linspace(0,TMax,num=100,endpoint=True)
	#theta = interp1d(x, theta)
	#sigma = 0.015
	#sigma = 0.0
	
	clb = HullWhite_calibrate_EUR()
	sigma = clb.sigma
	theta = clb.theta
	
	drift_args['theta'] = theta
	drift_args['a'] = clb.a
	vol_args['sigma'] = sigma
	return drift_args, vol_args
	
if __name__ == "__main__":
	r0 = -0.37/100
	T = 1.0
	T_init = 0.0
	eps = 1e-8
	delta_t = 1.0/12
	N = int((T - T_init) / delta_t)
	x = np.zeros(N+1)
	x[0] = r0
	time = np.zeros(N+1)
	time[0] = T_init
	ensure_positive = True
	#drift_arg = {}
	#vol_arg = {}
	model = "HoLee"
	if model == "HoLee":
		drift_arg_EUR, vol_arg_EUR = Ho_Lee_EUR(T)

		for i in range(1,N+1):
			_t = T_init + delta_t * i
			time[i] = _t
			#drift_arg['t'] = _t
			#drift_arg['x'] = r_lst[i-1]
			#vol_arg['t'] = _t
			#vol_arg['x'] = r_lst[i-1]
			drift = drift_arg_EUR['theta'](_t)
			#print(drift)
			vol = vol_arg_EUR['sigma']
			x_tilde = x[i-1] + drift * delta_t + vol * math.sqrt(delta_t)
			vol_tilde = vol_arg_EUR['sigma']
			Z = np.random.normal()
			x[i] = x[i-1] + drift * delta_t + vol * math.sqrt(delta_t) * Z \
				+ 0.5 * (vol_tilde - vol) * math.sqrt(delta_t) * (Z**2 - 1)
			#x[i] = x[i-1] + drift * delta_t + vol * math.sqrt(delta_t) * Z
			if(ensure_positive and x[i] < 0):
				x[i] = eps
		plt.plot(time, x)
		plt.title("EUR short rate simulate, Ho Lee")
		plt.show()
	elif model == "HullWhite":
		drift_arg_EUR, vol_arg_EUR = Hull_White_EUR(T)
		for i in range(1,N+1):
			_t = T_init + delta_t * i
			time[i] = _t
			#drift_arg['t'] = _t
			#drift_arg['x'] = r_lst[i-1]
			#vol_arg['t'] = _t
			#vol_arg['x'] = r_lst[i-1]
			drift = drift_arg_EUR['theta'](_t) - drift_arg_EUR['a'] * x[i-1]
			#print(drift)
			vol = vol_arg_EUR['sigma']
			x_tilde = x[i-1] + drift * delta_t + vol * math.sqrt(delta_t)
			vol_tilde = vol_arg_EUR['sigma']
			Z = np.random.normal()
			x[i] = x[i-1] + drift * delta_t + vol * math.sqrt(delta_t) * Z \
				+ 0.5 * (vol_tilde - vol) * math.sqrt(delta_t) * (Z**2 - 1)
			#x[i] = x[i-1] + drift * delta_t + vol * math.sqrt(delta_t) * Z
			if(ensure_positive and x[i] < 0):
				x[i] = eps
		plt.plot(time, x)
		plt.title("EUR short rate simulate, Hull White")
		plt.show()		
		
	
