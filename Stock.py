# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:29:55 2018

@author: andy
"""

import numpy as np
from scipy.interpolate import interp1d, interp2d
from Stock_cali import Stock_calibrate
import math
import matplotlib.pyplot as plt


def Stock_drift_calibrate(div, rho_XS, vol_arg_Stock, vol_FX, TMax):
	drift_args = {}
	def Stock_drift_with_rho(xx,tt, r_euro):
		return r_euro - div - rho_XS * vol_FX * vol_arg_Stock['sigma'](xx,tt)
	
	#drift_args['drift'] = drift
	drift_args['drift'] = Stock_drift_with_rho
	return drift_args
	
def Stock_vol_calibrate(TMax):
	vol_args = {}
	t = np.linspace(0,TMax,num=10,endpoint=True)
	x = np.linspace(2000, 4000, num=10, endpoint=True)
	tt,xx = np.meshgrid(t,x)
	#sigma = tt+xx
	sigma = np.ones(tt.shape) * 0.2
	sigma = interp2d(t,x,sigma)
	sc = Stock_calibrate()
	sigma = sc.sigma
	vol_args['sigma'] = sigma
	return  vol_args
	
if __name__ == "__main__":
	r_euro = -0.37/100
	div = 0.0269
	rho = -0.24785
	vol_FX = 0.07
	T = 1.0
	T_init = 0.0
	eps = 1e-8
	delta_t = 1.0/120
	N = int((T - T_init) / delta_t)
	s0 = 3168.29
	x = np.zeros(N+1)
	x[0] = s0
	time = np.zeros(N+1)
	time[0] = T_init
	ensure_positive = True
	#drift_arg = {}
	#vol_arg = {}
	vol_arg_stock = Stock_vol_calibrate(T)
	drift_arg_stock = Stock_drift_calibrate(r_euro, div, rho, vol_arg_stock, vol_FX, T)

	for i in range(1,N+1):
		_t = T_init + delta_t * i
		time[i] = _t
		#drift_arg['t'] = _t
		#drift_arg['x'] = r_lst[i-1]
		#vol_arg['t'] = _t
		#vol_arg['x'] = r_lst[i-1]
		drift = drift_arg_stock['drift'](x[i-1],_t)
		#print(drift)
		vol = vol_arg_stock['sigma'](x[i-1],_t)
		#print(vol)
		x_tilde = x[i-1] + drift * x[i-1] * delta_t + vol * x[i-1] * math.sqrt(delta_t)
		vol_tilde = vol_arg_stock['sigma'](x_tilde, _t)
		Z = np.random.normal()
		x[i] = x[i-1] + drift * x[i-1] * delta_t + vol * x[i-1] * math.sqrt(delta_t) * Z \
			+ 0.5 * (vol_tilde * x_tilde - vol  * x[i-1]) * math.sqrt(delta_t) * (Z**2 - 1)
		#x[i] = x[i-1] + drift * delta_t + vol * math.sqrt(delta_t) * Z
		if(ensure_positive and x[i] < 0):
			x[i] = eps
	plt.plot(time, x)
	plt.title("Stock (in Euro under us measure) simulate")
	plt.show()
