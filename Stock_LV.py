# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:29:55 2018

@author: andy
"""


from Stock_LV_calibrate import Stock_calibrate_LV


def Stock_drift_calibrate_LV(div, rho_XS, vol_arg_Stock, vol_FX):
	"""
	calculates (r_t-q-rho_XS*vol_FX*vol_Stock)
	Parameters
		div: stock dividend rate
		rho_XS: corr between FX and stock return
		vol_arg_Stock: a callable func(S_t,t,r_t), assuming constant vol
		vol_FX: float, vol of FX
	Returns a dictionary with key:'drift', value = func(S_t,t,r_t(euro))
	For example, drift_args['drift'](100,1,0.1) will give the drift when S_t = 100, t = 1.0, r_t(euro)=0.1
	"""
	drift_args = {}
	def Stock_drift_with_rho(xx,tt, r_euro):
		return r_euro - div - rho_XS * vol_FX * vol_arg_Stock['sigma'](xx,tt, r_euro)
	
	drift_args['drift'] = Stock_drift_with_rho
	return drift_args
	
def Stock_vol_calibrate_LV():
	vol_args = {}

	"""
	Returns a dictionary with key:'sigma', value = func(S_t, t, r_t(euro))
	"""
	sc = Stock_calibrate_LV()
	sigma = sc.sigma
	vol_args['sigma'] = sigma
	return  vol_args
	
#==============================================================================
# import math
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.interpolate import interp1d, interp2d
# if __name__ == "__main__":
# 	r_euro = -0.37/100
# 	div = 0.0269
# 	rho = -0.24785
# 	vol_FX = 0.07
# 	T = 1.0
# 	T_init = 0.0
# 	eps = 1e-8
# 	delta_t = 1.0/120
# 	N = int((T - T_init) / delta_t)
# 	s0 = 3168.29
# 	x = np.zeros(N+1)
# 	x[0] = s0
# 	time = np.zeros(N+1)
# 	time[0] = T_init
# 	ensure_positive = True
# 	#drift_arg = {}
# 	#vol_arg = {}
# 	vol_arg_stock = Stock_vol_calibrate(T)
# 	drift_arg_stock = Stock_drift_calibrate(div, rho, vol_arg_stock, vol_FX, T)
# 
# 	for i in range(1,N+1):
# 		_t = T_init + delta_t * i
# 		time[i] = _t
# 		#drift_arg['t'] = _t
# 		#drift_arg['x'] = r_lst[i-1]
# 		#vol_arg['t'] = _t
# 		#vol_arg['x'] = r_lst[i-1]
# 		drift = drift_arg_stock['drift'](x[i-1],_t, r_euro)
# 		#print(drift)
# 		vol = vol_arg_stock['sigma'](x[i-1],_t, r_euro)
# 		#print(vol)
# 		x_tilde = x[i-1] + drift * x[i-1] * delta_t + vol * x[i-1] * math.sqrt(delta_t)
# 		vol_tilde = vol_arg_stock['sigma'](x_tilde, _t, r_euro)
# 		Z = np.random.normal()
# 		x[i] = x[i-1] + drift * x[i-1] * delta_t + vol * x[i-1] * math.sqrt(delta_t) * Z \
# 			+ 0.5 * (vol_tilde * x_tilde - vol  * x[i-1]) * math.sqrt(delta_t) * (Z**2 - 1)
# 		#x[i] = x[i-1] + drift * delta_t + vol * math.sqrt(delta_t) * Z
# 		if(ensure_positive and x[i] < 0):
# 			x[i] = eps
# 	plt.plot(time, x)
# 	plt.title("Stock (in Euro under us measure) simulate")
# 	plt.show()
# 
#==============================================================================
