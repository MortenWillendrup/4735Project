# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:29:55 2018

@author: andy
"""


from USD_shortrate_cali import HoLee_calibrate, HullWhite_calibrate

# Ho-Lee
def Ho_Lee_USD(use_caplets):
	"""
	Return drift args, vol args, and libor simulator under HoLee Model
	drift_args: drft_args['theta'] is a function that takes in t as an argument, calibrated using HoLee
	vol_args: vol_args['sigma'] is a constant given by HoLee calibration
	clb.Libor: a function that takes (t, r_t) that gives L(t,t,t+3month)
	"""
	drift_args = {}
	vol_args = {}

	
	clb = HoLee_calibrate(use_caplets)
	sigma = clb.sigma
	theta = clb.theta
	
	drift_args['theta'] = theta
	vol_args['sigma'] = sigma
	return drift_args, vol_args, clb.Libor

# Hull-White
def Hull_White_USD(use_caplets):
	"""
	Return drift args, vol args, and libor simulator under HullWhite Model
	drift_args: drft_args['theta'] is a function that takes in t as an argument, calibrated using HullWhite
	vol_args: vol_args['sigma'] is a constant given by HullWhite calibration
	clb.Libor: a function that takes (t, r_t) that gives L(t,t,t+3month)
	"""
	drift_args = {}
	vol_args = {}
	
	clb = HullWhite_calibrate(use_caplets)
	sigma = clb.sigma
	theta = clb.theta
	
	drift_args['theta'] = theta
	drift_args['a'] = clb.a
	vol_args['sigma'] = sigma
	return drift_args, vol_args, clb.Libor
	
#==============================================================================
# import numpy as np
# from scipy.interpolate import interp1d
# import math
# import matplotlib.pyplot as plt
# if __name__ == "__main__":
# 	r0 = 0.021
# 	T = 1.0
# 	T_init = 0.0
# 	eps = 1e-8
# 	delta_t = 1.0/120
# 	N = int((T - T_init) / delta_t)
# 	x = np.zeros(N+1)
# 	x[0] = r0
# 	time = np.zeros(N+1)
# 	time[0] = T_init
# 	ensure_positive = True
# 	#drift_arg = {}
# 	#vol_arg = {}
# 	model = "HullWhite"
# 	if model == "HoLee":
# 		drift_arg_USD, vol_arg_USD,_ = Ho_Lee_USD(T)
# 
# 		for i in range(1,N+1):
# 			_t = T_init + delta_t * i
# 			time[i] = _t
# 			#drift_arg['t'] = _t
# 			#drift_arg['x'] = r_lst[i-1]
# 			#vol_arg['t'] = _t
# 			#vol_arg['x'] = r_lst[i-1]
# 			drift = drift_arg_USD['theta'](_t)
# 			#print(drift)
# 			vol = vol_arg_USD['sigma']
# 			x_tilde = x[i-1] + drift * delta_t + vol * math.sqrt(delta_t)
# 			vol_tilde = vol_arg_USD['sigma']
# 			Z = np.random.normal()
# 			x[i] = x[i-1] + drift * delta_t + vol * math.sqrt(delta_t) * Z \
# 				+ 0.5 * (vol_tilde - vol) * math.sqrt(delta_t) * (Z**2 - 1)
# 			#x[i] = x[i-1] + drift * delta_t + vol * math.sqrt(delta_t) * Z
# 			if(ensure_positive and x[i] < 0):
# 				x[i] = eps
# 		plt.plot(time, x)
# 		plt.title("USD short rate simulate, Ho Lee")
# 		plt.show()
# 	elif model == "HullWhite":
# 		drift_arg_USD, vol_arg_USD,_ = Hull_White_USD(T)
# 		for i in range(1,N+1):
# 			_t = T_init + delta_t * i
# 			time[i] = _t
# 			#drift_arg['t'] = _t
# 			#drift_arg['x'] = r_lst[i-1]
# 			#vol_arg['t'] = _t
# 			#vol_arg['x'] = r_lst[i-1]
# 			drift = drift_arg_USD['theta'](_t) - drift_arg_USD['a'] * x[i-1]
# 			#print(drift)
# 			vol = vol_arg_USD['sigma']
# 			x_tilde = x[i-1] + drift * delta_t + vol * math.sqrt(delta_t)
# 			vol_tilde = vol_arg_USD['sigma']
# 			Z = np.random.normal()
# 			x[i] = x[i-1] + drift * delta_t + vol * math.sqrt(delta_t) * Z \
# 				+ 0.5 * (vol_tilde - vol) * math.sqrt(delta_t) * (Z**2 - 1)
# 			#x[i] = x[i-1] + drift * delta_t + vol * math.sqrt(delta_t) * Z
# 			if(ensure_positive and x[i] < 0):
# 				x[i] = eps
# 		plt.plot(time, x)
# 		plt.title("USD short rate simulate, Hull White")
# 		plt.show()		
#==============================================================================
		
	
