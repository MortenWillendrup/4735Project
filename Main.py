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
import os

from InterestRate_USD import Ho_Lee_USD, Hull_White_USD
from Stock import Stock_drift_calibrate, Stock_vol_calibrate
from correlation import Correlation_calibrate

def get_discount(r, delta_t):
	sum_r = 0.0
	for i in range(0, len(r)):
		if i == 0 or i == len(r)-1:
			sum_r += 0.5*r[i]
		else:
			sum_r += r[i]
	return np.exp(-sum_r*delta_t)

def RungeKutta_simulator(Params):
	vol_arg_stock = Params['stock']['vol']
	drift_arg_stock = Params['stock']['drift']
	S0 = Params['stock']['S0']
	#div = Params['stock']['div']
	#r_Euro = Params['stock']['r']
	r0_USD = Params['r_USD']['r0_USD']
	drift_arg_USD = Params['r_USD']['drift']
	vol_arg_USD = Params['r_USD']['vol']
	libor_simulator =  Params['r_USD']['Libor']
	#vol_FX = Params['FX']['vol']
	#rho_XS = Params['corr']['rho_XS']
	rho_RS = Params['corr']['rho_RS']
	#num_iter = Params['Monte_Carlo']['num_iter']
	delta_t = Params['Monte_Carlo']['delta_t']
	T = Params['Monte_Carlo']['T']
	T1 = Params['Monte_Carlo']['T1']
	K = Params['Monte_Carlo']['K']
	L = Params['Monte_Carlo']['L']
	ensure_positive = Params['Monte_Carlo']['pos']
	eps = Params['Monte_Carlo']['eps']

	T_init = 0.0
	N = int((T - T_init) / delta_t)
	r = np.zeros(N+1)
	r[0] = r0_USD
	S = np.zeros(N+1)
	S[0] = S0
	N1 = int((T1 - T_init)/delta_t)

	
	time = np.zeros(N+1)
	time[0] = T_init
	
	for i in range(1,N+1):
		Z1 = np.random.normal()
		Z2 = rho_RS * Z1 + math.sqrt(1-rho_RS**2)*np.random.normal()
		_t = T_init + delta_t * i
		time[i] = _t
		drift_r = drift_arg_USD['theta'](_t)
		vol_r = vol_arg_USD['sigma']
		r[i] = r[i-1] + drift_r * delta_t + vol_r * math.sqrt(delta_t) * Z1 
		if(ensure_positive and r[i] < 0):
			r[i] = eps
		if(i == N1):
			Libor = libor_simulator(_t, r[i])
			if Libor > L:
				return 0
		drift_S = drift_arg_stock['drift'](S[i-1],_t)
		vol_S = vol_arg_stock['sigma'](S[i-1],_t)
		S_tilde = S[i-1] + drift_S * S[i-1] * delta_t + vol_S * S[i-1] * math.sqrt(delta_t)
		vol_S_tilde = vol_arg_stock['sigma'](S_tilde, _t)
		S[i] = S[i-1] + drift_S * S[i-1] * delta_t + vol_S * S[i-1] * math.sqrt(delta_t) * Z2 \
			+ 0.5 * (vol_S_tilde * S_tilde - vol_S  * S[i-1]) * math.sqrt(delta_t) * (Z2**2 - 1)		
	payoff = max(0, S[N]-K)
	discount = get_discount(r,delta_t)
	return discount*payoff
	

	
def get_prices(num_iter, Params):
	results = np.zeros(num_iter)
	for _it in range(num_iter):
		results[_it] = RungeKutta_simulator(Params)
	return results
	
	
if __name__ == "__main__":
	# user-defined parameters
	K = 3100
	L = 0.026
	T1 = 0.5
	T = 1.0
	
	# parameters for MonteCarlo
	num_iter = 500
	delta_t = 0.1
	ensure_positive = True
	eps = 1e-8
	payoffs = np.zeros(num_iter)
	
	# get available short rates data
	df_general = pd.read_csv(os.path.join("data","General.csv"),index_col=0, header=None)
	r0_Euro = df_general.loc['r_Euro'].squeeze()
	r0_USD = df_general.loc['r_US'].squeeze()

	# get available stocks data
	df_stock_stats = pd.read_csv(os.path.join("data","Stock.csv"),index_col=0, header=None)
	S0 = float(df_stock_stats.loc['Price'].squeeze())
	q =  df_stock_stats.loc['div'].squeeze()
	q = float(q.strip("%"))/100
	
	# calibrate correlations and FX sigma from historical time series
	corr_calib = Correlation_calibrate()
	rho_XS = corr_calib.rho_XS #X stands for FX, S stands for stock
	rho_RS = corr_calib.rho_RS #R stands for US short rate
	vol_FX = corr_calib.vol_FX
	
	
	#drift_arg_USD, vol_arg_USD, libor_simulator = Ho_Lee_USD(T)
	drift_arg_USD, vol_arg_USD, libor_simulator = Hull_White_USD(T)

	
	#Q0 = 1.13
	vol_arg_Stock = Stock_vol_calibrate(T)
	drift_arg_Stock = Stock_drift_calibrate(r0_Euro, q, rho_XS, vol_arg_Stock, vol_FX, T)
	
	Params = {}
	Params['stock'] = {'vol':vol_arg_Stock, 'drift':drift_arg_Stock, 'S0':S0, 'div':q, 'r':r0_Euro}
	Params['r_USD'] = {'r0_USD':r0_USD, 'drift':drift_arg_USD, 'vol': vol_arg_USD,'Libor':libor_simulator}
	Params['FX'] = {'vol':vol_FX}
	Params['corr'] = {'rho_XS':rho_XS, 'rho_RS':rho_RS}
	Params['Monte_Carlo'] = {'T':T, 'T1':T1, 'delta_t':delta_t, 'num_iter':num_iter, 'K':K, 'L':L, 'pos':ensure_positive, 'eps':eps}

		
	prices = get_prices(num_iter, Params)
	print("mean:" ,np.mean(prices))
	print("std of mean:", np.std(prices)/np.sqrt(num_iter))
	
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
