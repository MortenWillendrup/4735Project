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
from scipy import interpolate


from InterestRate_USD import Ho_Lee_USD, Hull_White_USD
from InterestRate_EUR import Ho_Lee_EUR, Hull_White_EUR

from Stock import Stock_drift_calibrate, Stock_vol_calibrate
from correlation import Correlation_calibrate

def Ho_Lee_drift(drift_arg):
	def drift(t,r):
		return drift_arg['theta'](t)
	return drift

def Ho_Lee_vol(vol_arg):
	return vol_arg['sigma']

def Hull_White_drift(drift_arg):
	def drift(t,r):
		return drift_arg['theta'](t) - drift_arg['a']*r
	return drift

def Hull_White_vol(vol_arg):
	return vol_arg['sigma']		

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
	drift_func_USD = Params['r_USD']['drift']
	vol_func_USD = Params['r_USD']['vol']
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
	
	r_Euro = Params['r_EUR']['interp']

	T_init = 0.0
	N = int((T - T_init) / delta_t)
	r_USD = np.zeros(N+1)
	r_USD[0] = r0_USD
	S = np.zeros(N+1)
	S[0] = S0
	N1 = int((T1 - T_init)/delta_t)

	
	time = np.zeros(N+1)
	time[0] = T_init
	
	for i in range(1,N+1):
		Z1 = np.random.normal()
		Z2 = rho_RS * Z1 + math.sqrt(1-rho_RS**2)*np.random.normal()
		_t = T_init + delta_t * i
		r_Euro_t = r_Euro(_t)
		time[i] = _t
		drift_r = drift_func_USD(_t, r_USD[i-1])
		vol_r = vol_func_USD
		r_USD[i] = r_USD[i-1] + drift_r * delta_t + vol_r * math.sqrt(delta_t) * Z1 
		if(ensure_positive and r_USD[i] < 0):
			r_USD[i] = eps
		if(i == N1):
			Libor = libor_simulator(_t, r_USD[i])
			if Libor > L:
				return 0
		drift_S = drift_arg_stock['drift'](S[i-1],_t, r_Euro_t)
		vol_S = vol_arg_stock['sigma'](S[i-1],_t, r_Euro_t)
		S_tilde = S[i-1] + drift_S * S[i-1] * delta_t + vol_S * S[i-1] * math.sqrt(delta_t)
		vol_S_tilde = vol_arg_stock['sigma'](S_tilde, _t, r_Euro_t)
		S[i] = S[i-1] + drift_S * S[i-1] * delta_t + vol_S * S[i-1] * math.sqrt(delta_t) * Z2 \
			+ 0.5 * (vol_S_tilde * S_tilde - vol_S  * S[i-1]) * math.sqrt(delta_t) * (Z2**2 - 1)		
	payoff = max(0, S[N]-K)
	discount = get_discount(r_USD,delta_t)
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
	drift_func_USD = Hull_White_drift(drift_arg_USD)
	vol_func_USD = Hull_White_vol(vol_arg_USD)
	drift_arg_EUR, vol_arg_EUR = Hull_White_EUR(T)
	drift_func_EUR = Hull_White_drift(drift_arg_EUR)
	vol_func_EUR = Hull_White_vol(vol_arg_EUR)
	
	#Q0 = 1.13
	vol_arg_Stock = Stock_vol_calibrate(T)
	drift_arg_Stock = Stock_drift_calibrate(q, rho_XS, vol_arg_Stock, vol_FX, T)
	
	Params = {}
	Params['stock'] = {'vol':vol_arg_Stock, 'drift':drift_arg_Stock, 'S0':S0, 'div':q, 'r':r0_Euro}
	Params['r_USD'] = {'r0_USD':r0_USD, 'drift':drift_func_USD, 'vol': vol_func_USD,'Libor':libor_simulator}
	Params['r_EUR'] = {'r0_EUR':r0_Euro, 'drift':drift_func_EUR, 'vol': vol_func_EUR}

	Params['FX'] = {'vol':vol_FX}
	Params['corr'] = {'rho_XS':rho_XS, 'rho_RS':rho_RS}
	Params['Monte_Carlo'] = {'T':T, 'T1':T1, 'delta_t':delta_t, 'num_iter':num_iter, 'K':K, 'L':L, 'pos':ensure_positive, 'eps':eps}
	delta_t = Params['Monte_Carlo']['delta_t']

	# get determinstic Euro short rate
	T_init = 0.0
	N = int((T - T_init) / delta_t)
	r0_EUR = Params['r_EUR']['r0_EUR']	
	r_EUR = np.zeros(N+1)
	r_EUR[0] = r0_EUR
	time = np.zeros(N+1)
	time[0] = T_init
	for i in range(1,N+1):
		_t = T_init + delta_t * i
		time[i] = _t
		drift = Params['r_EUR']['drift'](_t, r_EUR[i-1])
		#print(drift)
		r_EUR[i] = r_EUR[i-1] + drift * delta_t
	r_EUR_interp = interpolate.interp1d(time, r_EUR, 'linear', fill_value='extrapolate')

	Params['r_EUR']['interp'] = r_EUR_interp
		
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
