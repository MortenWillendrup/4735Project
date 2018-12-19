# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 19:24:54 2018

@author: andy
"""

import pandas as pd
from scipy import interpolate
import numpy as np
import os
import scipy.stats as stats
from scipy.optimize import minimize

# Calculate the Black Scholes price of European call option
def bs_price(s, k, p, T, q, sigma):
	F = s * np.exp(-q * T)/p
	d1 = (np.log(F / k) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
	d2 = d1 - sigma * np.sqrt(T)
	price = p * (F * stats.norm.cdf(d1) - k *  stats.norm.cdf(d2))
	return price
				
class Stock_calibrate_BS:
	"""
	BS model calibration, assuming deterministic short rates and constant dividend
	"""
	def __init__(self):
		"""
		1. Compute BS vol using Black Formula. 
		2. self.simga is a function that takes in (S_t,t,r_t), although the arguments do not affect the value returned (just to be consistent with LV sigma function)
		"""
		# Import raw data
		df_stock_stats = pd.read_csv(os.path.join("data", "Stock.csv"), index_col=0, header=None)
		spot_str = df_stock_stats.loc['Price'].squeeze()
		self.spot = float(spot_str)
		div_str = df_stock_stats.loc['div'].squeeze()
		self.q = float(div_str.strip("%"))/100
		df_s = pd.read_csv(os.path.join("data","StockCall_shortterm.csv"),index_col=0)
		df_m = pd.read_csv(os.path.join("data","StockCall_midterm.csv"),index_col=0)
		df_l = pd.read_csv(os.path.join("data","StockCall_longterm.csv"),index_col=0)
		
		# Drop deep ITM options data which have little liquidity
		self.df_all = pd.concat([df_s, df_m,df_l], axis=1)
		self.df_all = self.df_all.loc[self.df_all.index >=2500]
		self.df_all = self.df_all.loc[self.df_all.index<=3500 ]
		# Euro bond price
		df_Yield = pd.read_csv(os.path.join("data","GERYield.csv"),
		        header = None, delimiter = "\t", index_col=0)
		T = np.array(df_Yield.index)
		yields = np.array(df_Yield.loc[:,1])
		self.yields_interp = interpolate.interp1d(T.squeeze(), yields, 'cubic', fill_value='extrapolate')
		
		T_indays = self.df_all.columns.astype(float)
		self.T_inyears = T_indays/365.0
		
		self.observed_prices = np.array(self.df_all)
		res = minimize(lambda x: self.fit_MSE(x),0.2,method='BFGS', tol=1.0)
		print("BS impvol:", res.x[0])
		def BS_impvol(s,t,r):
			return res.x[0]
		self.sigma = BS_impvol
		
	def get_fitted_call_price(self,sig):
		fitted_prices = np.zeros(self.observed_prices.shape)
		for j,K in enumerate(self.df_all.index):
			for i in range(len(self.T_inyears)):
				T = self.T_inyears[i]
				p = self.Euro_bond_price(T)
				fitted_prices[j,i] = bs_price(self.spot,K,p,T,self.q,sig)
		
		return fitted_prices
	
	def fit_MSE(self,sig):
		fitted_prices=self.get_fitted_call_price(sig)
		err = (fitted_prices-self.observed_prices)**2
		return np.nanmean(err)
		
	def Euro_bond_price(self,T):
		T_yield = self.yields_interp(T)
		price = (1+T_yield/100)**(-T)
		return price


