# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 19:24:54 2018

@author: andy
"""

import numpy as np
from derivative import partial_derivative
from scipy.misc import derivative
import pandas as pd
import os 
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import optimize
import math
from mpl_toolkits.mplot3d import Axes3D

def BS(sig, S, K, r, q, T):
	d1 = (np.log(S/K)+(r-q+0.5*sig**2)*T)/sig/np.sqrt(T)
	d2 = d1 - sig*np.sqrt(T)
	Nd1 = norm.cdf(d1)
	Nd2 = norm.cdf(d2)
	call = S*np.exp(-q*T)*Nd1 - K * np.exp(-r*T) * Nd2
	return call


def BS_vol(S, K, C, r, q, T):
	sol = optimize.root(lambda x: BS(x, S, K, r, q, T)-C, 0.10, method='hybr')	
	return sol.x
	
class Stock_calibrate:
	def __init__(self):
		df_s = pd.read_csv(os.path.join("data","StockCall_shortterm.csv"),index_col=0)
		df_m = pd.read_csv(os.path.join("data","StockCall_midterm.csv"),index_col=0)
		df_l = pd.read_csv(os.path.join("data","StockCall_longterm.csv"),index_col=0)
		df_all = pd.concat([df_s, df_m, df_l], axis=1)
		df_all.columns = df_all.columns.astype(int)

		df_all.loc[4300,933] = 10.5
		df_all.loc[4400,933] = 8		
		df_all.columns = df_all.columns/365.0
		self.df_all = df_all.interpolate(axis=0, method="linear", limit_direction="both")
		call = np.array(self.df_all).T
		K = np.array(self.df_all.index).reshape((1,-1))
		T = np.array(self.df_all.columns).reshape((-1,1))
		self.call_interpolate = interpolate.interp2d(K,T,call,'quintic')
		#df_gen = pd.read_csv(os.path.join("data", "General.csv"), index_col=0)
		#self.r = float(df_gen.loc['r_Euro'])
		df_stock_stats = pd.read_csv(os.path.join("data", "Stock.csv"), index_col=0, header=None)
		spot_str = df_stock_stats.loc['Price'].squeeze()
		self.spot = float(spot_str)
		div_str = df_stock_stats.loc['div'].squeeze()
		self.div = float(div_str.strip("%"))/100
		bs_sigma_str = df_stock_stats.loc['sigma'].squeeze()
		self.bs_sigma = float(bs_sigma_str.strip("%"))/100
		self.sigma = self.local_vol
		KK, TT = np.meshgrid(K.squeeze(),T.squeeze())
		X = np.linspace(2000,4400,20)
		Y = np.linspace(0.05,10,20)
		XX,YY = np.meshgrid(X,Y)
		Z = self.call_interpolate(X,Y)
		
		fig = plt.figure()
		ax = plt.axes(projection='3d')
		ax.contour3D(KK, TT, call, 50, cmap='binary')
		plt.show()


		fig = plt.figure()
		ax = plt.axes(projection='3d')
		ax.contour3D(XX, YY, Z, 50, cmap='binary')
		plt.show()
	def local_vol(self, K, T, r):
		
		dCdT = partial_derivative(self.call_interpolate, 1 ,[K, T], dx=1e-4) 
		dCdK = partial_derivative(self.call_interpolate, 0, [K, T], dx=10) 
		d2CdK2 = partial_derivative(self.call_interpolate, 0, [K, T],2, dx=10) 
		C = self.call_interpolate(K,T)
		sig2 = 2 * (dCdT + self.div * C + (r-self.div)*K*dCdK)/(K**2*d2CdK2)
		#return 0
		if sig2 > 0 and sig2 < 0.5**2:
			return math.sqrt(sig2)
		elif sig2 <= 0:
			return 0.01
			#return self.bs_sigma
		else:
			return BS_vol(self.spot, K, C, r, self.div, T)[0]
			#return self.bs_sigma


if __name__ == "__main__":
	sc = Stock_calibrate()
	df = sc.df_all
	#print(sc.local_vol(4000,0.6))
	
	N = 30
	X = np.linspace(2500,3500,N)
	Y = np.linspace(0.1,2,N)
	XX,YY = np.meshgrid(X,Y)
	Z = np.zeros((N,N))
	for i in range(0,N):
		for j in range(0,N):
			Z[i,j] = sc.local_vol(X[i],Y[j], -0.37/100)


	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.contour3D(XX, YY, Z, 50, cmap='binary')
	plt.show()
	
	print(Z.mean())
