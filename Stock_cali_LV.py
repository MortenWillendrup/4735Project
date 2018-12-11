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
from scipy.optimize import minimize

def BS(sig, F, K, P, T):
	d1 = (np.log(F/K)+(0.5*sig**2)*T)/sig/np.sqrt(T)
	d2 = d1 - sig*np.sqrt(T)
	Nd1 = norm.cdf(d1)
	Nd2 = norm.cdf(d2)
	call = P*(F*Nd1 - K * Nd2)
	return call


def BS_vol(F, K, C, P, T):
	sol = optimize.root(lambda x: BS(x, F, K, P, T)-C, 0.50, method='hybr')	
	return sol.x
def raw_SVI(a,b,rho,m,sigma,k):
	res = a+b*(rho*(k-m)+np.sqrt((k-m)**2+sigma**2))
	if res <0:
		#print(res, a+b*sigma*np.sqrt(1-rho*rho))
		return 1e-8
	return res
	

def impvol_Error(param, k, impvol, FT, PT, T):
	N = len(k)
	fitted = np.zeros(N)
	a,b,rho,m,sigma = param
	for i in range(N):
		fitted_tot_var = raw_SVI(a,b,rho,m,sigma,k[i])
		fitted_var = fitted_tot_var / T
		fitted_vol = np.sqrt(fitted_var)
		fitted[i] = fitted_vol
	return np.nanmean((impvol-fitted)**2)	
class Stock_calibrate_LV:
	"""
	Local Vol model calibration, assuming deterministic short rates and constant dividend
	"""
	def __init__(self):
		"""
		1. Read available european call data and parametrize using raw SVI.
		2. Read relevant stock data (S0, q)
		3. Compute local vol using Dupire Formula. 
		4. self.simga is a function that takes in (S_t,t,r_t)
		"""
		
		df_stock_stats = pd.read_csv(os.path.join("data", "Stock.csv"), index_col=0, header=None)
		spot_str = df_stock_stats.loc['Price'].squeeze()
		self.S0 = float(spot_str)
		self.q = float(df_stock_stats.loc['div'].squeeze().strip("%"))/100
		df_s = pd.read_csv(os.path.join("data","StockCall_shortterm.csv"),index_col=0)
		df_m = pd.read_csv(os.path.join("data","StockCall_midterm.csv"),index_col=0)
		df_l = pd.read_csv(os.path.join("data","StockCall_longterm.csv"),index_col=0)
		df_all = pd.concat([df_s, df_m, df_l], axis=1)
		#df_all = pd.concat([df_s, df_m], axis=1)

		df_all = df_all.loc[df_all.index >=2500]
		df_all = df_all.loc[df_all.index<=3500 ]
		df_all.columns = df_all.columns.astype(int)
		#df_all = df_all.loc[:,df_all.columns <= 3*365]
	
		df_Yield = pd.read_csv(os.path.join("data","GERYield.csv"),
					header = None, delimiter = "\t", index_col=0)
		T = np.array(df_Yield.index)
		yields = np.array(df_Yield.loc[:,1])
		self.yields_interp = interpolate.interp1d(T.squeeze(), yields, 'cubic', fill_value='extrapolate')
		for _T in df_all.columns:
			T = _T/365.0
			PT = self.EUR_bond_price(T)
			FT = self.stock_forward_price(T)
			for K in df_all.index:
				if np.isnan(df_all.loc[K,_T]):
					continue
				df_all.loc[K,_T] = BS_vol(FT, K, df_all.loc[K,_T], PT, T)
		self.df_impliedvols = df_all
		self.get_RAW_SVI()
		self.sigma=self.local_vol
	def EUR_bond_price(self,T):
		T_yield = self.yields_interp(T)
		price = (1+T_yield/100)**(-T)
		return price
	def stock_forward_price(self,T):
		PT = self.EUR_bond_price(T)
		return self.S0/PT*np.exp(-self.q*T)
	

	
	def get_RAW_SVI(self):
		SVI = {}
		
		for _T in self.df_impliedvols.columns:
			#print("\n",_T)
			T = _T/365.0
			tmp_dict = {}
			PT = self.EUR_bond_price(T)
			FT = self.stock_forward_price(T)
			df_impvol = self.df_impliedvols.loc[:,_T].dropna()
			K = np.array(df_impvol.index)
			implied_vol = np.array(df_impvol)
			k = np.log(K/FT)
			#N = len(implied_vol)
			#print(df_impvol)
			raw_SVI_initial = [0.001, 0.2, 0.03, -0.2, 0.1]
			fun = lambda x: impvol_Error(x, k, implied_vol, FT,PT,T)
			bnds = ((None, None), (0, None), (-1,1), (None,None),(0,None))
			cons = ({'type': 'ineq', 'fun': lambda x:  x[0]+x[1]*x[4]*np.sqrt(1-x[2]**2)})
			res = minimize(fun, raw_SVI_initial, method='SLSQP', bounds=bnds,  constraints=cons)
			#print(res.success)			
			tmp_dict['a'] = res.x[0]
			tmp_dict['b'] = res.x[1]
			tmp_dict['rho'] = res.x[2]
			tmp_dict['m'] = res.x[3]
			tmp_dict['sigma'] = res.x[4]
			SVI[_T] = tmp_dict
			#tot_vol = []
			#imp_vol = []
			#for KK in np.linspace(2000,4000,21):
				
				#kk = np.log(KK/FT)
				#w = raw_SVI(res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],kk)
				#tot_vol.append(w)
				#imp_vol.append(np.sqrt(w/T))
			#print(T*365.0, tot_vol, imp_vol)
			#plt.plot(np.linspace(2000,4000,21), tot_vol, label="T="+str(_T) + "days")
		#plt.legend()
		#plt.title("implied tot var of different T")
		#plt.show()
		self.SVI_df = pd.DataFrame(SVI)
		
	def SVI_imp_vol(self,K,T):
		FT = self.stock_forward_price(T)
		k = np.log(K/FT)
		T_arr_indays = np.array(self.SVI_df.columns)
		T_indays = T*365.0
		_i = np.searchsorted(T_arr_indays,T_indays)
		if _i == 0 or _i == len(T_arr_indays):
			if _i == 0:
				#T1 = T_arr_indays[_i]/365.0
				param = self.SVI_df.loc[:,T_arr_indays[0]]

			else:
				#T1 = T_arr_indays[-1]/365.0
				param = self.SVI_df.loc[:,T_arr_indays[-1]]

			#FT1 = self.stock_forward_price(T1)
			a,b,rho,m,sigma = list(param)
			#k = np.log(K/FT)
			imp_tot_var = raw_SVI(a,b,rho,m,sigma,k)
			imp_vol = np.sqrt(imp_tot_var/T)
			return imp_vol
		else:
			T1 = T_arr_indays[_i-1]
			T2 = T_arr_indays[_i]
			param1 = self.SVI_df.loc[:,T1]
			a1 = param1['a']
			b1 = param1['b']
			m1 = param1['m']
			rho1 = param1['rho']
			sigma1 = param1['sigma']
			param2 = self.SVI_df.loc[:,T2]
			a2 = param2['a']
			b2 = param2['b']
			m2 = param2['m']
			rho2 = param2['rho']
			sigma2 = param2['sigma']
			T1 = T1/365.0
			T2 = T2/365.0
			#FT1 = self.stock_forward_price(T1)
			#FT2 = self.stock_forward_price(T2)

			#k1 = np.log(K/FT1)
			#k2 = np.log(K/FT2)
			imp_tot_var1 = raw_SVI(a1,b1,rho1,m1,sigma1,k)
			imp_tot_var2 = raw_SVI(a2,b2,rho2,m2,sigma2,k)
			imp_tot_var = (imp_tot_var2 - imp_tot_var1)/(T2-T1)*(T-T1)+imp_tot_var1
			imp_vol = np.sqrt(imp_tot_var/T)
			#PT = self.EUR_bond_price(T)
			#return BS(imp_vol,FT,K,PT,T)
			return imp_vol
	def local_vol(self, K, T, r):
		"""
		Compute local vol given K, t, and r_t using Dupire formula 
		assuming nonconstant but deterministic short rate		
		"""
		FT = self.stock_forward_price(T)
		imp_vol = self.SVI_imp_vol(K,T)
		d1 = (np.log(FT/K)+0.5*imp_vol*imp_vol)/(imp_vol*np.sqrt(T))
		d2 = d1 - imp_vol*np.sqrt(T)

		dsigmadT = partial_derivative(self.SVI_imp_vol, 1 ,[K, T], dx=1e-3) 
		dsigmadK = partial_derivative(self.SVI_imp_vol, 0, [K, T], dx=50) 
		d2sigmadK2 = partial_derivative(self.SVI_imp_vol, 0, [K, T],2, dx=50) 
		numerator = 0.5*imp_vol/T + dsigmadT + K*(r-self.q)* dsigmadK
		coef = 0.5*K*K
		first = 1.0/(imp_vol*K*K*T)
		second = 2*dsigmadK*d1/(imp_vol*K*np.sqrt(T))
		third = d2sigmadK2
		fourth = dsigmadK**2 * (d1*d2)/imp_vol
		denumerator = coef*(first+second+third+fourth)
		sig2 = numerator / denumerator
		if sig2 >1e-8 and sig2 < 0.7**2:
			return np.sqrt(sig2)
		elif sig2 <= 1e-8:
			return 0.01
		else:
			return 0.7


if __name__ == "__main__":
	sc = Stock_calibrate_LV()
	#df = sc.df_all
	#print(sc.local_vol(4000,0.6))
	
	NK= 20
	NT =10
	X = np.linspace(2500,4000,NK)
	Y = np.linspace(0.1,2,NT)
	XX,YY = np.meshgrid(X,Y)
	Z = np.zeros((NK,NT))
	for j in range(0,NT):
		for i in range(0,NK):
			Z[i,j] = sc.local_vol(X[i],Y[j], -0.00)


	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.contour3D(XX, YY, Z.T, 50, cmap='binary')
	plt.show()
	
	print(Z.mean())

