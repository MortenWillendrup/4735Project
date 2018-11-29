# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 17:37:54 2018

@author: andy
"""

import numpy as np
from scipy.misc import derivative
import pandas as pd
import os 
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import optimize

class RUSD_calibrate:
	def __init__(self):
		self.dx = 1e-2
		self.df_Yield = pd.read_csv(os.path.join("data","USYield.csv"),
					header = None, delimiter = ",", index_col=0)
		T = np.array(self.df_Yield.index)
		yields = np.array(self.df_Yield.loc[:,1])
		self.yields_interp = interpolate.interp1d(T.squeeze(), yields, 'quadratic', fill_value='extrapolate')
		
		# cap
		self.df_USD_Libor = pd.read_csv(os.path.join("data","USLibor.csv"), delimiter=",", index_col=0)
		self.libor_spot = self.df_USD_Libor.loc["3M"].squeeze()/100
		self.df_cap = pd.read_csv(os.path.join("data","USCap.csv"), delimiter=",", index_col=0, header=0)
		self.delta = 0.25
		#P_3M = self.USD_bond_price(0.25)
		#self.libor_spot = (1.0-P_3M)/(self.delta*P_3M)
		self.K_cap = self.df_cap.iloc[0,1]/100.0
		self.price_cap = self.df_cap.iloc[0,0]/100.0
		
	def USD_bond_price(self,T):
		T_yield = self.yields_interp(T)
		price = (1+T_yield/100)**(-T)
		return price

	def USD_forward_rate(self,T):
		return derivative(lambda x:-np.log(self.USD_bond_price(x)), T, dx=self.dx)
	

class HoLee_calibrate(RUSD_calibrate):
	def __init__(self):
		RUSD_calibrate.__init__(self)
		T_test = 1.0
		self.sigma = self.calib_sigma_HL(T_test)
		#self.sigma = 0.015

		self.theta = self.theta_HL
		
	def caplet_from_sigma_HL(self,sigma, t1):

		coef = (1+self.K_cap)/self.delta
		strike = 1.0/(1+self.K_cap)
		t2 = t1+self.delta
		PT = self.USD_bond_price(t1)
		PS = self.USD_bond_price(t2)
		if t1 == 0.0:
			return max(0, self.libor_spot-self.K_cap)*PS
			#return max(0, 1.0/self.delta*(1.0/PS-1)-self.K_cap)
		sigma_p = sigma*self.delta*np.sqrt(t1)
		d1 = 1.0/sigma_p * np.log(PS/PT/strike)+0.5*sigma_p
		d2 = d1 - sigma_p
		N_minusd1 = norm.cdf(-d1)
		N_minusd2 = norm.cdf(-d2)
		put = PT*strike*N_minusd2 - PS*N_minusd1
		return put * coef
		
	def cap_from_sigma_HL(self,sigma, T):
		tt = np.linspace(0.0,float(T)-self.delta,int(float(T)//self.delta))
		cap_price = 0
		for t1 in tt:
			cap_price += self.caplet_from_sigma_HL(sigma, t1)
		return cap_price	
	def calib_sigma_HL(self,T_test):
		sol = optimize.root(lambda x:self.cap_from_sigma_HL(x, T_test)-self.price_cap, 0.02, method='hybr')	
		print("HoLee sigma:",sol.x)
		sigma_HL = sol.x
		return sigma_HL
		
	def theta_HL(self,t):
		return derivative(lambda x:-np.log(self.USD_bond_price(x)), t, dx=self.dx, n=2, order = 3) + t*self.sigma**2

	def Libor(self, t, r, delta=0.25):
		T = t+delta
		first = self.USD_bond_price(T)/self.USD_bond_price(t)
		second = np.exp(delta*self.USD_forward_rate(t) - self.sigma**2/2*t*delta**2-delta*r)
		P = first*second
		return (1-P)/(delta*P)
		
class HullWhite_calibrate(RUSD_calibrate):
	def __init__(self):
		RUSD_calibrate.__init__(self)	
		self.a = 0.03
		T_test = 1.0
		self.sigma = self.calib_sigma_HW(T_test)
		#self.sigma = 0.015
		self.theta = self.theta_HW
		
	def caplet_from_sigma_HW(self, sigma, t1):
		coef = (1+self.K_cap)/self.delta
		strike = 1.0/(1+self.K_cap)
		t2 = t1+self.delta
		PT = self.USD_bond_price(t1)
		PS = self.USD_bond_price(t2)
		if t1 == 0.0:
			return max(0, self.libor_spot-self.K_cap) * PS
		sigma_p = 1.0/self.a*(1-np.exp(-self.a*self.delta))*np.sqrt(sigma**2/(2*self.a)*(1-np.exp(-2*self.a*self.delta)))
		d1 = 1.0/sigma_p * np.log(PS/PT/strike)+0.5*sigma_p
		d2 = d1 - sigma_p
		N_minusd1 = norm.cdf(-d1)
		N_minusd2 = norm.cdf(-d2)
		put = PT*strike*N_minusd2 - PS*N_minusd1
		return put * coef
		
	def cap_from_sigma_HW(self, sigma, T):
		tt = np.linspace(0.0,float(T)-self.delta,int(float(T)//self.delta))
		cap_price = 0
		for t1 in tt:
			cap_price += self.caplet_from_sigma_HW(sigma, t1)
		return cap_price
	def calib_sigma_HW(self, T_test):
		sol = optimize.root(lambda x:self.cap_from_sigma_HW(x, T_test)-self.price_cap, 0.2, method='hybr')	
		print("HullWhite sigma:",sol.x)
		sigma_HW = sol.x
		return sigma_HW
		
	def B_HW(self,T, t=0):
		return (1.0/self.a)*(1-np.exp(-self.a*(T-t)))
	
	def g_HW(self, T):
		return self.sigma**2/2 * self.B_HW(T)**2
	
	def theta_HW(self, t):
		first = derivative(lambda x:-np.log(self.USD_bond_price(x)), t, dx=self.dx, n=2)
		second = derivative(lambda x:self.g_HW(x), t, dx=self.dx, n=1)
		third = self.a * (derivative(lambda x:-np.log(self.USD_bond_price(x)), t, dx=self.dx, n=1, order=3) + self.g_HW(t))
		return first+second+third
		
	def Libor(self, t, r, delta=0.25):
		T = t+delta
		B = self.B_HW(T,t)
		first = self.USD_bond_price(T)/self.USD_bond_price(t)
		second = np.exp(B*self.USD_forward_rate(t) - self.sigma**2/(4*self.a)*B**2*(1-np.exp(-2*self.a*t)) - B*r)
		P = first*second
		return (1-P)/(delta*P)
		
if __name__ == "__main__":
	clb = HoLee_calibrate()
	tt = np.linspace(0,30,300)
	plt.plot(tt, clb.USD_bond_price(tt))
	plt.title("USD bond price")
	plt.show()
	delta = 0.25

	tt = np.linspace(delta,10.0,int(float(10.0)//delta))
	forwards = np.zeros(len(tt))
	for i in range(0,len(tt)):
		t1 = tt[i]
		forwards[i] = clb.USD_forward_rate(t1)
	plt.plot(tt, forwards)
	plt.title("forwards")
	plt.show()
	
	tt = np.linspace(0.01,float(10.0),100)
	theta_HL_tt = []
	forwards = []
	for t in tt:
		theta_HL_tt.append(clb.theta_HL(t))
	theta_HL_tt= np.array(theta_HL_tt)
	
	plt.plot(tt, theta_HL_tt)
	plt.title("HoLee theta")
	plt.show()
	
	clb2 = HullWhite_calibrate()
	theta_HW_tt = []

	for t in tt:
		theta_HW_tt.append(clb2.theta_HW(t))
	theta_HW_tt= np.array(theta_HW_tt)
	plt.plot(tt, theta_HW_tt)
	plt.title("HullWhite theta")
	plt.show()