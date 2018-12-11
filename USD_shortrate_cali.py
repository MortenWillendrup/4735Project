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
	"""
	Base class for HoLee class and HullWhite class
	"""
	def __init__(self, use_caplets=True):
		"""
		1. Read US yield data and interpolate
		2. read US Libor data and cap data, or artifical caplets data
		For cap data, we view it as the sum of multiple caplets
		For instance, for 1-year cap we view it as the sum of 4 caplets, thus 4 bond put
		we solve for 'sigma' to match that cap price.
		"""
		self.dx = 1e-2
		self.delta = 0.25
		self.use_caplets = use_caplets

		self.df_Yield = pd.read_csv(os.path.join("data","USYield.csv"),
					header = None, delimiter = ",", index_col=0)
		T = np.array(self.df_Yield.index)
		yields = np.array(self.df_Yield.loc[:,1])
		self.yields_interp = interpolate.interp1d(T.squeeze(), yields, 'quadratic', fill_value='extrapolate')
		three_Month_yield = self.df_Yield.loc[self.delta].squeeze()/100
		three_Month_bond_price = (1+three_Month_yield)**(-self.delta)
		self.libor_spot = (1.0-three_Month_bond_price)/(self.delta*three_Month_bond_price)
		# cap
		#self.df_USD_Libor = pd.read_csv(os.path.join("data","USLibor.csv"), delimiter=",", index_col=0)
		#self.libor_spot = self.df_USD_Libor.loc["3M"].squeeze()/100
		
		self.df_cap = pd.read_csv(os.path.join("data","USCap.csv"), delimiter=",", index_col=0, header=0)
		#P_3M = self.USD_bond_price(0.25)
		#self.libor_spot = (1.0-P_3M)/(self.delta*P_3M)
		self.K_cap = self.df_cap.iloc[0,1]/100.0
		self.price_cap = self.df_cap.iloc[0,0]/100.0
		
		self.df_caplets = pd.read_csv(os.path.join("data","USCaplets.csv"), delimiter=",", header=0)
		
		
	def USD_bond_price(self,T):
		"""
		USD treasury bond price with maturity T
		"""
		T_yield = self.yields_interp(T)
		price = (1+T_yield/100)**(-T)
		return price

	def USD_forward_rate(self,T):
		"""
		USD instantaneous forward rate at time T, estimated at time 0	
		"""
		return derivative(lambda x:-np.log(self.USD_bond_price(x)), T, dx=self.dx)
	

class HoLee_calibrate(RUSD_calibrate):
	"""
	HoLee calibrate, extending RUSD_calibrate
	"""
	def __init__(self, use_caplets=True):
		RUSD_calibrate.__init__(self, use_caplets)
		T_test = 1.0
		self.sigma = self.calib_sigma_HL(T_test)
		#self.sigma = 0.015

		self.theta = self.theta_HL
		
	def caplet_from_sigma_HL(self,sigma, t1, K_cap):
		"""		
		calculate caplet price using black formula
		sigma and t1 given
		t2 = t1 + delta (e.g. 3 month)
		strike = K_cap
		"""
		coef = (1+self.delta*K_cap)/self.delta
		strike = 1.0/(1+self.delta*K_cap)
		t2 = t1+self.delta
		PT = self.USD_bond_price(t1)
		PS = self.USD_bond_price(t2)
		if t1 == 0.0:
			return max(0, self.libor_spot-K_cap)*PS
			#return max(0, 1.0/self.delta*(1.0/PS-1)-self.K_cap)
		sigma_p = sigma*self.delta*np.sqrt(t1)
		d1 = 1.0/sigma_p * np.log(PS/PT/strike)+0.5*sigma_p
		d2 = d1 - sigma_p
		N_minusd1 = norm.cdf(-d1)
		N_minusd2 = norm.cdf(-d2)
		put = PT*strike*N_minusd2 - PS*N_minusd1
		return put * coef
		
	def cap_from_sigma_HL(self,sigma, T):
		"""
		Calculate cap price by adding 4 caplets prices
		sigma and T given
		"""
		tt = np.linspace(0.0,0.0+float(T)-self.delta,int(float(T)//self.delta))
		cap_price = 0
		for t1 in tt:
			cap_price += self.caplet_from_sigma_HL(sigma, t1, self.K_cap)
		return cap_price	
	def calib_sigma_HL(self,T_test):
		"""
		solve for sigma by matching cap market price (if self.use_caplets = False)
		solve for simga by matching caplet market price (if self.use_caplets = True)
		"""
		if not self.use_caplets:
			sol = optimize.root(lambda x:self.cap_from_sigma_HL(x, T_test)-self.price_cap, 0.1, method='hybr')	
			print("HoLee sigma:",sol.x)
			sigma_HL = sol.x
		else:		
			t1 = np.array(self.df_caplets.iloc[:,0])
			l_strike = np.array(self.df_caplets.iloc[:,2])/100
			prices = np.array(self.df_caplets.iloc[:,1])/100
			def caplet_price_mse(sig):
				res = 0
				N = len(t1)
				for i in range(N):
					res += (prices[i]-self.caplet_from_sigma_HL(sig,t1[i],l_strike[i]))**2
				return res/N
			sol = optimize.minimize(lambda x:caplet_price_mse(x), 0.2, method='BFGS', bounds = (0,None))	
			print("HoLee sigma:",sol.x)
			sigma_HL = sol.x
		return sigma_HL
		
	def theta_HL(self,t):
		"""
		After getting sigma, we can get theta following Bjork
		"""
		return derivative(lambda x:-np.log(self.USD_bond_price(x)), t, dx=self.dx, n=2, order = 3) + t*self.sigma**2

	def Libor(self, t, r, delta=0.25):
		"""
		given t, r_t, we can simulate Libor(t,t,t+delta) as in Bjork
		"""
		T = t+delta
		first = self.USD_bond_price(T)/self.USD_bond_price(t)
		second = np.exp(delta*self.USD_forward_rate(t) - self.sigma**2/2*t*delta**2-delta*r)
		P = first*second
		return (1-P)/(delta*P)
		
class HullWhite_calibrate(RUSD_calibrate):
	"""
	HullWhite calibrate, extending RUSD_calibrate
	"""
	def __init__(self, use_caplets=True):
		RUSD_calibrate.__init__(self, use_caplets=use_caplets)	
		self.a = 0.03
		T_test = 1.0
		self.sigma = self.calib_sigma_HW(T_test)
		#self.sigma = 0.015
		self.theta = self.theta_HW
		
	def caplet_from_sigma_HW(self, sigma, t1, K_cap):
		"""		
		calculate caplet price using black formula
		sigma and t1 given
		t2 = t1 + delta (e.g. 3 month)
		strike = self.K_cap
		"""
		coef = (1+self.delta*K_cap)/self.delta
		strike = 1.0/(1+self.delta*K_cap)
		t2 = t1+self.delta
		PT = self.USD_bond_price(t1)
		PS = self.USD_bond_price(t2)
		if t1 == 0.0:
			return max(0, self.libor_spot-K_cap) * PS
		sigma_p = 1.0/self.a*(1-np.exp(-self.a*self.delta))*np.sqrt(sigma**2/(2*self.a)*(1-np.exp(-2*self.a*self.delta)))
		d1 = 1.0/sigma_p * np.log(PS/PT/strike)+0.5*sigma_p
		d2 = d1 - sigma_p
		N_minusd1 = norm.cdf(-d1)
		N_minusd2 = norm.cdf(-d2)
		put = PT*strike*N_minusd2 - PS*N_minusd1
		return put * coef
		
	def cap_from_sigma_HW(self, sigma, T):
		"""
		Calculate cap price by adding 4 caplets prices
		sigma and T given
		"""
		tt = np.linspace(0.0,0.0+float(T)-self.delta,int(float(T)//self.delta))
		cap_price = 0
		for t1 in tt:
			cap_price += self.caplet_from_sigma_HW(sigma, t1, self.K_cap)
		return cap_price
	def calib_sigma_HW(self, T_test):
		"""
		solve for sigma by matching cap market price
		"""
		if self.use_caplets == False:
			sol = optimize.root(lambda x:self.cap_from_sigma_HW(x, T_test)-self.price_cap, 0.2, method='hybr')	
			print("HullWhite sigma:",sol.x)
			sigma_HW = sol.x
		else:		
			t1 = np.array(self.df_caplets.iloc[:,0])
			l_strike = np.array(self.df_caplets.iloc[:,2])/100
			prices = np.array(self.df_caplets.iloc[:,1])/100
			def caplet_price_mse(sig):
				res = 0
				N = len(t1)
				for i in range(N):
					res += (prices[i]-self.caplet_from_sigma_HW(sig,t1[i],l_strike[i]))**2
				return res/N
			sol = optimize.minimize(lambda x:caplet_price_mse(x), 0.2, method='BFGS', bounds = (0,None))	
			print("HullWhite sigma:",sol.x)
			sigma_HW = sol.x		
		return sigma_HW
		
	def B_HW(self,T, t=0):
		"""
		As in Bjork
		"""
		return (1.0/self.a)*(1-np.exp(-self.a*(T-t)))
	
	def g_HW(self, T):
		"""
		As in Bjork
		"""
		return self.sigma**2/2 * self.B_HW(T)**2
	
	def theta_HW(self, t):
		"""
		After getting sigma, we can get theta following Bjork
		"""
		first = derivative(lambda x:-np.log(self.USD_bond_price(x)), t, dx=self.dx, n=2)
		second = derivative(lambda x:self.g_HW(x), t, dx=self.dx, n=1)
		third = self.a * (derivative(lambda x:-np.log(self.USD_bond_price(x)), t, dx=self.dx, n=1, order=3) + self.g_HW(t))
		return first+second+third
		
	def Libor(self, t, r, delta=0.25):
		"""
		given t, r_t, we can simulate Libor(t,t,t+delta) as in Bjork
		"""
		T = t+delta
		B = self.B_HW(T,t)
		first = self.USD_bond_price(T)/self.USD_bond_price(t)
		second = np.exp(B*self.USD_forward_rate(t) - self.sigma**2/(4*self.a)*B**2*(1-np.exp(-2*self.a*t)) - B*r)
		P = first*second
		return (1-P)/(delta*P)
		
if __name__ == "__main__":
	clb = HoLee_calibrate(use_caplets=False)
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
	
	clb2 = HullWhite_calibrate(use_caplets=False)
	theta_HW_tt = []

	for t in tt:
		theta_HW_tt.append(clb2.theta_HW(t))
	theta_HW_tt= np.array(theta_HW_tt)
	plt.plot(tt, theta_HW_tt)
	plt.title("HullWhite theta")
	plt.show()