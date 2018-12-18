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

class REUR_calibrate:
	"""
	Base class for HoLee class and HullWhite class
	"""
	def __init__(self):
		"""
		Read Euro(GER) yield data and interpolate
		"""
		self.dx = 1e-1
		self.df_Yield = pd.read_csv(os.path.join("data","GERYield.csv"),
					header = None, delimiter = "\t", index_col=0)
		T = np.array(self.df_Yield.index)
		yields = np.array(self.df_Yield.loc[:,1])
		self.yields_interp = interpolate.interp1d(T.squeeze(), yields, 'cubic', fill_value='extrapolate')
		
		self.delta = 0.25
		self.sigma = 0.0
		
	def EUR_bond_price(self,T):
		"""
		Euro bond price with maturity T
		"""
		T_yield = self.yields_interp(T)
		price = (1+T_yield/100)**(-T)
		return price

	def EUR_forward_rate(self,T):
		"""
		Euro instantaneous forward rate at time T, estimated at time 0	
		"""
		return derivative(lambda x:-np.log(self.EUR_bond_price(x)), T, dx=self.dx)
	

class HoLee_calibrate_EUR(REUR_calibrate):
	"""
	HoLee calibrate, extending REUR_calibrate, assuming sigma = 0 (deterministic rates)
	"""
	def __init__(self):
		REUR_calibrate.__init__(self)
		self.theta = self.theta_HL
		
	def theta_HL(self,t):
		"""
		As in Bjork
		"""
		return derivative(lambda x:-np.log(self.EUR_bond_price(x)), t, dx=self.dx, n=2, order = 3) + t*self.sigma**2

	
class HullWhite_calibrate_EUR(REUR_calibrate):
	"""
	HullWhite calibrate, extending REUR_calibrate, assuming sigma = 0 (deterministic rates)
	"""
	def __init__(self):
		REUR_calibrate.__init__(self)	
		self.a = 0.03
		self.theta = self.theta_HW
		
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
		As in Bjork
		"""
		first = derivative(lambda x:-np.log(self.EUR_bond_price(x)), t, dx=self.dx, n=2)
		second = derivative(lambda x:self.g_HW(x), t, dx=self.dx, n=1)
		third = self.a * (derivative(lambda x:-np.log(self.EUR_bond_price(x)), t, dx=self.dx, n=1, order=3) + self.g_HW(t))
		return first+second+third
		
		
#==============================================================================
# if __name__ == "__main__":
# 	clb = HoLee_calibrate_EUR()
# 	tt = np.linspace(0,30,300)
# 	plt.plot(tt, clb.EUR_bond_price(tt))
# 	plt.title("EUR bond price")
# 	plt.show()
# 	delta = 0.25
# 
# 	tt = np.linspace(delta,10.0,int(float(10.0)//delta))
# 	forwards = np.zeros(len(tt))
# 	for i in range(0,len(tt)):
# 		t1 = tt[i]
# 		forwards[i] = clb.EUR_forward_rate(t1)
# 	plt.plot(tt, forwards)
# 	plt.title("forwards")
# 	plt.show()
# 	
# 	tt = np.linspace(0.01,float(10.0),100)
# 	theta_HL_tt = []
# 	forwards = []
# 	for t in tt:
# 		theta_HL_tt.append(clb.theta_HL(t))
# 	theta_HL_tt= np.array(theta_HL_tt)
# 	
# 	plt.plot(tt, theta_HL_tt)
# 	plt.title("HoLee theta")
# 	plt.show()
# 	
# 	clb2 = HullWhite_calibrate_EUR()
# 	theta_HW_tt = []
# 
# 	for t in tt:
# 		theta_HW_tt.append(clb2.theta_HW(t))
# 	theta_HW_tt= np.array(theta_HW_tt)
# 	plt.plot(tt, theta_HW_tt)
# 	plt.title("HullWhite theta")
# 	plt.show()
#==============================================================================
