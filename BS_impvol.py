# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 09:03:01 2018

@author: dvfzr
"""

import math
import numpy as np
import scipy.stats as stats
import pandas as pd
import os
import sys
from scipy import interpolate


# Import raw data
df_stock_stats = pd.read_csv(os.path.join("data", "Stock.csv"), index_col=0, header=None)
spot_str = df_stock_stats.loc['Price'].squeeze()
spot = float(spot_str)
div_str = df_stock_stats.loc['div'].squeeze()
q = float(div_str.strip("%"))/100
df_s = pd.read_csv(os.path.join("data","StockCall_shortterm.csv"),index_col=0)
df_s.dropna(inplace = True)
df_m = pd.read_csv(os.path.join("data","StockCall_midterm.csv"),index_col=0)
df_m.dropna(inplace = True)
df_l = pd.read_csv(os.path.join("data","StockCall_longterm.csv"),index_col=0)
df_l.dropna(inplace = True)

# Drop deep ITM options data which have little liquidity
df_s_trunc = df_s[13:]
df_m_trunc = df_m[17:]
df_l_trunc = df_l[11:]
df_all = pd.concat([df_s_trunc, df_m_trunc], axis=1)

# Euro bond price
df_Yield = pd.read_csv(os.path.join("data","GERYield.csv"),
        header = None, delimiter = "\t", index_col=0)
T = np.array(df_Yield.index)
yields = np.array(df_Yield.loc[:,1])
yields_interp = interpolate.interp1d(T.squeeze(), yields, 'cubic', fill_value='extrapolate')
def Euro_bond_price(T):
    T_yield = yields_interp(T)
    price = (1+T_yield/100)**(-T)
    return price

# Calculate the Black Scholes price of European call option
def bs_price(s, k, p, T, q, sigma):
    F = s * np.exp(-q * T)/p
    d1 = (np.log(F / k) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = p * (F * stats.norm.cdf(d1) - k *  stats.norm.cdf(d2))
    return price


# Calculate implied vol using bisection method
def impvol(s, k, p, T, q, price, tol = 1e-6, max_iteration = 1e4):
    sig = 0.2
    sig_u = 1
    sig_d = 1e-4
    it = 0
    err = bs_price(s, k, p, T, q, sig) - price

    while abs(err) > tol and it < max_iteration:
        if err < 0:
            sig_d = sig
            sig = (sig_u + sig)/2
        else:
            sig_u = sig
            sig = (sig_d + sig)/2
        
        err = bs_price(s, k, p, T, q, sig) - price
        it += 1
    
    if it == max_iteration:
        return -1
    else:
        return sig


def error_naiveloop(spot,q,sigma,df):
    k = df.index.values
    T = df.columns.values
    err = 0
    for i in range(0,len(k)):
        for j in range(0,len(T)):
            price = df.iloc[i,j]
            if(pd.isna(price) == True):
                continue
            maturity = int(T[j])/365
            p = Euro_bond_price(maturity)
            err += (bs_price(spot,k[i],p,maturity,q,sigma)-price) **2
    return err,sigma  

def optimal_impvol_naiveloop(spot,q,df):
    k = df.index.values
    T = df.columns.values
    s = spot
    vol_arr = np.zeros(1,)
    min_err = float('inf')
    min_vol = 0
    for i in range(0,len(k)):
        for j in range(0,len(T)):
            price = df.iloc[i,j]
            if(pd.isna(price) == True):
                continue
            maturity = int(T[j])/365
            p = Euro_bond_price(maturity)
            ret = impvol(s,k[i],p,maturity,q,price)
            vol_arr = np.append(vol_arr,ret)
            vol_arr = vol_arr[vol_arr>0]
        
    for i in range(0,len(vol_arr)):
        ret = error_naiveloop(s,q,vol_arr[i],df)
        curr_err = ret[0]
        curr_vol = ret[1]
        if(curr_err < min_err):
            min_err = curr_err
            min_vol = curr_vol
            print(min_vol)
    return min_vol


if __name__ == '__main__':
    print("Optimal BS implied volatility: ",optimal_impvol_naiveloop(spot,q,df_all))