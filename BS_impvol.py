# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 09:03:01 2018

@author: dvfzr
"""

import numpy as np
import scipy.stats as stats
import pandas as pd
import os


df_s = pd.read_csv(os.path.join("data","StockCall_shortterm.csv"),index_col=0)
df_s.dropna(inplace = True)
df_m = pd.read_csv(os.path.join("data","StockCall_midterm.csv"),index_col=0)
df_m.dropna(inplace = True)
df_l = pd.read_csv(os.path.join("data","StockCall_longterm.csv"),index_col=0)
df_l.dropna(inplace = True)


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
#print(impvol(100,100,np.exp(-0.05*30/365),30/365,0,1.358))

def error(sigma, df):
    k = df.index.values
    s = np.full((len(k),),3058)
    q = 0.01
    err = 0
    for i in range(0, len(df.columns)):
        price = df.iloc[:,i].values
        T = int(df.columns.values[i])/365
        p = np.exp(-T * 0.05)
        err += sum((bs_price(s,k,p,T,q,sigma) - price)**2)
    
    return err,sigma  

def optimal_impvol(df):
    k = df.index.values
    s = np.full((len(k),),3058)
    q = 0.01
    vol_arr = np.zeros(1,)
    min_err = float('inf')
    min_vol = 0
    for i in range(0,len(df.columns)):
        price = df.iloc[:,i].values
        T = int(df.columns.values[i])/365
        p = np.exp(-T * 0.05)
        ret = vec_impvol(s,k,p,T,q,price)
        vol_arr = np.append(vol_arr,ret)
        vol_arr = vol_arr[np.where(vol_arr>0)]
        
    for i in range(0,len(vol_arr)):
        ret = error(vol_arr[i],df)
        curr_err = ret[0]
        curr_vol = ret[1]
        if(curr_err < min_err):
            min_err = curr_err
            min_vol = curr_vol
    return min_vol


if __name__ == '__main__':
    vec_impvol = np.vectorize(impvol)
    print(optimal_impvol(df_s))