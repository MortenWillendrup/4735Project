# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 23:34:24 2018

@author: andy
"""

import numpy as np
import pandas as pd
import os

class Correlation_calibrate:
	def __init__(self):
		df_FX = pd.read_csv(os.path.join("data","DEXUSEU.csv"), index_col=0, parse_dates=[0], na_values=["."])
		df_FX['FX_log_ret'] = np.log(df_FX.DEXUSEU) - np.log(df_FX.DEXUSEU.shift(1))
		df_Stock = pd.read_csv(os.path.join("data","^STOXX50E.csv"), index_col=0, parse_dates=[0])
		df_Stock = df_Stock[['Adj Close']]
		df_Stock['Stock_log_ret'] = np.log(df_Stock['Adj Close']) - np.log(df_Stock['Adj Close'].shift(1))
		
		
		df_US_shortrate = pd.read_csv(os.path.join("data","USDLiborHist.csv"), index_col=0, parse_dates=[0], dayfirst=True)
		df_US_shortrate = df_US_shortrate.loc[:,"ON"]/100
		df_US_shortrate.sort_index(inplace=True)
		df_US_shortrate_diff = df_US_shortrate.diff()
		df_US_shortrate_diff = df_US_shortrate_diff.loc[df_US_shortrate.index>="2018-01-01"]
		df_US_shortrate_diff.name = 'ON_diff'
		
		
		df_all = pd.concat([df_FX, df_Stock, df_US_shortrate_diff],axis=1, join='inner')
		df_all = df_all[['FX_log_ret','Stock_log_ret', 'ON_diff']]
		df_all = df_all.loc[df_all.index>='2018-01-01']
		
		
		cov_mat = df_all.cov()
		
		var_FX_daily = cov_mat.loc['FX_log_ret','FX_log_ret']
		var_Stock_daily = cov_mat.loc['Stock_log_ret','Stock_log_ret']
		var_ShortRate_daily = cov_mat.loc['ON_diff','ON_diff']
		cov_FX_Stock = cov_mat.loc['FX_log_ret','Stock_log_ret']
		cov_FX_ShortRate = cov_mat.loc['FX_log_ret','ON_diff']
		cov_Stock_ShortRate = cov_mat.loc['Stock_log_ret','ON_diff']
		
		
		sig_FX_daily = np.sqrt(var_FX_daily)
		sig_Stock_daily = np.sqrt(var_Stock_daily)
		sig_ShortRate_daily = np.sqrt(var_ShortRate_daily)
		
		cor_FX_Stock = cov_FX_Stock / (sig_Stock_daily*sig_FX_daily)
		cor_FX_ShortRate = cov_FX_ShortRate / (sig_ShortRate_daily*sig_FX_daily)
		cor_Stock_ShortRate = cov_Stock_ShortRate / (sig_ShortRate_daily*sig_Stock_daily)
		
		sig_FX_yearly = sig_FX_daily * np.sqrt(252)
		sig_Stock_yearly = sig_Stock_daily * np.sqrt(252)
		sig_ShortRate_yearly = sig_ShortRate_daily * np.sqrt(252)
		
		print("FX annual vol:", sig_FX_yearly)
		print("Stock annual vol:", sig_Stock_yearly)
		print("Short Rate annual vol:", sig_ShortRate_yearly)
		print("cor FX Stock:", cor_FX_Stock)
		print("cor FX ShortRate:", cor_FX_ShortRate)
		print("cor ShortRate Stock:", cor_Stock_ShortRate)
		
		self.rho_XS = cor_FX_Stock
		self.rho_RS = cor_Stock_ShortRate
		self.vol_FX = sig_FX_yearly

