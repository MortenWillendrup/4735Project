Python files:
	1. Main.py 
		the main function. runs monte carlo.
	2. correlation.py
		estimates the correlations (between FX and stock, and between stock and short rates), and FX vol
	3. derivative.py
		helper functions that calculate numerical differentiation of interpolated or parametric functions
	4.  EUR_shortrate_cali.py
		calibrates parameters or Euro short rate models
	5. InterestRate_EUR.py
		some helper functions for simulating ho-lee and hull-white models for EUR short rates
	6. USD_shortrate_cali.py
		calibrates parameters or USD short rate models
	7. InterestRate_USD.py
		some helper functions for simulating ho-lee and hull-white models for USD short rates
	8. Stock_cali.py 
		calibrates parameters for log normal model
	9. Stock_BS.py 
		some helper functions for simulating stock prices under log normal (black scholes)
	10. Stock_cali_LV.py (very slow)
		calibrates parameters for local vol model using SVI
	11. Stock_LV.py (not recommended)
		some helper functions for simulating stock prices under Local vol


Data files:
	1. ^STOXX50E.csv
		historical stock price, used for estimating correlations
	2. DEXUSEU.csv
		historical FX, used for estimating correlations and FX vol
	3. General.csv
		containing some general parameters such as today's USD short rate and EUR short rate
	4. GERYield.csv
		Germany yield
	5. Stock.csv
		stock parameters such as today's price and dividend rate
	6. StockCall_longterm.csv, StockCall_midterm.csv, StockCall_shortterm.csv
		stock call surface
	7. USCap.csv
		US cap prices
		first column is term in years
		second column is price in basis points
		third column is strike in percentage
	7. USCaplets.csv
		artificial US Caplets price. Use either USCap.csv or USCaplets.csv
		first column is term in years
		second column is price in basis points
		third column is strike in percentage
	8. USLibor.csv
		today's US libor 
	9. USYield.csv
		US Yield 