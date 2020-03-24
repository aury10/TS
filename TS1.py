
# code du 22/03/2020
# Serie temporelle

#  aury kt

'''An End-to-End Project on Time Series Analysis and Forecasting with Python
'''

# data: Superstore sales data

'''There are several categories in the Superstore sales data, 
time series analysis and forecasting for furniture sales.
'''

# import library

import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib


matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

# read data from path

df = pd.read_excel('Superstore.xls')
print(df.head())

# iterating the columns 
'''for col in df.columns: 
    print(col) '''

# names col
col = list(df.columns.values) 
print(col)

'''
['Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode', 
'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 
'State', 'Postal Code', 'Region', 'Product ID', 'Category', 
'Sub-Category', 'Product Name', 'Sales', 'Quantity', 'Discount', 'Profit']
'''

# 2) Some extraction and statistics

print(df.describe())
cat_data = df['Category']
print(cat_data )

# furniture
furniture = df[df['Category']== 'Furniture']
times_stamps_furniture = furniture['Order Date'].min(), furniture['Order Date'].max()
print(times_stamps_furniture)
# (Timestamp('2014-01-06 00:00:00'), Timestamp('2017-12-30 00:00:00'))
# For this shop we have around 4 years furniture data set 


# Office_Supplies furniture
Office_Supplies = df[df['Category']== 'Office Supplies']
print( Office_Supplies)
times_stamps_Office_Supplies = Office_Supplies['Order Date'].min(), Office_Supplies['Order Date'].max()
print(times_stamps_Office_Supplies)

# Same conclusion as furniture


# 3 - Data processing 
'''
In this step i chech na value and remove not important colunm for this analysis
'''
#col = list(df.columns.values) 
cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 
'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 
'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']

furniture.drop(cols, axis=1, inplace=True)
furniture = furniture.sort_values('Order Date')

# find NA
furniture.isnull().sum()

# Indexing with Time Series Data

furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()
furniture = furniture.set_index('Order Date')
print(furniture.index)


# Average of sales by day to get a nice plot
y_sm = furniture['Sales'].resample('SM').mean()
y_ms = furniture['Sales'].resample('MS').mean()


# B         business day frequency
# C         custom business day frequency (experimental)
# D         calendar day frequency
# W         weekly frequency
# M         month end frequency
# SM        semi-month end frequency (15th and end of month)
# BM        business month end frequency
# CBM       custom business month end frequency
# MS        month start frequency
# SMS       semi-month start frequency (1st and 15th)
# BMS       business month start frequency
# CBMS      custom business month start frequency
# Q         quarter end frequency
# BQ        business quarter endfrequency
# QS        quarter start frequency
# BQS       business quarter start frequency
# A         year end frequency
# BA, BY    business year end frequency
# AS, YS    year start frequency
# BAS, BYS  business year start frequency
# BH        business hour frequency
# H         hourly frequency
# T, min    minutely frequency
# S         secondly frequency
# L, ms     milliseconds
# U, us     microseconds
# N         nanoseconds


print(y_sm['2014':])

#Visualizing Furniture Sales Time Series Data

y_ms.plot(figsize=(15, 6))
plt.xlabel('years order')
plt.ylabel('values')
plt.show()

y_sm.plot(figsize=(15, 6))
plt.show()

# Dapres legraphique nous pouvons observer un effet de saisonalité
# Au debut de chaque années nous observons queles vente sont base au debut de chaque années

y_sm['2017':].plot(figsize=(15, 6))
plt.xlabel('years order')
plt.ylabel('values')
plt.show()

# Nous avons le pic devente en decembre et les ventes les plus bases en janvier

# Decomposition de la serie temporelle  (trend, seasonality, and noise. )

decomposition_ts = sm.tsa.seasonal_decompose(y_ms, model='additive')

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
fig = decomposition_ts.plot()
plt.show()

#  On confirme l'effet de la saisonnalité

# Time series forecasting with ARIMA

p = d = q = range(0, 2)
pdq_coef = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq_coef[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq_coef[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq_coef[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq_coef[2], seasonal_pdq[4]))

# En theorie le choix du meilleur model est possible avec le critere BIC ou AIC

for param in pdq_coef:
	for param_seasonal in seasonal_pdq:
		try:
			mod = sm.tsa.statespace.SARIMAX(y_ms, 
				order=param, 
				seasonal_order=param_seasonal,
				 enforce_stationarity=False, 
				 enforce_invertibility=False)
			results = mod.fit()
			print('ARIMA{}x{}12 - AIC:{}'.format(param, 
			param_seasonal, results.aic))
		except:
			continue

# best model ARIMA(1, 1, 1)x(1, 1, 1, 12)12 - AIC:283.36610170351906



# Fitting the ARIMA model


mod = sm.tsa.statespace.SARIMAX(y_sm,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
resultss = mod.fit()
print(resultss.summary().tables[1])


# Diagnostique des erreurs (normalité correlation, )
resultss.plot_diagnostics(figsize=(16, 8))
plt.show()

# le model n'est pas trop bon car les erreurs sont presque normale


# Validating forecasts
#nous comparons les ventes prévues aux ventes réelles de la série chronologique
# et nous fixons les prévisions à partir du 2017-01-01 
#jusqu'à la fin des données.


pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y_sm['2014':].plot(label='observed values')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()



#print(furniture.index)


#fur = furniture['Order Date', 'Sales']
#print(fur)

#fur.plot(figsize=(15, 6))
#plt.show()








