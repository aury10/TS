
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
import pandas as pd
import matplotlib
import statsmodels.api as sm

# read data from path


df = pd.read_excel('Superstore.xls')

print(df.head())
print(df.describe())

# iterating the columns 
'''for col in df.columns: 
    print(col) '''

# names col
nom_col = list(df.columns.values) 
print(nom_col)

cat_data = df['Category']
print(cat_data.describe())

furniture = df[df['Category']== 'Furniture']

print(furniture )