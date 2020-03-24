
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

# 1) read data from path

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
furniture.drop('Row ID', 1)
furniture = furniture.sort_values('Order Date')
print(furniture.isnull().sum())

furniture = furniture.groupby('Order Date')
print(furniture )


# 4 - Indexing with Time Series Data

















