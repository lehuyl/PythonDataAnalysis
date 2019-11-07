# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:08:28 2019

@author: Steven
"""

import pandas as pd

totals = pd.read_csv('./e1/totals.csv').set_index(keys=['name'])
counts = pd.read_csv('./e1/counts.csv').set_index(keys=['name'])

cityPrecipitation = pd.Index(totals.sum(axis=1))
print('Row with lowest total precipitation:')
print(cityPrecipitation.argmin())

monthlyPrecipitation = pd.Index(totals.sum(axis=0))
monthlyCount  = pd.Index(counts.sum(axis=0))
averagePerMonth = monthlyPrecipitation/monthlyCount
print('Average precipitation in each month')
print(averagePerMonth.values)

totalCount = pd.Index(counts.sum(axis=1))
averagePerCity = cityPrecipitation/totalCount
print('Average precipitation in each city')
print(averagePerCity.values)

print(cityPrecipitation)