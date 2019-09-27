# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 22:27:21 2019

@author: Steven
"""

import numpy as np

data = np.load('./e1/monthdata.npz')

totals = data['totals']
counts = data['counts']
    
#print('totals=',np.sum(totals,axis=1))
#print('counts=',np.sum(counts,axis=0))

sumPrecipitationByCity = np.sum(totals, axis=1)
sumCountByMonth = np.sum(counts, axis=0)
sumPrecipitationByMonth = np.sum(totals,axis=0)
sumCountByCity = np.sum(counts, axis=1)

# sums precipitation in a year for each city
# axis=1 sums rows
# returns min value in resulting array
minPrecipitation = np.argmin(sumPrecipitationByCity)
# sums all precipitation of cities
# each entry in resulting array is a different city
print('Row with lowest total precipitation:')
print(minPrecipitation)

averageAllByMonth = np.divide(sumPrecipitationByMonth, sumCountByMonth)
# average precipitation in all cities per month
# axis=0 sums columns
print('Average precipitation in each month:')
print(averageAllByMonth)


averageDailyPrecipitationCity = np.divide(sumPrecipitationByCity, sumCountByCity)
print('Average precipitation in each city:')
print(averageDailyPrecipitationCity)

n = 9
# split each row into a quarter, every four rows is a city
quarterlyArray = np.reshape(totals, (4*n, 3))
# sum each quarter
sumQuarters = np.sum(quarterlyArray, axis=1)
# take the 4 quarters and place into 4 columns per city
quarterCities = np.reshape(sumQuarters, (n,4))
#print('quarterly\n', quarterlyArray)
#print('sum quarter\n', sumQuarters)
print('Quarter precipitation totals:')
print(quarterCities)

