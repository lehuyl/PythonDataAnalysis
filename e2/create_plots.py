# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:06:11 2019

@author: Steven
"""

import sys
import matplotlib.pyplot as plt
import pandas as pd

firstHour = sys.argv[1]
secondHour = sys.argv[2]

firstHourData = pd.read_csv(firstHour, sep=' ', header=None, index_col=1,
        names=['lang', 'page', 'views', 'bytes'])

secondHourData = pd.read_csv(secondHour, sep=' ', header=None, index_col=1,
        names=['lang', 'page', 'views', 'bytes'])

firstSeries = pd.Series(firstHourData['views'].values, index=firstHourData.index)
secondSeries = pd.Series(secondHourData['views'].values, index=secondHourData.index)

df = firstHourData.merge(secondHourData['views'], how='inner', on='page')

#plot 1
firstSort = firstHourData.sort_values(by='views',ascending=False)

plt.figure(figsize=(10, 5)) # change the size to something sensible
plt.subplot(1, 2, 1) # subplots in 1 row, 2 columns, select the first
plt.title('Popularity Distribution')
plt.ylabel('Views')
plt.xlabel('Rank')
plt.plot(firstSort['views'].values) # build plot 1

plt.subplot(1, 2, 2) # ... and then select the second
plt.title('Daily Correlation')
plt.ylabel('Day 1 Views')
plt.xlabel('Day 2 Views')
plt.xscale('log')
plt.yscale('log')
plt.plot(df['views_x'].values, df['views_y'].values, 'b.') # build plot 2
#plt.show()
plt.savefig('wikipedia.png')