# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:45:16 2019

@author: Steven
"""

import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt

stations_file = sys.argv[1]
city_data = sys.argv[2]

cities = pd.read_csv(city_data)
stations = pd.read_json(stations_file, lines=True)

# convert tmax from degree C*10 to degree C
stations['avg_tmax'] = stations['avg_tmax'].apply(lambda x: x/10)
#convert area from m^2 to km^2
cities['area'] = cities['area'].apply(lambda x: x/1000000)

# filter out bad values
cities = cities[pd.notnull(cities)['population']]
cities = cities[pd.notnull(cities)['area']]
cities = cities[cities['area'].map(lambda x: x < 10000)]

#print(haversine(49.28, 123.00, 49.26, 123.1) + haversine(49.26,123.1,49.26,123.05))
# https://stackoverflow.com/questions/23142967/adding-a-column-thats-result-of-difference-in-consecutive-rows-in-pandas
def distance(lat1, lon1, lat2, lon2):
    R = 6371000 # Radius of the earth in m
    dLat = np.deg2rad(lat2-lat1);  # deg2rad below
    dLon = np.deg2rad(lon2-lon1); 
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)); 
    d = R * c; # Distance in m
    
    return d;
    
def best_tmax(city, stations):
    df = distance(city['latitude'], city['longitude'], stations['latitude'], stations['longitude'])
    
    return (stations.iloc[df.idxmin(), 0])
    
    
cities['tmax'] = cities.apply(best_tmax, axis=1, stations=stations)
cities['density'] = cities['population']/cities['area']
    
#
plt.title('Temperature vs. Population Density')
plt.xlabel('Avg Max Temperature (\u00b0c)')
plt.ylabel('Population Density (people/km\u00b2)')
plt.plot(cities['tmax'].values, cities['density'].values, 'b.')
#plt.show()
output = sys.argv[3]
plt.savefig(output)    
    
    
    
    
    
    
    
    
    
    
    