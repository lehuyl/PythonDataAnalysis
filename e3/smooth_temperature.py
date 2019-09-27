# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:45:32 2019

@author: Steven
"""
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pykalman import KalmanFilter

from statsmodels.nonparametric.smoothers_lowess import lowess
sysinfo = sys.argv[1]
cpu_data = pd.read_csv(sysinfo, parse_dates=['timestamp'])
#print(sysinfoData)

plt.figure(figsize=(12, 4))
plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5)

loess_smoothed = lowess(cpu_data['temperature'], cpu_data['timestamp'], frac=0.02)
plt.plot(cpu_data['timestamp'], loess_smoothed[:,1], 'r-')

kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1']]

initial_state = kalman_data.iloc[0]
observation_covariance = np.diag([10, 8, 8]) ** 2 # TODO: shouldn't be zero
transition_covariance = np.diag([2, 2, 2]) ** 2 # TODO: shouldn't be zero
transition = [[1, -1, 0.7], [0, 0.6, 0.03], [0, 1.3, 0.8]] # TODO: shouldn't (all) be zero

kf = KalmanFilter(
        initial_state_mean=initial_state,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition)

kalman_smoothed, _ = kf.smooth(kalman_data)
plt.plot(cpu_data['timestamp'], kalman_smoothed[:, 0], 'g-')
plt.legend(['Data Points', 'LOESS', 'Kalman'])
plt.show() # maybe easier for testing
#plt.savefig('cpu.svg') # for final submission