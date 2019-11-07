# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 21:42:57 2019

@author: Steven
"""

import time
import pandas as pd
import numpy as np
from implementations import all_implementations
# ...

data = pd.DataFrame(columns=['qs1', 'qs2', 'qs3', 'qs4', 'qs5', 'merge1', 'partition_sort'], index=np.arange(250))


for i in range(250):
    random_array = np.random.randint(0,1000,1000)
    
    for sort in all_implementations:
        st = time.time()
        res = sort(random_array)
        en = time.time()
        actual = en - st
        data.iloc[i][sort.__name__] = actual
        
        
data.to_csv('data.csv', index=False)