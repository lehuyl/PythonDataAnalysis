# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:01:15 2019

@author: Steven
"""

import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


data_labelled = pd.read_csv(sys.argv[1])
data_unlabelled = pd.read_csv(sys.argv[2])

X = data_labelled.loc[:,'tmax-01':'snwd-12'].values
y = data_labelled['city'].values

X_train, X_test, y_train, y_test = train_test_split(X,y)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel='linear', C=1)        
)

model.fit(X_train, y_train)

print('Score of SVC= ', model.score(X_test, y_test))

model.fit(X_train, y_train)

X_unlabelled = data_unlabelled = data_unlabelled.loc[:,'tmax-01':'snwd-12'].values
predictions = model.predict(X_unlabelled)

pd.Series(predictions).to_csv(sys.argv[3], index=False, header=False)

#df = pd.DataFrame({'truth': y_test, 'prediction': model.predict(X_test)})
#print(df[df['truth'] != df['prediction']])