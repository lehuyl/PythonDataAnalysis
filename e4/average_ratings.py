# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:02:13 2019

@author: Steven
"""
import pandas as pd
import sys 
import difflib as diff

movie_input = sys.argv[1]
rating_input = sys.argv[2]
colNames = ['title']
names = pd.read_csv(movie_input, '\n', names=colNames);


movieScores = pd.read_csv(rating_input, ',');

def correctTitles(x, names):
    result = diff.get_close_matches(x, names)
    if(len(result) == 0):
        return 
    else:
        return result[0]
    
    

movieScores['title'] = movieScores['title'].apply(correctTitles, names=names['title'])
movieScores = movieScores[pd.notnull(movieScores['title'])]



avg = pd.DataFrame(movieScores.groupby(movieScores['title'])['rating'].mean().round(2), columns=['rating'])

output = sys.argv[3]
avg.to_csv(output)