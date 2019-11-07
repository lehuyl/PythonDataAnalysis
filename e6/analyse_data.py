# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 22:28:58 2019

@author: Steven
"""

import sys
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

input = sys.argv[1]

data = pd.read_csv(input)


result_anova_p = stats.f_oneway(data['qs1'], data['qs2'], data['qs2'], data['qs4'], data['qs5'], data['merge1'], data['partition_sort']).pvalue
# p value = 0 < 0.05

# anova is significant proceed to post-hoc analysis

#adapted from statistics test slides
melt = pd.melt(data)

posthoc = pairwise_tukeyhsd(
    melt['value'], melt['variable'],
    alpha=0.05)

print(posthoc)

fig = posthoc.plot_simultaneous()