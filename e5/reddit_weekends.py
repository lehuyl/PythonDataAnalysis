import sys
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import datetime


OUTPUT_TEMPLATE = (
    "Initial (invalid) T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann–Whitney U-test p-value: {utest_p:.3g}"
)

def isoweekday(date):
    return date.weekday();

def main():
    #print('something here',sys.argv[1])
    reddit_counts = sys.argv[1]
    
    counts = pd.read_json(reddit_counts, lines=True)
    #print(counts)
    
    # filter only 2012 and 2013 dates
    filtered = counts[(counts['date'].dt.year == 2012) | (counts['date'].dt.year == 2013)]
    filtered = filtered[filtered['subreddit'] == 'canada']
    #print(filtered)
    
    weekends = filtered[(filtered['date'].dt.weekday == 5) | (filtered['date'].dt.weekday == 6)]
    weekdays = filtered[(filtered['date'].dt.weekday != 5) & (filtered['date'].dt.weekday != 6)]
    #print(weekends)
    #print(weekdays)
    
    #print(stats.ttest_ind(weekends['comment_count'], weekdays['comment_count']))
    #print(stats.normaltest(weekends['comment_count']))
    #print(stats.normaltest(weekdays['comment_count']))
    #print(stats.levene(weekends['comment_count'], weekdays['comment_count']))
    
    # initial pvalues
    initial_ttest_p = stats.ttest_ind( weekends['comment_count'],weekdays['comment_count']).pvalue
    initial_weekend_normality_p = stats.normaltest(weekends['comment_count']).pvalue
    initial_weekday_normality_p = stats.normaltest(weekdays['comment_count']).pvalue
    initial_levene = stats.levene(weekends['comment_count'], weekdays['comment_count']).pvalue
    
    # transform 2
    
    #plt.hist(np.histogram(np.log(weekends['comment_count'])))
    #plt.hist(np.histogram(np.log(weekdays['comment_count'])))
    #plt.hist(np.histogram(np.sqrt(weekends['comment_count'])))
    #plt.hist(np.histogram(np.sqrt(weekdays['comment_count'])))
    
    # sqrt is produces most normal graph after transform but still not normal
    weekend_sqrt = np.sqrt(weekends['comment_count'])
    weekday_sqrt = np.sqrt(weekdays['comment_count'])
    weekend_sqrt_p = stats.normaltest(weekend_sqrt).pvalue
    weekday_sqrt_p = stats.normaltest(weekday_sqrt).pvalue   
    weekend_levene_p = stats.levene(weekend_sqrt, weekday_sqrt).pvalue                     
    
    #plt.hist(np.histogram(np.exp(weekends['comment_count'])))
    #plt.hist(np.histogram(np.exp(weekdays['comment_count'])))
    #plt.hist(np.histogram(weekends['comment_count']**2))
    #plt.hist(np.histogram(weekdays['comment_count']**2))
    
    
    # transform 2
    weekends_iso = weekends['date'].apply(datetime.date.isocalendar).apply(pd.Series)
    weekends_iso.columns = ['year','week','weekday']
    weekends['year'] = weekends_iso['year']
    weekends['week'] = weekends_iso['week']
    
    weekdays_iso = weekdays['date'].apply(datetime.date.isocalendar).apply(pd.Series)
    weekdays_iso.columns = ['year','week','weekday']
    weekdays['year'] = weekdays_iso['year']
    weekdays['week'] = weekdays_iso['week']   
    
    
    weekends_grouped = weekends.groupby(['year','week']).aggregate('mean').reset_index()
    weekdays_grouped = weekdays.groupby(['year','week']).aggregate('mean').reset_index()
    
    
    weekend_grouped_p = stats.normaltest(weekends_grouped['comment_count']).pvalue
    weekdays_grouped_p = stats.normaltest(weekdays_grouped['comment_count']).pvalue
    weekend_grouped_levene_p = stats.levene(weekends_grouped['comment_count'], weekdays_grouped['comment_count']).pvalue 
    weekly_ttest = stats.ttest_ind( weekends_grouped['comment_count'],weekdays_grouped['comment_count']).pvalue
    
    
    utest_p = stats.mannwhitneyu(weekends['comment_count'], weekdays['comment_count']).pvalue
    
    # ...
    
    

    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=initial_ttest_p,
        initial_weekday_normality_p=initial_weekday_normality_p,
        initial_weekend_normality_p=initial_weekend_normality_p,
        initial_levene_p=initial_levene,
        transformed_weekday_normality_p=weekday_sqrt_p,
        transformed_weekend_normality_p=weekend_sqrt_p,
        transformed_levene_p=weekend_levene_p,
        weekly_weekday_normality_p=weekdays_grouped_p,
        weekly_weekend_normality_p=weekend_grouped_p,
        weekly_levene_p=weekend_grouped_levene_p,
        weekly_ttest_p=weekly_ttest,
        utest_p=utest_p,
    ))


if __name__ == '__main__':
    main()