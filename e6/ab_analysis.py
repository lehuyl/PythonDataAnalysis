# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:29:56 2019

@author: Steven
"""

import sys
import scipy.stats as sp
import pandas as pd


OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value: {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value: {more_searches_p:.3g}\n'
    '"Did more/less instructors use the search feature?" p-value: {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value: {more_instr_searches_p:.3g}'
)


def main():
    searchdata_file = sys.argv[1]
   
    searches = pd.read_json(searchdata_file, orient='records', lines=True)
    
    odd = searches[searches['uid']%2 == 1]
    even = searches[searches['uid']%2 == 0]
    
    
    never_searched = searches[searches['search_count'] == 0]
    searched = searches[searches['search_count'] > 0]
    
    total_even_notsearched = len(never_searched[never_searched['uid']%2 == 0])
    total_even_searched = len(searched[searched['uid']%2 == 0])
    total_odd_notsearched = len(never_searched[never_searched['uid']%2 == 1])
    total_odd_searched = len(searched[searched['uid']%2 == 1])
  
    
    chi_result_p = sp.chi2_contingency([[total_even_searched, total_even_notsearched], [total_odd_searched, total_odd_notsearched]])[1]
    mann_p = sp.mannwhitneyu(odd['search_count'], even['search_count'])[1]
    
    instr_odd_list = odd[odd['is_instructor'] == True]
    instr_even_list = even[even['is_instructor'] == True]
    
    instr_never_searched = never_searched[never_searched['is_instructor'] == True]
    instr_searched = searched[searched['is_instructor'] == True]
    
    instr_total_even_notsearched = len(instr_never_searched[instr_never_searched['uid']%2 == 0])
    instr_total_even_searched = len(instr_searched[instr_searched['uid']%2 == 0])
    instr_total_odd_notsearched = len(instr_never_searched[instr_never_searched['uid']%2 == 1])
    instr_total_odd_searched = len(instr_searched[instr_searched['uid']%2 == 1])
    
    chi_result2_p = sp.chi2_contingency([[instr_total_even_searched, instr_total_even_notsearched], [instr_total_odd_searched, instr_total_odd_notsearched]])[1]
    mann2_p = sp.mannwhitneyu(instr_odd_list['search_count'], instr_even_list['search_count'])[1]
    

    # Output
    print(OUTPUT_TEMPLATE.format(
        more_users_p=chi_result_p,
        more_searches_p=mann_p,
        more_instr_p=chi_result2_p,
        more_instr_searches_p=mann2_p,
    ))


if __name__ == '__main__':
    main()