# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:55:20 2020
sorting
Utility scripts for PACS queries and query results

@author: WILLCX
"""



from pydicom import Dataset
import pydicom
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian, ExplicitVRBigEndian
from pynetdicom import AE
from pynetdicom import QueryRetrievePresentationContexts
from pynetdicom import VerificationPresentationContexts
from pynetdicom import StoragePresentationContexts
from time import sleep
import datetime
import pandas as pd
import logging
import json
import pathlib
import copy
#from pynetdicom import debug_logger
#debug_logger()

#%%

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

if len(logger.handlers) == 0:
    logger.addHandler(logging.StreamHandler())


#%%
def find_best_series_match(test_series,
                           filter_list,
                           default='shortest'):
    """
    Search in the pandas series provided for the values in filter list.
    Don't apply any filters that have no results, prefer the shortest result

    Parameters
    ----------
    test_series : pd.Series
        series containing strings.
    filter_list : list of strings
        List of strings to be used as search terms. Regex is allowed I think.
    default : string - shortest, first, or all.
        Determines which result to keep if multiple persist past filtering.

    Returns
    -------
    index
        pandas index for the result.
    test_series : pandas.Series
        Value fulfilling the search results.

    """
    #try filtering for the following terms. if a term filters to 0, don't filter
    test_series = test_series.copy()
    for filter_term in filter_list:
        # if one series left, finish
        if test_series.shape[0] <= 1:
            break
        new_series = test_series.loc[test_series.str.contains(filter_term, case=False)]
        if new_series.shape[0] > 0:
            test_series = new_series
    
    if default=='shortest':
        #if multiple series left, pick the shortest description
        if test_series.shape[0] > 1:
            lengths = test_series.str.len()
            test_series = test_series.loc[lengths == lengths.max()]
            if test_series.shape[0] > 1:
                test_series = test_series.iloc[:1]
    elif default=='first':
        test_series = test_series.iloc[:1]
    elif default=='all':
        # return everything
        pass
        
    
    return test_series.index, test_series



def test_study_date_within_window(test_series, reference_date, date_window=1):
    ref_date = pd.to_datetime(reference_date)
    test_series = pd.to_datetime(test_series)
    day_delta = test_series-ref_date
    day_delta = day_delta.dt.total_seconds().abs()/60/60/24
    return day_delta < date_window
    
    

def best_result_match(pacs_find_result, reference_date=None, date_window=1, study_description_search_terms=None):

    if reference_date is not None:
        studies_in_window = test_study_date_within_window(pacs_find_result.StudyDate,
                                                          reference_date, date_window)
        pacs_find_result = pacs_find_result.loc[studies_in_window,:]
        
    if study_description_search_terms is not None:
        search_match = find_best_series_match(
            pacs_find_result.StudyDescription, study_description_search_terms)
        pacs_find_result = pacs_find_result.loc[search_match[0]]
    
    return pacs_find_result
    
    