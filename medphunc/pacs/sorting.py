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
                           filter_list
                           ):
    """
    Search in the pandas series provided for the values in filter list.
    Don't apply any filters that have no results, prefer the shortest result

    Parameters
    ----------
    test_series : pd.Series
        series containing strings.
    filter_list : list of strings
        List of strings to be used as search terms. Regex is allowed I think.

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
            
    #if multiple series left, pick the shortest description
    if test_series.shape[0] > 1:
        lengths = test_series.str.len()
        test_series = test_series.loc[lengths == lengths.max()]
    
    return test_series.index[0], test_series



