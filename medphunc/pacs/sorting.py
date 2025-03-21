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

from typing import Type, Union, List



#%%

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

if len(logger.handlers) == 0:
    logger.addHandler(logging.StreamHandler())


#%%


def select_final_search_result(test_series:pd.Series, default: str) -> pd.Series:
    """
    Select a single item from a dataframe based on some simple criteria

    Parameters
    ----------
    test_series : pd.Series
        The series to be sorted.
    default : str in ['shortest','first','all']
        The method to sort. Shortest returns the shortest string in the series.
        first returns the top value.
        all retains everything (i.e. calling this function with 'all' does nothing)

    Returns
    -------
    pd.Series
        A series containing only the values which correspond to the 'default' argument.

    """

    if default=='shortest':
        #if multiple series left, pick the shortest description
        if test_series.shape[0] > 1:
            lengths = test_series.str.len()
            test_series = test_series.loc[lengths == lengths.min()]
            if test_series.shape[0] > 1:
                test_series = test_series.iloc[:1]
    if default=='longest':
        #if multiple series left, pick the shortest description
        if test_series.shape[0] > 1:
            lengths = test_series.str.len()
            test_series = test_series.loc[lengths == lengths.max()]
            if test_series.shape[0] > 1:
                test_series = test_series.iloc[:1]
    elif default=='first':
        test_series =  test_series.iloc[:1]
    elif default=='all':
        # return everything, so do nothing
        pass
    return test_series


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
    

    test_series = select_final_search_result(test_series,default)
    
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
    


def search(df: Type[pd.DataFrame],
            search_terms: Union[str, List[str], int, float, List[int], List[float], datetime.datetime],
            column: str,
            negation: bool,
            search_type: str,
            search_strictness: str) -> pd.DataFrame:
    """
    Function for filtering dataframes according a series of search terms or datetime info.

    Parameters
    ----------
    df : Type[pd.DataFrame]
        A dataframe containing, at least, the column specified in the column argument.
    search_terms : Union[str, List[str], datetime.datetime]
        a string, a list of strings, a numeric, a list of numerics, or a datetime object.
    column : str
        The column to search the dataframe on. For dicom-related searching,
        it's likely to be one of the standard pydicom tags like StudyDescription.
    negation : bool
        Whether to only return the inverse of the search terms, rather than 
        the term itself.
    search_type : str in ['str', 'greater']
        'str' string-type search, check whether the search term is anywhere in the values
        case insensitive.
        'greater' for quantitative comparisons. Use the negation argument for lesser.
    search_strictness : str in ['all','filter']
        'all' - the final results must meet all the provided search terms
        'filter' - any term which results in zero applicable results will be
        ignored


    Returns
    -------
    pd.DataFrame
      A dataframe containing only the rows which meet the search criteria
        

    """
    
    
    tdf = df.copy()

    if type(search_terms) is not list:
        search_terms = [search_terms]

    matches = []
    # Collect a series for each search term that shows whether the term was found for each data point
    for search_term in search_terms:
        if search_type == 'str':
            matching = tdf[column].astype(str).str.lower().str.contains(search_term.lower())
        elif search_type == 'greater':
            matching = tdf[column] > search_term
        else:
            raise (NotImplementedError('An unimplemented matching value was used'))
        if negation:
            matching = ~matching
        matches.append(matching)

    # Depending on the search strictness, return a result
    aggregate_matches = pd.DataFrame(matches).T
    if search_strictness == 'any':
        match_result = aggregate_matches.any(axis=1)
    elif search_strictness == 'all':
        # Strictly require that all search terms are met.
        # For instance, strictly require that modalities in study includes CT
        match_result = aggregate_matches.all(axis=1)
    elif search_strictness == 'filter':
        # Filter means apply each search term and exclude anything that doesn't meet it
        # UNLESS this would mean zero results remain.
        match_result = df[column] == df[column]  # Start with all true
        for match in matches:
            test_match = match_result & match  # Try to do an and between the previous answer and the new match
            if test_match.sum() >= 1:
                match_result = test_match  # Unless there is at least one remaining match, ignore the search term
    else:
        raise (NotImplementedError('An unimplemented search strictness value was used'))
    return tdf.loc[match_result, :].copy()


#%% some little convenience functions for extracting certain data types when using the searchset and thanks objects.


def get_scouts(thanks_object):
    if thanks_object.query_level == 'study':
        if thanks_object.result.shape[0] == 0:
            raise(ValueError('No results available for study level query.'))
        if thanks_object.result.shape[0] > 1:
            raise(ValueError('Too many results for study level query.'))
        return get_scouts(thanks_object.drill_down(thanks_object.result.index[0],find=True))
    if thanks_object.query_level in ['instance', 'image']:
        raise(ValueError('Cannot be used on a instance-level query'))
    r_series = thanks_object.result
    scout_series = r_series.loc[lambda x:(x.NumberOfSeriesRelatedInstances<3) & (x.SeriesNumber < 100)]
    
    ds_scout = []
    for scout_index in scout_series.index:
        ds = thanks_object.retrieve_or_move_and_retrieve(scout_index)
        for d in ds[0]:
            ds_scout.append(d)
    return ds_scout


def get_axial_index(thanks_series_object):
    if thanks_series_object.result.shape[0] == 0:
        raise(ValueError('No results available for this query. Search first.'))
    if thanks_series_object.query_level != 'series':
        raise(ValueError('Only series-level objects should be supplied'))

    r_series = thanks_series_object.result
    ax_series = search(r_series, ['saggittal','coronal'], column='SeriesDescription',search_type='str',negation=True,
                   search_strictness='all')
    ax_series = search(ax_series, ['axial','ax','vol'], column='SeriesDescription', search_type='str',negation=False,search_strictness='filter')
    ax_series = ax_series.loc[lambda x: x.NumberOfSeriesRelatedInstances > 20,:]
    ax_series = ax_series.loc[lambda x: x.NumberOfSeriesRelatedInstances == x.NumberOfSeriesRelatedInstances.min()]
    axial_index = ax_series.index[0]
    return axial_index
    


def get_first_last_axial_slices(thanks_series_object):
    axial_index = get_axial_index(thanks_series_object)
    
    t_instance = thanks_series_object.drill_down(axial_index)
    r_instance = t_instance.find()
    r_instance = r_instance.sort_values('InstanceNumber')
    min_instance_index = r_instance.iloc[0].name # Skip the very first one because of those annoying measurement slices.
    max_instance_index = r_instance.iloc[-1].name
    d_min = t_instance.retrieve_or_move_and_retrieve(min_instance_index)[0]
    d_max = t_instance.retrieve_or_move_and_retrieve(max_instance_index)[0]
    if d_min.ImageOrientationPatient != d_max.ImageOrientationPatient:
        # The first slice was one of those weird measurement images.
        min_instance_index = r_instance.iloc[1].name # Skip the very first one because of those annoying measurement slices.
        d_min = t_instance.retrieve_or_move_and_retrieve(min_instance_index)[0]
    return d_min, d_max
    
    
    






















