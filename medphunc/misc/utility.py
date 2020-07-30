# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:52:03 2020

Variety of utility functions, mostly for dataframes

@author: Chris Williams
"""


import pandas as pd 
import numpy as np

#%%

def series_interp(lookup, value):
    """
    Interpolate value(s) from a pandas series with numerical index

    Parameters
    ----------
    lookup : pd.Series
        pandas series with numerical index and values.
    value : numeric or numeric list
        value or values to be interpolated.

    Returns
    -------
    pd.Series
        series with interpolated values corresponding to the input value(s)

    """
    lookup = lookup.copy()
    value = np.array(value).flatten()
    try:
        return lookup.loc[value]
    except KeyError:
        searchables = pd.DataFrame(index=np.array(value))
        searchables.loc[:,lookup.columns[0]] = np.nan
        lookup = pd.concat([lookup, searchables], axis=0)
        lookup = lookup.loc[~lookup.index.duplicated(keep='first')]
        lookup = lookup.interpolate('values') #interpolate using the values strategy
        return lookup.loc[value].iloc[:,0].copy()
    
    
    
def search_combine_results(col:pd.Series, search_terms:list) -> pd.DataFrame:
    """
    Search a pandas series for each regex string in search terms. Combine the results and return the first hit

    Parameters
    ----------
    col : pd.Series
        pandas series of strings.
    search_terms : list
        Iterable containing regex search terms with a single capture group.
        The highest priority terms should be first.

    Returns
    -------
    combined_results : pd.DataFame
        Dataframe with containing capture results, ignoring any results after
        the first successful one.
    """

    search_results = []
    
    for search_term in search_terms:
        test = col.str.extract(search_term)[0]
        search_results.append(test)
        print((~test.isna()).sum())
    
    
    results  = pd.concat(search_results, axis=1)
    
    combined_results = results.apply(lambda x: x[~x.isna()].head(1), axis=1)
    return combined_results