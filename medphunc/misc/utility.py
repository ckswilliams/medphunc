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



def convert_unit(unit):
    unit_ref={'':1,
       'm':0.001,
       'c':0.01}
    return unit.map(unit_ref)

def split_units(unit_strings, seps):
    """
    Split a string into units, retrieving the magnitude prefix from each component

    Parameters
    ----------
    units : TYPE
        DESCRIPTION.
    *seps : list of lists
        Each sep should be 3 parts:
            
            search from left to right.

    Returns
    -------
    None.

    """
    output = []
    leftover = pd.Series(unit_strings)
    for sep in seps:
        # Workaround because rsplit wasn't matching my regex properly
        # problem regex: Gy\.{0,1}
        split = leftover.str.split(sep)
        output.append(split.str[:-1].str.join(sep=sep))
        leftover = split.str[-1]
    return output

    

def dap_unit_conversion(dap_unit, preferred_unit='Gym2'):
    input_units = split_units(dap_unit, ['Gy\.{0,1}','m'])
    input_magnitudes = [convert_unit(u) for u in input_units]
    input_mult = input_magnitudes[0]*input_magnitudes[1]**2
    
    pref_units = split_units(preferred_unit, ['Gy\.{0,1}','m'])
    pref_magnitudes = [convert_unit(u) for u in pref_units]
    pref_mult = (pref_magnitudes[0]*pref_magnitudes[1]**2)[0]
    

    return input_mult/pref_mult

def dose_unit_conversion(dose_unit, preferred_unit='mGy'):
    input_units = split_units(dose_unit, ['Gy'])
    input_magnitudes = [convert_unit(u) for u in input_units]
    input_mult = input_magnitudes[0]
    
    pref_units = split_units(preferred_unit, ['Gy'])
    pref_magnitudes = [convert_unit(u) for u in pref_units]
    pref_mult = pref_magnitudes[0][0]
    return input_mult/pref_mult