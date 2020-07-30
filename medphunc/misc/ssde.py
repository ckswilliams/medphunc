# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:37:02 2020

@author: willcx
"""
import pandas as pd
import numpy as np
import os
from medphunc.misc.utility import series_interp


ssde_chart_fn = os.path.split(__file__)[0]+'/ssde.csv'

ssde_chart = pd.read_csv(ssde_chart_fn)


#%%

def wed_from_measurement(x, region='body', kind='latap'):
    region_lookup = ssde_chart.loc[:, ssde_chart.columns.str.split('_').str[0] == region]
    eff_lookup = region_lookup.loc[region_lookup.iloc[:,0].notna(),
                                   region_lookup.columns.str.split('_').str[1]==kind]
    eff_lookup = eff_lookup.set_index(eff_lookup.columns[0])
    eff_values = series_interp(eff_lookup, x)
    return eff_values

def ssde_from_measurement(x, region='body', kind='latap'):
    """
    Look up SSDE based on the body region, measured length, and type of measurement.
    See AAPM Report 204 for source data.

    Parameters
    ----------
    region : str, optional
        Body region for the dose. The default is 'body', other options are 'head'.
    x : numeric
        Measured distance in cm, or array of distances.
    kind : str, optional
        What measurement has been supplied. The default is 'latap' (lat + ap). Other options are 'lat' and 'ap'

    Returns
    -------
    SSDE conversion factor as series, if vector input, or float if float input.

    """
    
    eff_values = wed_from_measurement(x, region, kind)
    
    output = ssde_from_wed(eff_values, region)
    output.index = np.array(x).flatten()
    if output.shape[0] == 1:
        return output.iloc[0]
    return output
    

def ssde_from_wed(x, region='body'):
    """
    Look up SSDE based on the body region and effective diameter, also known 
    as the water equivalent diameter.
    See AAPM Report 204 / 220 for source data and calculation methods for 
    effective diameter.

    Parameters
    ----------
    x : numeric
        Measured effective diameter/WED in cm, or array of values.
    region : str, optional
        Body region for the dose. The default is 'body', other options are 'head'.

    Returns
    -------
    SSDE conversion factor as series, if vector input, or float if float input.

    """
    region_lookup = ssde_chart.loc[:, ssde_chart.columns.str.split('_').str[0] == region]
    conv_lookup = region_lookup.loc[region_lookup.iloc[:,0].notna(),
                                   region_lookup.columns.str.split('_').str[1] == 'eff']
    conv_lookup = conv_lookup.set_index(conv_lookup.columns[0])
    output = series_interp(conv_lookup, x)
    output.index = np.array(x).flatten()
    
    return output
    