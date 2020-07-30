# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:13:54 2018
Calculate shielding coefficients using archer method

@author: WilliamCh
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

from medphunc.misc.utility import series_interp

weight_chart = os.path.split(__file__)[0]+'/weight_chart.csv'

weight_chart = pd.read_csv(weight_chart)



#%%

def series_interp(lookup, value):
    try:
        return lookup.loc[value]
    except KeyError:
        lookup.loc[value] = np.nan
        lookup = lookup.interpolate('values')
        return lookup.loc[value]
    

def weight_to_age(weight, gender=None):
    "Lookup average age by weight. Gender M or F"
    if gender:
        lookup = weight_chart.loc[weight_chart.gender == gender]
        lookup.index = lookup.weight
        lookup = lookup.age
    else:
        lookup = weight_chart.groupby('weight').age.mean()
    return series_interp(lookup, weight)
    

def age_to_weight(age, gender=None):
    "Lookup average weight by age. Gender M or F"
    if gender:
        lookup = weight_chart.loc[weight_chart.gender == gender]
        lookup.index = lookup.age
        lookup = lookup.weight
    else:
        lookup = weight_chart.groupby('age').weight.mean()
    return series_interp(lookup, age)

    
    
#%% Testing
if __name__=='__main__':
    #Initialise a class object
    123