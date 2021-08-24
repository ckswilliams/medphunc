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


att_coeff_fn = os.path.split(__file__)[0]+'/input_shielding_coefficients.csv'

class Archer:
    def __init__(self):
        self.df_att_coeff = pd.read_csv(att_coeff_fn)
        
    #Transmission calculation functions
    def shielding_to_transmission(self, thickness, material, kV):
        a, b, y = self.get_shielding_coefficients(material, kV)
        return ((1 + b/a) * np.exp(a*y*thickness) - b/a)**(-1/y)

    def transmission_to_shielding(self, transmission, material, kV):
        a, b, y = self.get_shielding_coefficients(material, kV)
        x = 1 / (a*y) * np.log((transmission**(-y) + (b/a)) / (1+(b/a)))
        return x

    def get_shielding_coefficients(self, material, kV):
        view = self.df_att_coeff.loc[self.df_att_coeff.Material == material,
                                     ['kV', 'a', 'b', 'y']]
        #view.index = view.kV
        xp = np.array(view['kV'])
        yp = np.array(view[['a', 'b', 'y']])
        a = np.interp(kV, xp, yp[:, 0])
        b = np.interp(kV, xp, yp[:, 1])
        y = np.interp(kV, xp, yp[:, 2])
        return a, b, y


#%%

class Kusano:
    def __init__(self):
        self.df_att_coeff = pd.DataFrame([['I-131','Lead',0.1078,0.2003,0.4957],
                                            ['I-131','Concrete', 0.1705, -0.1308, 1.038]],
                                         columns=['Isotope','Material','a','b','y'])
        
    #Transmission calculation functions
    def shielding_to_transmission(self, thickness, material, isotope):
        a, b, y = self.get_shielding_coefficients(material, isotope)
        return ((1 + b/a) * np.exp(a*y*thickness) - b/a)**(-1/y)

    def transmission_to_shielding(self, transmission, material, isotope):
        a, b, y = self.get_shielding_coefficients(material, isotope)
        x = 1 / (a*y) * np.log((transmission**(-y) + (b/a)) / (1+(b/a)))
        return x

    def get_shielding_coefficients(self, material, isotope):
        view = self.df_att_coeff.query("Isotope==@isotope").query('Material==@material')
        return view.loc[:,['a','b','y']].iloc[0,:]
    
k = Kusano()
k.shielding_to_transmission(15.4, 'Lead','I-131')

#%%    
    
        #view.index = view.kV
        xp = np.array(view['Isotope'])
        yp = np.array(view[['a', 'b', 'y']])
        a = np.interp(isotope, xp, yp[:, 0])
        b = np.interp(isotope, xp, yp[:, 1])
        y = np.interp(isotope, xp, yp[:, 2])
        return a, b, y


#%% Testing
if __name__=='__main__':
    #Initialise a class object
    a=Archer()
    #Create an array of lenghts
    xs = np.linspace(0,2,100)
    #Create an array of kVs
    kvs = np.linspace(50,140,5)
    #Use the Archer class to caluclate shielding transmissions
    t = a.shielding_to_transmission(xs,'Lead',80)
    #Plot the results
    plt.plot(xs,t)
    
