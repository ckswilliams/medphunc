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
        
    @staticmethod
    def calculate_transmission(a, b, y, thickness):
        return ((1 + b/a) * np.exp(a*y*thickness) - b/a)**(-1/y)
    
    @staticmethod
    def calculate_shielding(a, b, y, transmission):
        return 1 / (a*y) * np.log((transmission**(-y) + (b/a)) / (1+(b/a)))
    
    #Transmission calculation functions
    def shielding_to_transmission(self, thickness, material, kV):
        a, b, y = self.get_shielding_coefficients(material, kV)
        return self.calculate_transmission(a, b, y, thickness)

    def transmission_to_shielding(self, transmission, material, kV):
        a, b, y = self.get_shielding_coefficients(material, kV)
        return self.calculate_shielding(a, b, y, transmission)

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

class Kusano(Archer):
    "Superceded by NM, which includes other isotopes"
    att_coeff_fn = os.path.split(__file__)[0]+'/nm_input_shielding_coefficients.csv'
    
    
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

class NM(Archer):
    #att_coeff_fn = os.path.split(__file__)[0]+'/nm_input_shielding_coefficients.csv'
    att_coeff_fn = 'nm_input_shielding_coefficients.csv'
    
    def __init__(self):
        self.att_coeff = pd.read_csv(self.att_coeff_fn)
        self.att_coeff.value = pd.to_numeric(self.att_coeff.value, errors='coerce')
    
    
    def show_options(self):
        print('Radionuclides:')
        print(self.att_coeff.radionuclide.unique())
        print('Materials:')
        print(self.att_coeff.query("fit_method=='Archer'").material.unique())
        print('Only Archer fits have been implemented')
    
    def get_shielding_coefficients(self, radionuclide, material, fit_method='Archer'):
        
        df = self.att_coeff
        names = ['radionuclide', 'fit_method', 'material']
        values = [radionuclide, fit_method, material]
        
        for value, name in zip(values, names):
            df = df.loc[df[name]==value,:]
            if df.shape[0] == 0:
                raise(ValueError(f'Invalid combination of inputs: failed at finding {value} in {name}'))
        
        if all(df.value.isna()):
            raise(ValueError(f'Invalid combination of inputs: no published value for {radionuclide} using {fit_method} in {material}'))

        return {k:v for k, v in zip(df.parameter_name, df.value)}
    
    
    #Transmission calculation functions
    def shielding_to_transmission(self, radionuclide, material, thickness):
        coeff = self.get_shielding_coefficients(radionuclide, material, fit_method='Archer')
        a, b, y = coeff['alpha'], coeff['beta'], coeff['gamma']
        return self.calculate_shielding(a, b, y, thickness)

    def transmission_to_shielding(self, radionuclide, material, transmission):
        coeff = self.get_shielding_coefficients(radionuclide, material, fit_method='Archer')
        a, b, y = coeff['alpha'], coeff['beta'], coeff['gamma']
        return self.calculate_shielding(a, b, y, transmission)
    
#    def nth_vl_best_match(self, radionuclide, material):
#        isotope_coeffs = self.get_shielding_coefficients(radionuclide, 'Nth value layer', material)
#        over_values = self.att_coeff.query("fit_method=='Nth value layer'").query("material==@material")



class Groth(NM):
    pass
    
    
   

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
    
