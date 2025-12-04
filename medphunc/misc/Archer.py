# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:13:54 2018
Calculate shielding coefficients using archer method


Shielding coefficients are based on
BJR Rdaiation Shielding for Diagnostic Radiology 2nd edition, table 4.1
kV of 25-49 is from Transmission of broad W/Rh and W/Al (target/filter) x‐ray beams operated at 25–49 kVp through common shielding materials - Li - 2012 - Medical Physics - Wiley Online Library). 
Kusano
and other? #todo write reference for other nuc med isotopes here.


@author: WilliamCh
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

att_coeff_fn = os.path.split(__file__)[0] + '/input_shielding_coefficients.csv'



# Copy/pasted here from pytch. consider refactor so that there's one source for this function
def select_item_enumerated(items, item_name, multiple=False, return_key=False):
    if type(items) is dict:
        item_list = items.keys()
        is_dict = True
    elif type(items) is list:
        item_list = items
        is_dict = False
    else:
        raise(ValueError('bad input type'))
    
    s = f'Select {item_name} from the list'
    if multiple:
        s = s+'\n separate multiple selections with comma.'
    else:
        s = s+'\n please input a single integer to choose'
    s = s+'\n'+'\n'.join([f'{i} - {name}' for i, name in enumerate(item_list)])
    s = s+' -> '
    sel = input(s)
    sel = sel.split(',')
    sel = [int(s) for s in sel]
    if is_dict:
        out = [list(items.keys())[s] for s in sel]
    else:
        out = sel
    if not return_key:
        out = [items[s] for s in sel]
    if not multiple:
        out = out[0]
    
    return out


#%%

def calculate_transmission(a, b , y, thickness):
    return ((1 + b / a) * np.exp(a * y * thickness) - b / a) ** (-1 / y)


def calculate_shielding(a, b, y, transmission):
    return 1 / (a * y) * np.log((transmission ** (-y) + (b / a)) / (1 + (b / a)))



class Archer:
    df_att_coeff : pd.DataFrame = None
    
    shielding_source_options = {
        'bjr + Li (W/Rh)':'input_shielding_coefficients.csv',
        'BJR + Li (W/Al)':'shielding_coefficients_bjr_li(MG  W Al).csv',
        'Simpkin':'input_shielding_coefficients_simpkin.csv'}
    
    
    def __init__(self, source = 'bjr + Li (W/Rh)'):
        self._load_shielding_coefficients(source)
        
        
    def _load_shielding_coefficients(self, source=None):
        if source is None:
            source = select_item_enumerated(self.shielding_source_options,
                                            'Shielding coefficient source',
                                            return_key=True)
        att_coeff_fn = os.path.split(__file__)[0] + '/' + self.shielding_source_options[source]
        self._load_shielding_fn(att_coeff_fn)
    
    
    def _load_shielding_fn(self, fn):
        self.df_att_coeff = pd.read_csv(fn)
        
    

    # Transmission calculation functions
    def shielding_to_transmission(self, thickness, material, kV):
        a, b, y = self.get_shielding_coefficients(material, kV)
        return calculate_transmission(a, b, y, thickness)

    def transmission_to_shielding_table(self, transmission:pd.Series, kV:int):
        material_results = [transmission]
        for material in self.df_att_coeff.Material.unique():
            result =  self.transmission_to_shielding(transmission, material, kV)
            result = result.iloc[:,0]
            result.name=material
            material_results.append(result)
        return pd.concat(material_results, axis=1)
        
        
        
    def transmission_to_shielding(self, transmission, material, kV):
        a, b, y = self.get_shielding_coefficients(material, kV)
        return calculate_shielding(a, b, y, transmission)
    


    def get_shielding_coefficients(self, material, kV):
        view = self.df_att_coeff.loc[self.df_att_coeff.Material == material,
                                     ['kV', 'a', 'b', 'y']]
        # view.index = view.kV
        xp = np.array(view['kV'])
        yp = np.array(view[['a', 'b', 'y']])
        a = np.interp(kV, xp, yp[:, 0])
        b = np.interp(kV, xp, yp[:, 1])
        y = np.interp(kV, xp, yp[:, 2])
        return a, b, y
    
    
    def transmission_clipboard_to_shielding_clipboard(self, kV):
        t = pd.read_clipboard()
        return self.transmission_to_shielding_table(t, kV).iloc[:,1:]


# %%

class Kusano(Archer):
    """
    Superceded by NM, which includes other isotopes
    """

    att_coeff_fn = os.path.split(__file__)[0] + '/nm_input_shielding_coefficients.csv'

    def __init__(self):
        super().__init__()
        self.df_att_coeff = pd.DataFrame([['I-131', 'Lead', 0.1078, 0.2003, 0.4957],
                                          ['I-131', 'Concrete', 0.1705, -0.1308, 1.038]],
                                         columns=['Isotope', 'Material', 'a', 'b', 'y'])

    # Transmission calculation functions
    def shielding_to_transmission(self, thickness, material, isotope):
        a, b, y = self.get_shielding_coefficients(material, isotope)
        return calculate_transmission(a, b, y, thickness)

    def transmission_to_shielding(self, transmission, material, isotope):
        a, b, y = self.get_shielding_coefficients(material, isotope)
        return calculate_shielding(a, b, y, transmission)

    def get_shielding_coefficients(self, material, isotope):
        view = self.df_att_coeff.query("Isotope==@isotope").query('Material==@material')
        return view.loc[:, ['a', 'b', 'y']].iloc[0, :]


# %%

class NM(Archer):
    att_coeff_fn = os.path.split(__file__)[0] + '/nm_input_shielding_coefficients.csv'

    def __init__(self):
        super().__init__()
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
            df = df.loc[df[name] == value, :]
            if df.shape[0] == 0:
                raise (ValueError(f'Invalid combination of inputs: failed at finding {value} in {name}'))

        if all(df.value.isna()):
            raise (ValueError(
                f'Invalid combination of inputs:' +
                ' no published value for {radionuclide} using {fit_method} in {material}'))

        return {k: v for k, v in zip(df.parameter_name, df.value)}

    # Transmission calculation functions
    def shielding_to_transmission(self, radionuclide, material, thickness):
        coeff = self.get_shielding_coefficients(radionuclide, material, fit_method='Archer')
        a, b, y = coeff['alpha'], coeff['beta'], coeff['gamma']
        return calculate_transmission(a, b, y, thickness)

    def transmission_to_shielding(self, radionuclide, material, transmission):
        coeff = self.get_shielding_coefficients(radionuclide, material, fit_method='Archer')
        a, b, y = coeff['alpha'], coeff['beta'], coeff['gamma']
        return calculate_shielding(a, b, y, transmission)


#    def nth_vl_best_match(self, radionuclide, material):
#        isotope_coeffs = self.get_shielding_coefficients(radionuclide, 'Nth value layer', material)
#        over_values = self.att_coeff.query("fit_method=='Nth value layer'").query("material==@material")


class Groth(NM):
    pass


# %% Testing
if __name__ == '__main__':
    # Initialise a class object
    archer = Archer()
    # Create an array of lenghts
    xs = np.linspace(0, 2, 100)
    # Create an array of kVs
    kvs = np.linspace(50, 140, 5)
    # Use the Archer class to caluclate shielding transmissions
    t = archer.shielding_to_transmission(xs, 'Lead', 80)
    # Plot the results
    plt.plot(xs, t)
