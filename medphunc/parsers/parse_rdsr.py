# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:41:53 2019

Crunch an RDSR

@author: willcx
"""

#%%

import math
import logging


import rdsr_navigator as nav
import pandas as pd


from medphunc.parsers import generic_dict_parser as dp
from medphunc.parsers.parse_dicom import parse_single_dcm
from medphunc.misc import utility


#%%
logger = logging.getLogger(__name__)
#logging.basicConfig(filename='rdsr.log', filemode='w',
# format='%(name)s - %(levelname)s - %(message)s')


#%%
def rdsr_line_to_dic(rdsr):
    output = {}

    if len(rdsr.content) > 0:
        for rdsr_item in rdsr.content:
            try:
                output[rdsr_item.concept_name.code_meaning] = rdsr_line_to_dic(rdsr_item)
            except AssertionError as e:
                print(e)
                continue
            except AttributeError as e:
                print(e)
                continue
    else:
        try:
            output = rdsr.value
        except TypeError:
            output = None

    return output

def rdsr_to_dic(rdsr):
    output = []
    for rdsr_ct_items in rdsr.get_all('ct_acquisition'):
        output.append(rdsr_line_to_dic(rdsr_ct_items))
    for rdsr_xray_items in rdsr.get_all('irradiation_event_x-ray_data'):
        output.append(rdsr_line_to_dic(rdsr_xray_items))
    return output


def split_unit_cols(df):
    for col in df.columns:
        test = df[col].apply(type)
        if test.isin([tuple,list]).sum() > 0:
            df[col+'_unit'] = df[col].str[-1]
            df[col] = df[col].str[0]

    return df


def rdsr_items_to_df(rdsr_items):
    df = pd.DataFrame(rdsr_items)
    df = dp.find_explode_dict_to_columns(df)
    df = split_unit_cols(df)
    df = df[df.columns.sort_values()]
    return df

def rdsr_dose_data_to_df(rdsr):
    rdsr_items = rdsr_to_dic(rdsr)
    df = rdsr_items_to_df(rdsr_items)
    return df

#%%
def rdsr_meta_data_to_df(rdsr):
    output = {}
    i = 0
    bonus = ['B', 'C', 'D']
    for r in rdsr.content:
        if r.concept_name.code_meaning  in ['irradiation_event_x-ray_data', 'ct_acquisition']:
            continue
        #Slightly dodgy try-except to build output. Could be refactored to an if
        try:
            # check whether the code meaning is already in the output dic
            output[r.concept_name.code_meaning] = output[r.concept_name.code_meaning]
            # If it is, assume this is for a new plane/tube/whatever.
            # Add a second version with a letter designation.
            output[r.concept_name.code_meaning + '_'+bonus[i]] = rdsr_line_to_dic(r)
            i = i + 1
        except IndexError:
            # If it isn't in the dictionary, just create it as normal
            output[r.concept_name.code_meaning] = rdsr_line_to_dic(r)
        except KeyError:
            output[r.concept_name.code_meaning] = rdsr_line_to_dic(r)
    #Need to turn into a list to match the expected input of function, which is a list of dicts
    df = rdsr_items_to_df([output])
    return df

#%%


# Sanity check to see whether the full RDSR is present

def check_rdsr_completeness(dose_data, rdsr_metadata):
    check_cols = ['dose_(rp)', 'dose_area_product']
    for col in check_cols:
        if col in dose_data:
            meta_cols = rdsr_metadata.loc[rdsr_metadata.index.str.contains('.'+col, regex=False),
                                           'value']
            meta_total = meta_cols[~meta_cols.index.str.contains('_unit')].sum()
            dose_data_total = dose_data[col].sum()
            if not math.isclose(meta_total, dose_data_total,rel_tol=3e-2):
                raise ValueError(f'The individual values in dose data column \
                                 "{col}" do not add up to the RDSR metadata column, \
                                     indicating that the file input is corrupted')
    return True




#%%



    
    

def export_for_skin_dose_spreadsheet(dose_data):
    df = dose_data.copy()
    # Adjust unit magnititudes

    # Want DAP in Gycm^2, assuming Gym^2
    dap_conversion_factors = utility.dap_unit_conversion(df['dose_area_product_unit'], 'Gycm2')
    df.dose_area_product = df.dose_area_product*dap_conversion_factors
    # df.dose_area_product = 
    # if (df['dose_area_product_unit'] == 'Gy.m2').all():
    #     df.dose_area_product = df.dose_area_product*10000
    # else:
    #     raise ValueError('DAP in unexpected unit')

    # Want dose in mGy, assuming Gy
    dose_conversion_factors = utility.dose_unit_conversion(df['dose_(rp)_unit'], 'mGy')
    df['dose_(rp)'] = df['dose_(rp)'] * dose_conversion_factors

    # Find instances of field area which are 0, and try setting them from dose data
    
    df.distance_source_to_detector.loc[df.distance_source_to_detector==0] = df.distance_source_to_detector.max()
    m = df.collimated_field_area == 0
    if m.sum() > 0:
        try:
            # Wherever there's no field area, estimate it based off the DAP, dose, SID (and correct the oom)
            df.loc[m, 'collimated_field_area'] = utility.field_area_from_dose_data(df.loc[m,:]) / 10
        except AttributeError:
            logger.info('Tried to estimate collimated field size')

    # Want field area in cm^2, assume ?? m^2
    df.collimated_field_area *= 10000
    
    # If the material is not copper, set the thickness to 0.
    # %todo make the new thickness depend on the material rather than copper or nothing
    df.loc[df['x-ray_filters.x-ray_filter_material'] != 'Copper or Copper compound',
           'x-ray_filters.x-ray_filter_thickness_maximum'] = 0

    export_cols = ['irradiation_event_type',
    'kvp',
    'fluoro_mode',
    'exposure_time',
    'dose_area_product',
    'dose_(rp)',
    'positioner_primary_angle',
    'positioner_secondary_angle',
    'collimated_field_area',
    'distance_source_to_detector',
    'x-ray_tube_current',
    'acquisition_protocol',
    'x-ray_filters.x-ray_filter_thickness_maximum',
    'acquisition_plane',
    'table_height_position',
    'table_lateral_position',
    'table_longitudinal_position']
    df = df.reindex(columns=export_cols)
    
    return df

def dump_rdsr(fn, output_fn = 'rdsr_dump.xlsx'):
    rdsr = nav.read_file(fn)
    output = {}
    writer = pd.ExcelWriter(output_fn, engine='xlsxwriter')
    rdsr_metadata = rdsr_meta_data_to_df(rdsr).T
    rdsr_metadata.columns = ['value']
    dicom_metadata = parse_single_dcm(fn)
    dose_data = rdsr_dose_data_to_df(rdsr)

    # Is this fluoro? if so try to make an abbreviated sheet that can be used to calculate PSD
    if dose_data.columns.str.contains('dose_(rp)', regex=False).any():
        try:
            df = export_for_skin_dose_spreadsheet(dose_data)
            df.to_excel(writer, 'psd_export', index=False)
            output['skin_dose_data'] = df

        except Exception as e:
            print(e)

    try:
        check_rdsr_completeness(dose_data, rdsr_metadata)
    except ValueError as e:
        #raise(e)
        logger.warning('Inconsistency encountered in RDSR contents: %s', e)
        dose_data.index+=1
        dose_data =  dose_data.sort_index()
        dose_data.loc[0,:] = f'WARNING: error during processing  - {e}'

    rdsr_metadata.to_excel(writer, 'rdsr_metadata')
    dicom_metadata.to_excel(writer, 'dicom_metadata')
    output['rdsr_metadata'] = rdsr_metadata
    output['dicom_metadata'] = dicom_metadata
    dose_data.to_excel(writer, 'rdsr_dump')
    output['dose_data'] = dose_data
    output['rdsr'] = rdsr
    writer.save()
    return output

def load_rdsr(fn):
    rdsr = nav.read_file(fn)
    rdsr_data = rdsr_dose_data_to_df(rdsr)
    return rdsr_data

#%%

if __name__ == "__main__":
    rdsr_fn = 'M:/MedPhy/General/^TEAP/Chris/SR.1.2.840.113619.2.416.\
        319378472768174790945601907133479830326.dcm'
    #rdsr_fn = sys.argv[1]
    print('test layer 2')
    print(f'fn{rdsr_fn}')
    datas = dump_rdsr(rdsr_fn)
