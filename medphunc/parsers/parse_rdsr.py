# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:41:53 2019

Crunch an RDSR

@author: willcx
"""

#%%
import rdsr_navigator as nav
import pandas as pd

from medphunc.parsers import generic_dict_parser as dp
from medphunc.parsers.parse_dicom import parse_single_dcm


import logging
logging.basicConfig(filename='rdsr.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
   
   
#%%
def rdsr_line_to_dic(rdsr):
    output = {}

    if len(rdsr.content) > 0:
        for r in rdsr.content:
            try:
                output[r.concept_name.code_meaning] = rdsr_line_to_dic(r)
            except AssertionError as e:
                print(e)
                continue
    else:
        output = rdsr.value
           
    return output
    
def rdsr_to_dic(rdsr):
    output = []
    for r in rdsr.get_all('ct_acquisition'):
        output.append(rdsr_line_to_dic(r))
    for r in rdsr.get_all('irradiation_event_x-ray_data'):
        output.append(rdsr_line_to_dic(r))
    return output


def split_unit_cols(df):
    for col in df.columns:
        test = df[col].apply(type)
        if (test.isin([tuple,list]).sum() > 0):
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
        try:
            output[r.concept_name.code_meaning]
            output[r.concept_name.code_meaning + '_'+bonus[i]] = rdsr_line_to_dic(r)
            i = i + 1
        except:
            output[r.concept_name.code_meaning] = rdsr_line_to_dic(r)
    #Need to turn into a list to match the expected input of function, which is a list of dicts
    df = rdsr_items_to_df([output])
    return df

#%%
    
def export_for_skin_dose_spreadsheet(dose_data, fn='temp.xlsx'):
    df = dose_data.copy()
    df.dose_area_product = df.dose_area_product*10000
    df['dose_(rp)'] = df['dose_(rp)']*1000
    df.collimated_field_area = df.collimated_field_area * 10000
    
    export_cols = ['kvp',
    'fluoro_mode',
    'exposure_time',
    'dose_area_product',
    'dose_(rp)',
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
    df = df.loc[:,export_cols]
    df.to_excel(fn, 'psd_export')
    return df

def dump_rdsr(fn, output_fn = 'rdsr_dump.xlsx'):
    rdsr = nav.read_file(fn)
    output = {}
    writer = pd.ExcelWriter(output_fn, engine='xlsxwriter')
    rdsr_metadata = rdsr_meta_data_to_df(rdsr).T
    rdsr_metadata.columns = ['value']
    dicom_metadata = parse_single_dcm(fn)
    dose_data = rdsr_dose_data_to_df(rdsr)
    try:
        df = export_for_skin_dose_spreadsheet(dose_data, writer)
        output['skin_dose_data'] = df
    except:
        pass
    
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
    rdsr_fn = 'M:/MedPhy/General/^TEAP/Chris/SR.1.2.840.113619.2.416.319378472768174790945601907133479830326.dcm'
    #rdsr_fn = sys.argv[1]
    print('test layer 2')
    print(f'fn{rdsr_fn}')
    datas = dump_rdsr(rdsr_fn)
    rdsr = datas['rdsr']
    dose_data = datas['dose_data']
    
    