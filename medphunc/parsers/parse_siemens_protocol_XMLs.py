# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 09:00:49 2018

@author: WilliamCh
"""
import numpy as np
import pandas as pd
from glob import glob
import xmltodict
import logging
import codecs
import generic_dict_parser as dp

#%%

logger = logging.getLogger(__name__)

#%%



def process_xml_protocol_folder_to_df(path, suffix='.Adult'):
    
    dfs = []
    fns = glob(path + '**/*'+suffix, recursive=True)
    for fn in fns:
        logger.info('Loading new file: %s', fn)
        with codecs.open(fn, 'rb') as f:
            xml_string = f.read()
        xml_dict = xmltodict.parse(xml_string)
        df = dp.xml_dict_to_df(xml_dict, remove_attribs = True)
        #df = pd.DataFrame.from_dict(xml_dict, orient='index')
        dfs.append(dp.really_screw_with_a_dataframe(df))
    dfout = pd.concat(dfs).assign(FilePath=path)
    return dfout

#%%

def get_exposure_data(df):
    nixstrings = ['Bolus', 'Contrast', 'PauseType', 'ReconType', 'MemoType']
    df = df.loc[:,~df.columns.str.contains('|'.join(nixstrings))].drop_duplicates()
    return df


def parse_siemens_xml(fn):
    # Load all the approriate data
    df = dp.xml_file_to_df(fn)
    df.columns = df.columns.str.replace('.\b','')
    col_types = df.columns.str.split('.').str[-2]
    recon_columns = col_types == 'ReconJob'
    scan_columns = col_types == 'ScanEntry'
    df.columns = df.columns.str.split('.').str[-1]
    
    df = df.rename(columns={'BodySize':'patient_type',
               'RegionName':'anatomy',
               'ProtocolName':'study_description',
               'SeriesDescription':'series_description'
               })
        
    # apply column name mapping
    #name_map = ??
    #df.rename(name_map)
    return df, recon_columns, scan_columns

def parse_toshiba_xml(fn):
    #Load the xml
    xml_dict = dp.read_xml_to_dict(fn)
    #xml_dict = dp.recursive_remove_attribs(xml_dict)
    xml_protocol_dict = xml_dict['EPReport']['ReportData']['EPProtocols']
    df = dp.xml_dict_to_df(xml_protocol_dict)
    df, redundant_columns = dp.redundant_columns_to_dict(df)
    df = dp.snip_redundant_column_names(df)
    df = df.loc[:,~df.columns.str.contains('@DisplayName')]
    df.columns = df.columns.str.replace('.#text','')
    
    # Apply column name mapping
    col_types = df.columns.str.split('.').str[-2]
    recon_columns = col_types == 'ReconModes'
    scan_columns = col_types == 'ScanModeParam'
    
    df = df.rename(columns={'PatientTypeName':'patient_type',
               'OrganName':'anatomy',
               'ExamPlanName':'study_description',
               'ApplicationType':'series_description'
               })
    
    
    return df, recon_columns, scan_columns

    



#%%
if __name__ == '__main__':
    
    
    p = 'C:/Users/WilliamCh/Desktop/ct_protocols/gympie/ProtocolsList.xml'
    
    suf = '.Adult'
    df = dp.xml_file_to_df(p)
    df, d = dp.redundant_columns_to_dict(df)
    df = dp.snip_redundant_column_names(df)
    df = process_xml_protocol_folder_to_df(p, suf)
    edf = get_exposure_data(df)

#%%





