# -*- coding: utf-8 -*-
"""
Created on Wed May  9 15:54:30 2018

Set of functions for parsing CareAnalytics XML files

@author: WilliamCh
"""

import pandas as pd
from glob import glob
import xmltodict
import logging
import re
import generic_dict_parser as dp
import time
import pdb

logger = logging.getLogger('CareAnalytics_XML_extractor')
logger.setLevel(logging.INFO)
if len(logger.handlers) < 1:
    print('adding logger?')
    ch = logging.StreamHandler()
    logger.addHandler(ch)

#%% Load data files

COLUMN_MAP = pd.read_excel('data/CANameMap.xlsx')
TYPE_MAP = pd.read_excel('data/CAColumnTypes.xlsx')

#%% XML parsing

def CA_xml_file_to_df(fn):
    logger.debug('Reading file: %s',fn)
    with open(fn) as f:
        xml = f.read()
    try:
        x = xmltodict.parse(xml)
    except Exception as e:
        logger.info('Could not read xml file: %s' % fn)
        logger.info('Reason: %s' % e)
        return pd.Dataframe({})
    dose_info = x['root']['Query_Criteria']['DoseInfo']
    if type(dose_info) != list:
        dose_info = [dose_info]
    try:
        dfout = pd.io.json.json_normalize(dose_info)
        dfout = dp.really_screw_with_a_dataframe(dfout)
    except:
        logger.info('Failed to convert xml to dataframe in fn: %s' % fn)
        return  pd.Dataframe({})
    useless_columns = dfout.columns[dfout.columns.contains('@')]
    for column in useless_columns:
        logger.info('Dropping column %s with contents %s' % (column, dfout[column].unique()))
    dfout = dfout.loc[:,dfout.columns[dfout.columns.str.contains('@')]]
    dfout['fn'] = fn.replace('\\','/')
    logger.debug('Returning dataframe with %s lines', dfout.shape[0])
    return dfout
   
def CA_directory_to_df(fn_list):
    logger.info('Processing directory containing %d files', len(fn_list))
    CA_dataframes = [CA_xml_file_to_df(fn) for fn in fn_list]
    output = pd.concat(CA_dataframes)
    return output

def CA_directories_to_df(rootdir, filter_term=None):
    xml_files = glob(rootdir + '/**/*.xml',recursive=True)
    
    if filter_term:
        xml_files = [fn for fn in xml_files if filter_term in fn]
    xml_df = pd.DataFrame({'fn':xml_files})
    xml_df = xml_df.assign(folder=xml_df.fn.str.split('\\').str[0:-1].str.join('/'))
    logger.info('Parsed folders, found %d files',xml_df.shape[0])
    results = []
    for folder in xml_df.folder.unique():
        logger.info('Began processing folder %s',folder)
        fn_list = xml_df.loc[xml_df.folder == folder,'fn'].values
        df = CA_directory_to_df(fn_list)
        df['folder'] = folder.replace('\\','/')
        results.append(df)
    return pd.concat(results, axis=0)

#%% Post import processing

def camelCaseTo_snake(string):
    if string.count('_') > 0:
        return(string.lower())
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def convert_to_openREM_column_style(df):
    logger.debug('Converting CA column style to OpenREM')
    col_names = df.columns
    col_names = [s.split('@')[-1] for s in col_names]
    col_names = [s.replace('-','') for s in col_names]
    col_names = [camelCaseTo_snake(s) for s in col_names]
    df.columns = col_names
    return df

def apply_openREM_headers(df):
    logger.debug('Manually overriding specific CA headers')
    heading_map = COLUMN_MAP.set_index('ca_column_name').openrem_column_name.to_dict()
    return df.rename(columns=heading_map)
   
#Removes the units out of a dataframe
#Default values are for CareAnalytics data
def remove_units(df):
    logger.debug('Removing CA units')
    unit_list = TYPE_MAP[TYPE_MAP.column_name.isin(df.columns)]
    unit_list = unit_list[unit_list['split'] == 1]
    for column in unit_list.column_name:
        try:
            df[column] = df[column].str.split(' ').str[0]       
        except AttributeError as e:
            print(e)
            logger.debug('Could not split column %s, because %s' % (column, e))
    return df

def enforce_column_types(df):
    logger.debug('enforcing column types')
    type_map = TYPE_MAP[TYPE_MAP.column_name.isin(df.columns)]
    type_map = type_map.set_index('column_name').type.to_dict()
    for column, column_type in type_map.items():
        try:
            if column_type == 'datetime64':
                df[column] = df[column].str.split('.').str[0]
            df[column] = df[column].astype(column_type, errors='ignore')
        except Exception as e:
            print('Column %s could not be converted to %s, due to %s' % (column, column_type, e))
            logger.debug('Column %s could not be converted to %s, due to %s' % (column, column_type, e))
    return df

def calculate_age(df):
    logger.debug('Calculating ages')
    try:
        agetype = df.patients_age.str[-1]
        agetype = agetype.map({'Y':1, 'M':12, 'D':365})
        df = df.assign(patient_age_decimal = pd.to_numeric(
                df.patients_age.str[:-1])/agetype)
        df = df.drop(columns = 'patients_age')
    except:
        logger.debug('Patient age not available in data, filling column with 0')
        df['patient_age_decimal'] = 0
    return df
   
def convert_CA_df_to_OpenREM(df):
    df = (df
    .pipe(convert_to_openREM_column_style)
    .pipe(apply_openREM_headers)
    .pipe(remove_units)
    .pipe(enforce_column_types)
    .pipe(calculate_age))
    return df

#%% Process directory containing XML files

def CA_directories_to_openREM_df(rootdir, filter_term = None):
    logger.info('Processing %s', rootdir)
    df = CA_directories_to_df(rootdir, filter_term = filter_term)
    df = convert_CA_df_to_OpenREM(df)
    df = df.drop_duplicates('irradiation_event_uid')
    df = df.reset_index(drop=True)
    df['id'] = df.irradiation_event_uid
    logger.info('XML data converted to dataframe with %s rows', df.shape[0])
    return df

#%% Run the script


if __name__ == '__main__':
    import pickle
    t0 = time.time()
    rootdir = 'Y:/BTS-Herston-MedPhys/MedPhys/BTS Central/Projects/Historic dose data/West Moreton/FL'
    rootdir = 'Y:\\BTS-Herston-MedPhys\\MedPhys\\BTS Central\\Projects\\Historic dose data\\test'
    df = CA_directories_to_openREM_df(rootdir)
    with open('c:/Chris/dockershare/moreton_fl.p', 'wb') as f:
        pickle.dump(df, f)
    #df.to_csv('C:/Users/WilliamCh/Desktop/sillydata.csv')
    t1 = time.time()
    print('Completed in {} seconds'.format(t1-t0))