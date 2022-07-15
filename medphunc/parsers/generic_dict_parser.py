# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 09:00:49 2018

Generic dictionary parser designed for use dataframes with nested dictionaries and lists.

Dictionaries are expanded into multiple columns, while lists are created as multiple entries.

As a broad comment, this module has a moderate ammount of sorcery and is poorly commented. However, it is used
extensively in my other projects.

@author: WilliamCh
"""

import pandas as pd
import xmltodict
import logging
import collections
import codecs
import os
import json
from typing import List, Union

# %%
logger = logging.getLogger(__name__)


# %% XML parsing

def expand_dictionary_to_columns(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Expand any dictionaries in the specified column of the dataframe, so that 
    each key is converted to a new column.
    """
    logger.debug('Expanding all dictionaries in dataframe')

    tdf = df.loc[:, [column]]
    json_struct = json.loads(tdf.to_json(orient="records"))
    tdf = pd.json_normalize(json_struct)
    tdf.index = df.index

    df = pd.concat([df.loc[:, df.columns != column], tdf], axis=1)
    df = df.loc[:, df.columns != column].copy()
    return df


def explode_list(df: pd.DataFrame, lst_cols: List[str]) -> pd.DataFrame:
    """
    Explode any lists in the specified column such that each element is in
    it's own row. All other columns are replicated for each element in the list
    """
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]

    for lst_col in lst_cols:
        df = df.explode(lst_col)
    return df


# %%

def find_explode_list(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find columns that contain lists, then explode them
    """
    for column in df.columns:
        test = df[column].apply(lambda x: type(x) == list)
        if test.any():
            logger.debug('Exploding lists in %s to new rows', column)
            return pd.concat([explode_list(df.loc[test, :], column), df.loc[~test, :]]).reset_index().iloc[:, 1:]
    return df


# %%

def find_explode_dict_to_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find columns that contain dictionaries, then expand them
    """
    for column in df.columns:
        test = df[column].apply(lambda x: isinstance(x, collections.Mapping))

        if test.any():
            logger.debug('Found dictionary column named %s, exploding all columns', column)
            df = expand_dictionary_to_columns(df, column)
    return df


# %% Explode and expand, repeatedly

def really_screw_with_a_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    As the name implies
    """
    while True:
        # test for columns with lists
        try:
            tdf = find_explode_list(df)
            if tdf is df:
                break
            df = tdf

            tdf = find_explode_dict_to_columns(df)
            if tdf is df:
                break
            df = tdf
        except Exception as e:
            logging.warning('Could not screw with dataframe due to %s', e)
            return df
    return df


# %%

def recursive_remove_attribs(xml_dict: dict) -> dict:
    """
    Remove any elements in the dictionary beginning with @
    De-nest elements in the dictionary that are dictionaries with one element
    Highly recursive in the worst possible way
    """
    poplist = []
    for k, v in xml_dict.items():
        if type(v) == list:
            for i in range(len(xml_dict[k])):
                xml_dict[k][i] = recursive_remove_attribs(xml_dict[k][i])
        if type(v) == collections.OrderedDict:
            xml_dict[k] = recursive_remove_attribs(v)
        if k[0] == '@':
            poplist.append(k)
    for k in poplist:
        xml_dict.pop(k)

    if len(xml_dict) == 1:
        if type(xml_dict) == collections.OrderedDict:
            xml_dict = list(xml_dict.values())[0]

    return xml_dict


# %%

def redundant_columns_to_dict(df: pd.DataFrame) -> (pd.DataFrame, dict):
    redundant_dict = {}
    for c in df.columns:
        if len(df[c].unique()) == 1:
            redundant_dict[c] = df[c][0]
            df = df.drop(columns=c)
    return df, redundant_dict


def snip_redundant_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolves some messy column names that arise from this method of converting xml dictionaries to dataframes.

    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    pd.DataFrame

    """
    colsplit = df.columns.str.split('.')
    i = 0
    for i in range(colsplit.str.len().min()):
        if len(colsplit.str[i].unique()) > 1:
            break
    newcols = colsplit.str[i:]
    newcols = newcols.str.join('.')
    df.columns = newcols
    return df


# %% External call functions. May not be as useful in general cases

def xml_dict_to_df(xml_dict: dict, remove_attribs: bool = False) -> pd.DataFrame:
    """
    Convert an xml dictionary to a dataframe
    """
    if remove_attribs:
        xml_dict = recursive_remove_attribs(xml_dict)
        logger.debug('Removed attributes')
    df = pd.json_normalize(xml_dict)
    df = really_screw_with_a_dataframe(df)
    logger.debug('Returning dataframe with %s lines', df.shape[0])
    return df


def read_xml_to_dict(fn: Union[str, bytes, os.PathLike]) -> dict:
    with codecs.open(fn, 'rb') as f:
        xml_string = f.read()
    return xmltodict.parse(xml_string)


def xml_file_to_df(fn: Union[str, bytes, os.PathLike], remove_attribs: bool = False) -> pd.DataFrame:
    xml_dict = read_xml_to_dict(fn)
    df = xml_dict_to_df(xml_dict, remove_attribs)
    df, d = redundant_columns_to_dict(df)
    df = snip_redundant_column_names(df)
    return df
