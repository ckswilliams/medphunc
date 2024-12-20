# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:05:44 2019

@author: WILLCX
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 09:53:17 2019

@author: willcx
"""


import glob
import pydicom
import pandas as pd
import pathlib
import datetime
from typing import Type

#%%

def extract_metadata(d: Type[pydicom.Dataset], include_image=False) -> dict:
    """
    Extract metadata from a pydicom object for the purpose of creating a table
    for inter-object comparison

    Parameters
    ----------
    d : Type[pydicom.Dataset]
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    if not isinstance(d, pydicom.Dataset):
        raise(ValueError('Defined for pydicom objects only, the input was of type %s', type(d)))
    output = {}
    for item in d:
        if item.name in ['Pixel Data', 'pixel_array']:
            if include_image:
                output[item.name] = d.pixel_array
                continue
        
        if (item.name in output) or (item.name == 'Private Creator'):
            export_tag_name = f'{item.name}_{item.tag}'
        else:
            export_tag_name = item.name
        output[export_tag_name] = item.value
    return output

def parse_single_dcm(fn, include_image=False):
    if not isinstance(fn, pydicom.dataset.FileDataset):
        d = pydicom.dcmread(fn)
    else:
        d = fn
    output = extract_metadata(d, include_image)
    df = pd.DataFrame.from_dict(output, orient='index')
    df.columns = ['value']
    return df


#%% 
def dicom_files_to_metadata(fns, include_image=False):
    output = []
    for fn in fns:
        try:
            d = pydicom.dcmread(fn)
        except Exception as e:
            dd = {'fn':fn,
                  'error':e}
            output.append(dd)
            continue
        dd = extract_metadata(d, include_image)
        dd['fn'] = fn
        output.append(dd)
    df = pd.DataFrame(output)
    return df

def dicom_folder_to_metadata(folder, suffix='.dcm', recursive=True, one_per_folder=False, include_image=False):
    p = pathlib.Path(folder)
    fns = list(p.glob('**/*'+suffix))
    if one_per_folder:
        fns = {fn.parent:fn for fn in fns}.values()
    df = dicom_files_to_metadata(fns, include_image)
    return df

def dicom_objects_to_dataframe(dicom_objects, include_image=False):
    dd = [extract_metadata(d, include_image) for d in dicom_objects]
    return pd.DataFrame(dd)


def calculate_age_decimal(dicom_object):
    x=(pd.to_datetime(dicom_object.StudyDate, yearfirst=True)- pd.to_datetime(dicom_object.PatientBirthDate, yearfirst=True))
    return x.days/365.25


#%%





if __name__ == '__main__':
    folder = 'D:\\bts exit strategy\\RANZCR Certification Tests\\System 1 - Hologic\\dcm'
    fns = glob.glob(folder+'**/*', recursive=True)
    fn = 'M:/MedPhy/General/^TEAP/Chris/sample_rdsr.dcm'
    
    df = parse_single_dcm(fn)



