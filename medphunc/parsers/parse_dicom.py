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

#%%

def extract_metadata(d):
    output = {}
    for item in d:
        if item.name in ['Pixel Data', 'pixel_array']:
            continue
        
        output[item.name] = item.value
    return output

def parse_single_dcm(fn):
    if not isinstance(fn, pydicom.dataset.FileDataset):
        d = pydicom.read_file(fn)
    else:
        d = fn
    output = extract_metadata(d)
    df = pd.DataFrame.from_dict(output, orient='index')
    df.columns = ['value']
    return df


#%% 
def dicom_files_to_metadata(fns):
    output = []
    for fn in fns:
        try:
            d = pydicom.read_file(fn)
        except Exception as e:
            dd = {'fn':fn,
                  'error':e}
            output.append(dd)
            continue
        dd = extract_metadata(d)
        dd['fn'] = fn
        output.append(dd)
    df = pd.DataFrame(output)
    return df

def dicom_folder_to_metadata(folder, suffix='.dcm', recursive=True):
    fns = glob.glob(folder+'/**/*'+suffix, recursive=recursive)
    df = dicom_files_to_metadata(fns)
    return df

def dicom_objects_to_dataframe(dicom_objects):
    dd = [extract_metadata(d) for d in dicom_objects]
    return pd.DataFrame(dd)

#%%


if __name__ == '__main__':
    folder = 'D:\\bts exit strategy\\RANZCR Certification Tests\\System 1 - Hologic\\dcm'
    fns = glob.glob(folder+'**/*', recursive=True)
    fn = 'M:/MedPhy/General/^TEAP/Chris/sample_rdsr.dcm'
    
    df = parse_single_dcm(fn)



