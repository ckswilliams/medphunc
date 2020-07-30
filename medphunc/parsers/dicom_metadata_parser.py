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
from matplotlib import pyplot as plt

import cv2

#%%

def extract_metadata(d):
    output = {}
    for item in d:
        if item.name in ['Pixel Data', 'pixel_array']:
            continue
        
        output[item.name] = item.value
    return output

#%%
def dicom_to_metadata(fns):
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

#%%


if __name__ == '__main__':
    folder = 'D:\\bts exit strategy\\RANZCR Certification Tests\\System 1 - Hologic\\dcm'
    fns = glob.glob(folder+'**/*', recursive=True)
    
    df = dicom_to_metadata(fns)



