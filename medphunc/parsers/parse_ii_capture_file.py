# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:51:05 2019

@author: WilliamCh
"""
from PIL import Image
import pytesseract
import pydicom
import re
import pandas as pd
import cv2
import numpy as np
import math

import logging

logger = logging.getLogger('ii_dose_extractor')
logger.setLevel(logging.INFO)

logger.addHandler(logging.StreamHandler())

#%% Parse data and extract relevant text for philips reports

def segment_lines(X):

    horizontal = np.copy(X)
    vertical = np.copy(X)
    
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = math.ceil(cols / 25)
    
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    
    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = math.ceil(rows / 25)
    
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    
    res = vertical + horizontal
    return res


def remove_lines_from_image(X):
    lines = segment_lines(X)
    X = X-lines
    X[X<X.max()] = 0
    return X


def make_8bit(X):
    Y = X.copy()
    Y = (Y/Y.max()*255).astype(np.uint8)
    return Y    


def make_binary(X, background='dark'):
    X = make_8bit(X)
    if background=='light':
        X = 255-X
    X = cv2.adaptiveThreshold(X, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY,
                              15, 9)
    if X[0,0] == 255:
        X=255-X
    return X

def resize_array(X, ratio=5):
    X = cv2.resize(X, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    return X

def ocr_pixel_array_to_dataframe(X):
    dd = pytesseract.image_to_boxes(Image.fromarray(X), output_type='dict')
    df = pd.DataFrame(dd)
    return df

def extract_dose_total_section(pixel_array):
    df = ocr_pixel_array_to_dataframe(pixel_array)
    
    #Arrange the df chars into a string
    text = ''.join(df.char)
    #Find 'total' in the capture image
    r = re.search('Total', text)
    tdf = df.iloc[r.span()[0]:r.span()[1],:]
    
    minval = tdf.bottom.min()
    maxval = tdf.top.max()
    padval = maxval - minval
    
    return pixel_array[-maxval-padval:-minval+padval, 
                       tdf.right.max():-padval*3].copy()

def extract_words_from_section(X, ratio=1):
    X = resize_array(X, ratio)
    df = ocr_pixel_array_to_dataframe(255-np.pad(X,10,'constant'))
    
    minval = df.bottom.min()
    maxval = df.top.max()
    padval = maxval - minval
    
    df['entry_number'] = (df.right - df.shift(1).left  > padval*3).cumsum()
    words = df.groupby('entry_number').apply(lambda x: x.sum()).char.tolist()
    logger.debug(words)
    return words    


def index_pixels_to_dataframe(X, df):
    minx = df.left.min()
    maxx = df.right.max()
    miny = df.bottom.min()
    maxy = df.top.max()
    padval = (maxy - miny)//2
    return X[-(maxy+padval):-(miny-padval),minx-padval:maxx+padval]


def get_numerical_section(X, line_number_count_cutoff=3):
    X = np.pad(X, X.shape[0]//5, 'constant')
    df = ocr_pixel_array_to_dataframe(255-X)
    df = df[df.char.str.contains('[\d:.]')].copy()
    df['linebin'] = df.bottom//(X.shape[0]//35)*(X.shape[0]//35)
    tdf = df.groupby('linebin').char.count()
    linebin = tdf[tdf>line_number_count_cutoff].index.min()
    df=df[df.linebin==linebin]
    X = index_pixels_to_dataframe(X, df)
    return X   


def philips_get_numerical_section(pixel_array, ratio=1):
    X = make_binary(pixel_array)
    X = remove_lines_from_image(X)
    X = resize_array(X, ratio)
    #x, X = cv2.threshold(X, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    X = get_numerical_section(X, 3)
    return X

def philips_extractor(pixel_array):

    for i in range(1,8):
        X = philips_get_numerical_section(pixel_array, i)
        words = extract_words_from_section(X)
        if i == 6:
            X = philips_get_numerical_section(pixel_array, 4)
        if i == 7:
            X = philips_get_numerical_section(pixel_array.max()-pixel_array, 4)
        if len(words) > 2:
            break     
    
    try:
        float(words[0])
        float(words[1])
        if words[2].find('\d:\d') == -1:
            raise Exception('No colon in time value')
    except:
        words2 = extract_words_from_section(X, 3)
        try:
            float(words2[0])
            float(words2[1])

            words = words2
        except:
            if len(words2) > len(words):
                words = words2

    words = [re.match('[0-9.:]+', str(w)).group() for w in words]
    data = {'screening_time':words[2], 'dap':words[1], 'dose':words[0]}
    #Should be dose units of Gycm^2 and mGy
    return data


def hologic_extractor(pixel_array):
    X = pixel_array
    #X = make_binary(pixel_array, background='light')
    #X = 255-remove_lines_from_image(X)
    X = resize_array(X, 3)
    X = get_numerical_section(X, 9)
    #X = resize_array(X, 5)
    words = extract_words_from_section(X)
    data = {'screening_time':words[-4], 'dap':words[-2], 'dose':words[-1]}
    data['dap'] = data['dap']/100 #Convert to Gycm^2
    
    return data


def dose_data_from_dataset(d):
    if d.Manufacturer.find('Philips') != -1:
        #if d.ManufacturerModelName.find('BV') != -1:
        #    data = bv_extractor(d.pixel_array)
        #else:
        data = philips_extractor(d.pixel_array)
    elif d.Manufacturer.find('Hologic') != -1:
        data = hologic_extractor(d.pixel_array)

    return data

#%% Extract dicom metadata

def convert_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def get_dicom_tag_info(dcm):
    out_list = ['StudyDate',
                'InstitutionName',
                'Manufacturer',
                'ManufacturerModelName',
                'StationName',
                'StudyDescription',
                'StudyTime']
    out_dic = {}
    for o in out_list:
        try:
            out_dic[convert_to_snake(o)] = dcm.get(o)
        except:
            pass
    return out_dic

#%% Almagomation functions

def decompress_and_load(fn, temp_fn='temp.dcm'):
    import subprocess
    GDCM = 'C:/Program Files/GDCM 3.0/bin/gdcmconv.exe'
    args = [GDCM, '-w', fn, temp_fn]
    p = subprocess.Popen(args, shell = False)#, cwd = '/app/openrem/remapp/netdicom',
    x = p.communicate()
    d = pydicom.read_file(temp_fn)
    #os.remove(temp_fn)
    return d


def dose_data_from_dicom(fn):
    try:
        dcm = pydicom.read_file(fn)
        dcm.pixel_array
    except:
        dcm = decompress_and_load(fn)
    dose_data = dose_data_from_dataset(dcm)
    tag_info = get_dicom_tag_info(dcm)
    return {**dose_data, **tag_info}

#%% Postprocessing/validation


def validate_ii_data(df):
    dap_ratio = (pd.to_numeric(df.dap, errors='coerece')
                /pd.to_numeric(df.dose, errors='coerce'))
    
    dodgy_mask = dap_ratio != dap_ratio
    
    dodgy_mask = dodgy_mask | (np.abs(dap_ratio-dap_ratio.mean()) >= (2*dap_ratio.std()))
    
    df = df[~dodgy_mask]
    df = df[~df.duplicated()]
    
    return df


#%%


if __name__ == '__main__':
    import glob
    fns = glob.glob('Y:/Temp/chris_shared_data/magic_ii_datapull/twb/tf/*.dcm')
    output = []
    current = fns.index(fn)
    for fn in fns[current:]:
    #for fn in fns:
        results = dose_data_from_dicom(fn)
        output.append(results)
        print(fn)
        print(results)
    

#%%








