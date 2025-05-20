# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:51:05 2019

@author: WilliamCh
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

import cv2
import math

import logging

logger = logging.getLogger('ii_dose_extractor')
logger.setLevel(logging.INFO)

logger.addHandler(logging.StreamHandler())

#%% Functions written for helping with parsing paddleocr data

def index_nearest(value, reference_values):
    diffs = np.array([np.abs(value-ref_val) for ref_val in reference_values])
    out = np.where(diffs == diffs.min())
    if len(out[0]) == 1:
        return out[0][0]
    else:
        raise(ValueError('failed to get a unique result for value %s', value))


def bin_into_rows_or_columns(series, n_bins=None, do_print=False):

    if not n_bins:
        n_bins = int((series.max()-series.min())//8)
        print(n_bins)
    hist, bin_edges = np.histogram(series, bins=n_bins)

    hist = np.pad(hist,1)
    bin_edges = np.pad(bin_edges,1)
    bin_edges[0] = bin_edges[1]-(bin_edges[2]-bin_edges[1])
    bin_edges[-1] = bin_edges[-2]-(bin_edges[-3]-bin_edges[-2])
    peaks, _ = find_peaks(hist, height=0)  # Adjust 'height' as needed
    
    if do_print:
        plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="black", align="edge")
        plt.plot(bin_edges[peaks], hist[peaks], "x")
        plt.show()
        
    bin_edge_l = bin_edges[peaks]
    bin_edge_r = bin_edges[1:][peaks]
    centerpoints = (bin_edge_l+bin_edge_r)/2
    return series.apply(lambda x: index_nearest(x, centerpoints))


def pivot_results_to_original_table(df):
    return df.pivot_table(index='row_number', columns='col_number',values='text',aggfunc=concat_strings, fill_value='')


def convert_header_rows_to_columns(df, header_rows=1):
    #over-ride column names and note that the columns actually takes up two rows
    df.columns = df.iloc[0:header_rows,:].sum()
    df = df.loc[header_rows:,:]
    return df


def paddle_results_to_dataframe(result,
                                bin_rows=None,
                                bin_cols=None,
                                do_pivot=False,
                                do_print=False,
                               col_alignment = 'center',
                               row_alignment='center'):
    df=pd.DataFrame(result[0]['res'])

    df['x_start'] = df.text_region.str[0].str[0]
    df['x_end'] = df.text_region.str[1].str[0]
    df['x_av'] = df.x_start+df.x_end
    df['y_start'] = df.text_region.str[0].str[1]
    df['y_end'] = df.text_region.str[2].str[1]
    df['y_av'] = df.y_start+df.y_end
    if col_alignment=='left':
        df['x_ref'] = df.x_start
    elif col_alignment=='right':
        df['x_ref'] = df.x_end
    elif col_alignment == 'center':
        df['x_ref'] = df.x_av
    else:
        raise(ValueError('alignment needs to be left, right, or center'))
    if row_alignment=='top':
        df['y_ref'] = df.y_start
    elif row_alignment=='bottom':
        df['y_ref'] = df.y_end
    elif row_alignment == 'center':
        df['y_ref'] = df.y_av
    else:
        raise(ValueError('alignment needs to be left, right, or center'))
    
    df = df.sort_values('y_ref').reset_index()
    
    if bin_rows:
        if bin_rows is True:
            df['row_number'] = bin_into_rows_or_columns(df.y_ref, do_print=do_print)
        else:
            df['row_number'] = bin_into_rows_or_columns(df.y_ref, bin_rows, do_print=do_print)

    
    if bin_cols:
        if bin_cols is True:
            df['col_number'] = bin_into_rows_or_columns(df.x_ref, do_print=do_print)
        else:
            df['col_number'] = bin_into_rows_or_columns(df.x_ref, bin_cols, do_print=do_print)
        
    if do_pivot:
        df = pivot_results_to_original_table(df)
    return df


def concat_strings(series):
    return ' '.join(series)


#%% Functions originally written for parsing using pytesseract

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


if __name__ == '__main__':
    pass

