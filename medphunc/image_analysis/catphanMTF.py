# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 09:34:08 2018

Script for calculating the MTF of the wire tool in a catphan

@author: WilliamCh
"""
import pydicom
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
#import cv2
from scipy.ndimage import morphology
from scipy import integrate
import sys

from medphunc.image_analysis import image_utility as iu

import cvlog as log
import logging


#%% debugging
log_verbosity = 1
def log_image(im_array, priority = 2):
    if priority <= log_verbosity:
        fig, ax = plt.subplots()
        ax.imshow(im_array)
        fig.show()
def log_plot(x, y, priority = 2, **kwargs):
    if priority <= log_verbosity:
        fig, ax = plt.subplots()
        ax.plot(x,y, **kwargs)
        fig.show()
def log_text(text, priority = 2):
    if priority <= log_verbosity:
        print(text)

#%% Extract a ROI containing the wire from a CatPhan image

#Could we have a dictionary containing ROI positions? EG:
catphan_high_contrast_ROIs = {
        'MTF':{'xratio':0.50921,
               'yratio':0.61707,
               'shape':'square',
               'radius':14}
                }
    
def catphan_select_wire_roi(im):
    """
    Take an image of the contrast segment of a catphan. If the phantom
    is correctly oriented, the wire will be in a similar position relative to
    the phantom boundaries.
    """
    #Based on phantom measurements made in ImageJ
    #y_min, y_max = 18.1, 219
    #x_min, x_max = 16.9, 217.2
    #w_x, w_y = 140.5, 120.4
    #yratio = (w_y-y_min)/(y_max - y_min)
    #xratio = (w_x-x_min)/(x_max - x_min)

    yratio = 0.5092085614733699
    xratio = 0.6170743884173739
    
    #Threshold the phantom out
    p_shape = im > -900
    log.image(log.Level.TRACE, im * p_shape)
    p_shape = morphology.binary_erosion(p_shape, iterations = 12)
    p_shape = morphology.binary_dilation(p_shape, iterations = 12)
    log.image(log.Level.TRACE, im * p_shape)
    
    p_y, p_x = np.where(p_shape==True)
    p_y.min(), p_x.max()
    
    #Select the position of the wire based on previous measurements
    ypos = yratio * (p_y.max() - p_y.min()) + p_y.min()
    xpos = xratio * (p_x.max() - p_x.min()) + p_x.min()
    
    r = np.array([int(xpos)-14, int(ypos)-14,28,28])
    a = im[iu.Rslice(r)].copy()
    log.image(log.Level.INFO, a)
    return a


def make_LSF(im_ROI, row_vals, axis):
    """
    Create a line spread function from a ROI
    
    Args:
        im_ROI: the region in which we'll be calculating the LSF
        row_vals:
            the particular rows/columns to average over
        axis:
            the axis in which to calculate the LSF. 0 = vertical, 1 = horiztontal
    
    Returns:
        One dimensional LSF
    """
    if axis == 1:
        LSF = (im_ROI[row_vals, :])
    elif axis == 0:
        LSF = (im_ROI[:, row_vals]).T
    #log_image(LSF, priority = 3)
    LSF = LSF.sum(axis=0)
    LSF_rescale = integrate.simps(LSF)
    LSF = LSF / LSF_rescale
    #log_plot(np.arange(len(LSF)), LSF, priority = 3)
    return LSF


def subtract_background(a, mask):
    # Subtract the background from an image, where the foreground is dictated by a mask
    return a - a[~mask].mean()


def calculate_MTF_from_ROI(im_ROI):

    mask = iu.mask_point(im_ROI)
    mask = morphology.binary_opening(mask)
    log.image(log.Level.TRACE, im_ROI * mask)
    mask = morphology.binary_dilation(mask, iterations = 3)
    im_ROI = subtract_background(im_ROI, mask)
    localisation = iu.localise_segmented_object(mask)
    y, x = localisation['center']
    y_vals = iu.get_nearby_integers(y)
    x_vals = iu.get_nearby_integers(x)
    log_text(x_vals, priority = 5)
    log_text(y_vals, priority = 5)
    
    h_LSF = make_LSF(im_ROI, y_vals, 1)
    v_LSF = make_LSF(im_ROI, x_vals, 0)
    h_MTF = iu.mtf_from_lsf(h_LSF)
    v_MTF = iu.mtf_from_lsf(v_LSF)
    return h_MTF, v_MTF

#%% Import the CatPhan image
def catphan_mtf(im, pixel_spacing):
    im_ROI = catphan_select_wire_roi(im)
    
    h_MTF, v_MTF = calculate_MTF_from_ROI(im_ROI)
    h_f = np.arange(len(h_MTF)) * pixel_spacing[1] / 4
    v_f = np.arange(len(v_MTF)) * pixel_spacing[0] / 4
    
    output = {'Horizontal':{'MTF':h_MTF,
                   'frequency':h_f},
     'Vertical':{'MTF':v_MTF,
                 'frequency':v_f}}

    return output

def catphan_mtf_from_dicom(fn):
    d = pydicom.read_file(fn)
    im = d.pixel_array
    output = catphan_mtf(im, d.PixelSpacing)
    return output
    
    
#%%
def calculate_and_plot_mtf_from_fn(fn):
    plot_name = fn.split('/')[-1]
    MTF_data = catphan_mtf_from_dicom(fn)
    fig = iu.plot_mtf(MTF_data, plot_name)
    return MTF_data, fig

def save_MTF_data(MTF_data, fn):
    MTF_data = pd.io.json.json_normalize(MTF_data, sep='_').to_dict()
    MTF_data = {k:v[0] for k,v in MTF_data.items()}
    df = pd.DataFrame(MTF_data)
    df.to_excel(fn)

#%%

if __name__ == '__main__':
    #Try checking the argument for a filename
    try:
        fn = sys.argv[1]
        print(fn)
    except:
        #If nothing supplied, ask for a filename
        fn = input('Please enter the filename for a CatPhan CT image containing the MTF wire object:\n')
    
    try:
        output = calculate_and_plot_mtf_from_fn(fn)
        out_fn_base = fn.split('\\')[-1].split('/')[-1].split('.')[0]
        output[1].savefig(out_fn_base+'_MTF.png')
        save_MTF_data(output[0], out_fn_base+'_MTF_data.xlsx')
    except Exception as e:
        print(e)
        input('Script failed to run, press enter to continue')
    
#%% Work in progress detection of wires
        '''

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[starty:starty+cropy, startx:startx+cropx]

def catphan_select_wire_roi(im):
    """
    Take an image of the contrast segment of a catphan. If the phantom
    is correctly oriented, the wire will be in a similar position relative to
    the phantom boundaries.
    """
    #Based on phantom measurements made in ImageJ
    #y_min, y_max = 18.1, 219
    #x_min, x_max = 16.9, 217.2
    #w_x, w_y = 140.5, 120.4
    #yratio = (w_y-y_min)/(y_max - y_min)
    #xratio = (w_x-x_min)/(x_max - x_min)
    
    center = crop_center(im, 50,50)
    cutoff = center.mean() + center.std()*4
    global df
    lbl, nlbl = ndimage.label(im>cutoff)
    lbls = np.arange(1, nlbl+1)
    dic = {'id':lbls}
    dic['mean_val'] = ndimage.labeled_comprehension(im, lbl, lbls, np.mean, float, 0)
    dic['size'] = ndimage.labeled_comprehension(lbl, lbl, lbls, np.sum, float, 0)
    df = pd.DataFrame(dic)
    df = df.sort_values('mean_val',ascending=False)
    wire_id = df.iloc[5,0]
    
    wire_mask = lbl==wire_id
    
    pixels = np.array(np.where(im*wire_mask==np.max(im*wire_mask)))
    mid_y, mid_x = pixels.mean(axis=1).astype(np.int)

    buffer = 12

    
    im_out = im[mid_y-buffer:mid_y+buffer,mid_x-buffer:mid_x+buffer].copy()
    plt.imshow(im_out)
    return im_out
'''