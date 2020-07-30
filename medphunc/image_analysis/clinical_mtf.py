# -*- coding: utf-8 -*-
"""
Functions for calculating the MTF of arbitrary CT images

Created on Fri Jan 24 15:04:56 2020



@author: Chris Williams
"""

from matplotlib import pyplot as plt
import numpy as np


import pydicom
from scipy.signal import find_peaks

from medphunc.image_io.ct import rescale_ct_dicom
from medphunc.image_analysis import image_utility as iu

import cvlog as log
import logging

#%%
log_verbosity = 5
def log_plot(x, y, priority = 2, **kwargs):
    if priority <= log_verbosity:
        fig, ax = plt.subplots()
        ax.plot(x,y, **kwargs)
        fig.show()
        
#%%

def clinical_mtf(im, pixel_spacing, roi_size=(50, 30)):
    """
    Calculate the MTF from a patient/phantom image. Requires air on the side of
    the patient corresponding to the zero direction of the y axis

    Parameters
    ----------
    im : np.array
        2d or 3d axial CT image. im[Z, Y, X] 
    pixel_spacing : tuple
        pixel size of (y, x) in mm.
    roi_size : tuple,
        Size of the ROI that the MTF will be calculated across. The default is (50, 30).

    Returns
    -------
    output : dictionary
        Dictionary containing the key 'clinical', which contains 'MTF' and 'frequency'

    """
    
    if im.ndim==2:
        im = im[np.newaxis,]
    
    mtfs = []
    fs = []
    
    for i in range(im.shape[0]):
        im_2d = im[i,:]
    
        #find head
        try:
            seg_data = iu.localise_phantom(im_2d, -300)
            if seg_data['anterior_point'][0] < 15:
                im_2d = np.pad(im_2d,15,mode='constant',constant_values=-1024)
                seg_data = seg_data = iu.localise_phantom(im_2d, -300)
        except:
            continue
        y = seg_data['anterior_point'][0]
        x = seg_data['anterior_point'][1]
        
        roi = iu.extract_patch_around_point(im_2d, (y, x), roi_size)
        log.image(log.Level.TRACE, iu.apply_window(roi))
        
        # find the index of array element with the highest rate of change
        roi = iu.align_line_profiles(roi, -500)
        
        w = 10
        c = roi.shape[0]//2
        
        esf = roi[c-w:c+w,:].mean(axis=1)
        
        
        #log_plot(range(len(esf)), esf)
    
        
        mtf = iu.mtf_from_esf(esf)
        mtf = mtf/mtf[0]
        f = pixel_spacing[0]/2*np.arange(mtf.shape[0])
        mtfs.append(mtf)
        fs.append(f)
    
    mtf = np.array(mtfs)
    mtf = np.median(mtf, axis=0)
    f = np.array(fs)
    f = np.median(f, axis=0)
    
    output = {'Clinical':{'MTF':mtf,
                      'frequency':f}}
    
    return output
   

def clinical_mtf_from_dicom(fn):
    "Calculate the clinical MTF from a dicom file, across the patient/air anterior border"
    d = pydicom.read_file(fn)
    im = rescale_ct_dicom(d)
    return clinical_mtf(im, d.PixelSpacing)
    
#%%
    
if __name__ == '__main__':
    #Testing for 
    fn = 'images/catphan/'
    from medphunc.image_io import ct
    im, d = ct.load_ct_folder(fn)
    im = np.moveaxis(im, 2,0)
    results = clinical_mtf(im[20,], d.PixelSpacing)
    iu.plot_mtf(results)
    results = clinical_mtf(im, d.PixelSpacing)
    iu.plot_mtf(results)
    
    fn = 'images/catphan1.dicom'
    results = clinical_mtf_from_dicom(fn)
    iu.plot_mtf(results)
    
