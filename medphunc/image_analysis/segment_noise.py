# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:33:52 2020

Calculate brain std

@author: willcx
"""


import pydicom

import pandas as pd
import numpy as np
import scipy
from matplotlib import pyplot as plt


from scipy.ndimage.morphology import binary_fill_holes, binary_opening, binary_erosion

import cv2

from medphunc.image_analysis.image_utility import window_std, apply_cv_roi, cv_roi_to_index
from medphunc.image_io.ct import rescale_ct_dicom, load_ct_folder


import logging
logger = logging.getLogger(__name__)

#%%

class Segmenter:
    
    data = {'tissue':{'threshold':(-30,30)},
            'bone':{'threshold':(200,2500)}
            }
    # Addtional options: 
    # ROI:
    # 'topleft':{'cv_roi':(z0,z1, cv_roi)}
    
    
    window_diameter = 20
    
    def __init__(self,
                 im,
                 z_input = 'all'):
        self.im = im
        self.set_z(z_input)
        self.im = self.im[self.z,]
    
    def set_z(self, z = 'all'):
        if z == 'all':
            if len(self.im.shape) == 3:
                self.z = np.arange(self.im.shape[0])
            else:
                self.z = 0
                self.im = im[np.newaxis,]
        else:
            self.z = z
        
    @classmethod
    def from_dicom(cls, dcm):
        return cls(dcm.pixel_array)
    
    @classmethod
    def from_nifti(cls, n, z = 'all'):
        # NOT IMPLEMENTED
        raise(ValueError('not implemented'))

    @classmethod
    def from_ct_folder(cls, folder, z = 'all'):
        im, d, end_points = load_ct_folder(folder)
        return cls(im, z)
        
    @classmethod
    def from_filename(cls, fn):
        d = pydicom.dcmread(fn)
        return cls(d.pixel_array)
    
    @classmethod
    def from_image(cls, im):
        return cls(im)

    @classmethod
    def from_image_stack(cls, imstack, z = 'all'):
        return cls(imstack, z)


    def generate_masks(self):
        for region, region_data in self.data.items():
            if 'threshold' in region_data:
                threshold = region_data['threshold']
                self.data[region]['mask'] = (self.im > threshold [0]) & (self.im < threshold[1])
                
            if 'cv_roi' in region_data:
                m = np.zeros(self.im.shape)
                i = cv_roi_to_index(region_data['cv_roi'])
                m[:,i[0]:i[1], i[2]:i[3]] = 1
                self.data[region]['mask'] = m==1
                
                
    
    def generate_std_map(self):
        self.im_std = np.array([window_std(self.im[i,], self.window_diameter) for i in range(self.im.shape[0])])

    def calculate_signal(self):
        for region, region_data in self.data.items():
            mask = self.data[region]['mask']
            vals = self.im[mask]
            self.data[region]['HU'] = scipy.stats.mode(vals).mode.mean()
        

    def calculate_noise(self):
        for region, region_data in self.data.items():
            mask = self.data[region]['mask']
            std_vals = self.im_std[mask]
            z = np.histogram(std_vals,
                             bins=200,
                             range=[np.percentile(std_vals, 3),
                                    np.percentile(std_vals,97)])
            std_mode = z[1][np.where(z[0]==z[0].max())][0]
            self.data[region]['noise'] = std_mode

            if std_mode == 0:
                logger.error('Calculated a standard deviation of 0, which is not plausible')

    def calculate_results(self):
        self.generate_masks()
        self.generate_std_map()
        self.calculate_noise()
        self.calculate_signal()
        results = pd.DataFrame(self.data).T
        self.results = results.drop(columns='mask')
        return self.results
        


#%%

class HeadNoise(Segmenter):
    
    data = {}
    
    def __init__(self,
                 im,
                 z_input = 'all'):
        self.im = im
        self.set_z(z_input)
        self.im = self.im[self.z,]
        self.segment_brain()
        self.calculate_results()
        
    def segment_brain(self, threshold=200):
        
        masks = []
        for i in range(self.im.shape[0]):
            
            mask = self.im[i,:] > threshold
            inner_mask = binary_fill_holes(mask) ^ mask
            inner_mask = binary_opening(inner_mask, iterations=24)
            inner_mask = binary_erosion(inner_mask, iterations=30)
            masks.append(inner_mask)
        mask = np.array(masks)
        if mask.sum() == 0:
            raise(ValueError("The segmentation of the image supplied contained was empty"))
        self.data['brain'] = {}
        self.data['brain']['mask'] = mask

#%%

def segment_head_to_inner_brain(im, threshold=200):
    mask = im > threshold
    inner_mask = binary_fill_holes(mask) ^ mask
    inner_mask = binary_opening(inner_mask, iterations=12)
    inner_mask = binary_erosion(inner_mask, iterations=18)
    return inner_mask
    
    
def calculate_head_noise_mode_from_image(im, threshold=200, window_diameter=20):

    inner_mask = segment_head_to_inner_brain(im, threshold)
    
    im_std = window_std(im, window_diameter)

    std_vals = im_std[inner_mask]
    
    #bin the pixel values, then find the most common bin
    z = np.histogram(std_vals, bins=200, range=[0,20])
    std_mode = z[1][np.where(z[0]==z[0].max())][0]
    if std_mode == 0:
        logger.error('Calculated a standard deviation of 0, which is not plausible')
    return std_mode

def calculate_head_noise_mode_from_dicom(d, threshold=200, window_diameter=20):
    '''
    Calculate the mode of the noise within the brain in an axial CT brain image

    Parameters
    ----------
    d : pydicom dataset
        Contain CT brain axial image data.
    threshold : int, optional
        Threshold for segmenting bone in CT head. The default is 200.

    Returns
    -------
    std_mode : float
        Mode of noise in the input image.

    '''
    im = rescale_ct_dicom(d)
    return calculate_head_noise_mode_from_image(im, threshold, window_diameter)


def calculate_head_noise_mode_from_file(fn, threshold=200):
    d = pydicom.dcmread(fn)
    return calculate_head_noise_mode_from_dicom(d)


def show_noise_thresholds(d, threshold=200):
    '''
    Calculate the mode of the noise within the specified region of a CT brain
    image, then plot the results

    Parameters
    ----------
    d : pydicom dataset
        Contain CT brain axial image data.
    threshold : int, optional
        Threshold for segmenting bone in CT head. The default is 200.

    Returns
    -------
    None

    '''
    im = rescale_ct_dicom(d)
    inner_mask = segment_head_to_inner_brain(im, threshold)
    im_std = window_std(im, 7)
    std_vals = im_std[inner_mask]
    
    #bin the pixel values, then find the most common bin
    plt.hist(std_vals,bins=50)
    plt.show()
    
    z = np.histogram(std_vals, bins=50)
    std_mode = z[1][np.where(z[0]==z[0].max())][0]
    

    ntiles = [10,20,30,40,50]
    cutoffs = np.percentile(std_vals, ntiles)
    
    cutoffs = np.concatenate([[std_mode], cutoffs])
    
    for ntile, cutoff in zip(ntiles, cutoffs):
        m = inner_mask * (im_std < cutoff)
        imm = im * m
        
        print(f'ntile:{ntile}\ncutoff:{cutoff}\n')
        plt.imshow(imm)
        plt.show()



#%%

if __name__ == '__main__':
    

    fn = 'M:/MedPhy/General/Export_2020-01-20_12-05-01/10000000/10000001/10000098/100000BC'
    
    d = pydicom.dcmread(fn)
    im = d.pixel_array
    show_noise_thresholds(d)


