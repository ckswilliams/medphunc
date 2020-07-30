# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:52:50 2018

Script for calculating NNPS. Not finished!

@author: WilliamCh
"""
#%%

#External imports

import numpy as np
from matplotlib import pyplot as plt
import pydicom
from scipy import fftpack
import scipy.ndimage
from scipy.signal.signaltools import detrend
#from scipy.signal import medfilt
from astropy.modeling import models, fitting


# Interal imports

from medphunc.image_analysis.radial_data import radial_data
from medphunc.image_io.ct import load_ct_folder
from medphunc.image_analysis.image_utility import localise_phantom, detrend_region, calculate_radial_average
from medphunc.image_analysis import image_utility as iu



import logging
logger = logging.getLogger(__name__)
import cvlog
    

#%% General functions

def show_region_coords(im, region_coords, region_size=70, pixel_value=0):
    """Show the regions selected for analysis overlaid on the original image"""
    im = im.copy()
    for xy in region_coords:
        im[int(xy[0]-region_size/2) : int(xy[0]+region_size/2),
                     int(xy[1]-region_size/2) : int(xy[1]+region_size/2),] = pixel_value
    if im.ndim ==3:
        imshow = im[:,:,2]
    else:
        imshow=im
    plt.imshow(imshow)
    

def get_regions(im, region_coords, region_size=70,):
    """Extract patches from the image corresponding to squares having the coordinates and size shown"""
    regions = [iu.apply_cv_roi(im, [(xy[1]-region_size/2)//1, (xy[0]-region_size/2)//1, region_size, region_size]) for xy in region_coords]
    #regions = [im[int(xy[0]-region_size/2) : int(xy[0]+region_size/2),
    #                 int(xy[1]-region_size/2) : int(xy[1]+region_size/2),] for xy in region_coords]
    return regions


def detrend_regions(regions):
    flat_regions = [detrend_region(r) for r in regions]
    return flat_regions


def aggregate_nps(regions):
    ffts = [np.fft.fftn(region) for region in regions]
    ffts = [np.abs(fft)**2 for fft in ffts]
    aggregated_nps = np.sum(ffts,axis=0)
    aggregated_nps = np.fft.fftshift(aggregated_nps)
    return aggregated_nps


def calculate_normalisation(pixel_dims, region_size, n_regions):
    return np.product(pixel_dims) * np.product(region_size)


def extract_profiles(fft):
    dim = np.array(fft.shape)
    center = dim//2
    fft = (fft[::-1,::-1]+fft)/2
    x_profile = fft[center[0]-5:center[0]+5,center[1]:]
    x_profile = x_profile.mean(axis=0)
    y_profile = fft[center[0]:,center[1]-5:center[1]+5]
    y_profile = y_profile.mean(axis=1)
    return x_profile, y_profile


def analyse_nnps(nnps, pixel_size, plot=False):
    
    x, y = extract_profiles(nnps[:,:])
    
    if nnps.ndim == 3:
        nnps = nnps[:,:,nnps.shape[2]//2]
    
    f,r = calculate_radial_average(nnps[:,:], pixel_size)
    if plot:
        plt.imshow(nnps[:,:])
        plt.show()
        plt.plot(f,r)
        plt.show()
    
    output = {'x':x,
             'y':y,
             'frequency':f,
             'radial':r}
    return output

#%% Coordinate functions

def calculate_planar_region_coords(yrange, xrange, overlap_size):
    y = np.arange(yrange[0]+overlap_size/2, yrange[1]-overlap_size/2, overlap_size)
    x = np.arange(xrange[0]+overlap_size/2, xrange[1]-overlap_size/2, overlap_size)
    ya, xa = np.meshgrid(y,x)
    ya = ya.ravel()
    xa = xa.ravel()
    return list(zip(ya,xa))


def get_planar_region_coords(im,
                                  region_size=128,
                                  region_overlap=.5):
    
    region_info = iu.localise_homogenous_region(im)


    yrange = region_info['y_range']
    xrange = region_info['x_range']
    overlap_size = region_size*region_overlap
    
    return calculate_planar_region_coords(yrange, xrange, overlap_size)

    

def calculate_circular_coords(center, radius, n):
    k = np.arange(n)/n*np.pi*2
    y = np.int32(radius*np.cos(k)) + center[0]
    x = np.int32(radius*np.sin(k)) + center[1]
    return list(zip(y,x))

def get_ct_region_coords(im, radius = 120, n=16):
    center = localise_phantom(im)['center']
    return calculate_circular_coords(center, radius, n)


#%% Amalgamation functions
    
def calculate_nnps_from_region_coords(im, region_coords, region_size, pixel_size):
    """
    Flexible function for accumulating a set of regions in a uniform phantom
    into an NNPS.

    Parameters
    ----------
    im : np.array
        2d or 3d array, showing a uniform circular object.
    region_size : int, optional
        Width/height of the regions, in pixels. The default is 70.

    Returns
    -------
    nnps : TYPE
        DESCRIPTION.

    """
    regions = get_regions(im, region_coords, region_size)
    regions = detrend_regions(regions)
    nps = aggregate_nps(regions)
    nnps = nps / calculate_normalisation(pixel_size, regions[0].shape, len(regions))
    
    return nnps


#%% CT amalgamation function

def calculate_ct_nnps(im, pixel_size,
                      n_regions=16,
                      radius=120,
                      region_size=70):
    """
    Calculate the NNPS for a 2d or 3d CT noise image, sampling
    regions in a circular pattern. Normalisation not properly validated.

    Parameters
    ----------
    im : np.array
        2d or 3d array, showing a uniform circular object.
    pixel_size : tuple
        x and y pixel dimensions.
    n_regions : int, optional
        number of regions to sample. The default is 16.
    radius : float, optional
        radius of the circlular pattern of regions, in pixels. The default is 120.
    region_size : int, optional
        Width/height of the regions, in pixels. The default is 70.

    Returns
    -------
    nnps : np.array
        normalised noise power spectrum.

    """
    #find_phantom_center(im)
    region_coords=get_ct_region_coords(im, radius, n_regions)
    show_region_coords(im, region_coords, region_size*2//3)
    
    return calculate_nnps_from_region_coords(im, region_coords, region_size, pixel_size)

def calculate_planar_nnps(im, pixel_size,
                      region_size=128,
                      region_overlap = 0.5):
    """
    Calculate the NNPS for a 2d noise image, sampling
    regions in a circular pattern. Normalisation not properly validated.

    Parameters
    ----------
    im : np.array
        2d or 3d array, showing a uniform circular object.
    pixel_size : tuple
        x and y pixel dimensions.
    n_regions : int, optional
        number of regions to sample. The default is 16.
    radius : float, optional
        radius of the circlular pattern of regions, in pixels. The default is 120.
    region_size : int, optional
        Width/height of the regions, in pixels. The default is 70.

    Returns
    -------
    nnps : np.array
        normalised noise power spectrum.

    """
    #find_phantom_center(im)
    region_coords=get_planar_region_coords(im, region_size, region_overlap)
    show_region_coords(im, region_coords, region_size*2//3)
    
    return calculate_nnps_from_region_coords(im, region_coords, region_size, pixel_size)

#%%

if __name__ == '__main__':
    
    im, d = load_ct_folder('images/nps/')
    #im = im[:,:,0]
    ct_nnps = calculate_ct_nnps(im, d.PixelSpacing)
    ct_results = analyse_nnps(ct_nnps, d.PixelSpacing[0])
    
    
    d = pydicom.read_file('images/MammoDaily1.IMA')
    region_size=128
    region_overlap=1.1
    mg_nnps = calculate_planar_nnps(d.pixel_array.copy(), 0.085, region_size, region_overlap)
    mg_results = analyse_nnps(mg_nnps, 0.085)
    
