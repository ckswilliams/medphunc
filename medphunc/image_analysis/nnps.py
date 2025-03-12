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
import typing
#from scipy import fftpack
#import scipy.ndimage
#from scipy.signal.signaltools import detrend
#from scipy.signal import medfilt
#from astropy.modeling import models, fitting

from scipy.ndimage import binary_fill_holes, binary_erosion

# Internal imports

#from medphunc.image_analysis.radial_data import radial_data
from medphunc.image_io.ct import load_ct_folder
from medphunc.image_analysis.image_utility import localise_phantom, detrend_region, calculate_radial_average
from medphunc.image_analysis import image_utility as iu



import logging
logger = logging.getLogger(__name__)
    

#%% General functions

def show_region_coords(im, region_coords, region_size=70, pixel_value=0):
    """Show the regions selected for analysis overlaid on the original image"""
    im = im.copy()
    for yx in region_coords:
        im[int(yx[0]-region_size/2) : int(yx[0]+region_size/2),
                     int(yx[1]-region_size/2) : int(yx[1]+region_size/2)] = pixel_value
    if im.ndim ==3:
        imshow = im[1,:,:]
    else:
        imshow=im
    plt.imshow(imshow)
    

def get_regions(im, region_coords, region_size=70,):
    """Extract patches from the image corresponding to squares having the coordinates and size shown"""
    regions = []
    for yx in region_coords:
        cvroi = [int((yx[1]-region_size/2)), int((yx[0]-region_size/2)), region_size, region_size]
        if len(im.shape)==3:
            for i in range(im.shape[0]):
                regions.append(iu.apply_cv_roi(im[i,], cvroi))
        else:
            regions.append(iu.apply_cv_roi(im, cvroi))
    #regions = [iu.apply_cv_roi(im, [(yx[1]-region_size/2)//1, (yx[0]-region_size/2)//1, region_size, region_size]) for yx in region_coords]
    #regions = [im[int(xy[0]-region_size/2) : int(xy[0]+region_size/2),
    #                 int(xy[1]-region_size/2) : int(xy[1]+region_size/2),] for xy in region_coords]
    return regions


def detrend_regions(regions):
    flat_regions = [detrend_region(r) for r in regions]
    return flat_regions


def aggregate_nps(regions):
    ffts = []
    for region in regions:
        fft = np.abs(np.fft.fftn(region))**2
        ffts.append(fft)
    aggregated_nps = np.sum(ffts,axis=0)
    aggregated_nps = np.fft.fftshift(aggregated_nps)
    return aggregated_nps


#todo should I divide or multiply by pixel dimensions?!
def calculate_normalisation(pixel_dims, region_size, n_regions):
    return np.prod(pixel_dims) / np.prod(region_size) / n_regions



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

    

def calculate_circular_coords(center: list, radius: float, n: int) -> list:
    """
    Create a set of coordinates, typically for use in gathering ROIs in a phantom.
    
    In the normal image orientation, starts at the 6-o'clock position and goes counter-clockwise.

    Parameters
    ----------
    center : tuple
        Center y,x values.
    radius : float
        Radius of the circle.
    n : TYPE
        How many coordinates to create.

    Returns
    -------
    list
        List of tuple coordinates [(y,x)].

    """
    k = np.arange(n)/n*np.pi*2
    y = np.int32(radius*np.cos(k)) + center[0]
    x = np.int32(radius*np.sin(k)) + center[1]
    return list(zip(y,x))

def get_ct_region_coords(im, radius = 120, n=16):
    "Localise a CT phantom and find the circular coordinates for ROI extractions"
    center = localise_phantom(im)['center'][-2:]
    return calculate_circular_coords(center, radius, n)

#%%

def analyse_phantom_find_homogeneity(im: np.array) -> typing.List[int]:
    """
    Assuming a cylindrical phantom, defined with HU within -300 < I < 300,
    find the most homogeneous slices, returning all slices indexes 
    with std within 20% of the minimum.
    Ignore the outermost 10 pixels of the phantom.
    Doesn't require contiguous indices at this time

    Parameters
    ----------
    im : np.array
        3d image of a phantom, indices as z,y,x.

    Returns
    -------
    np.array(z_indices)
        array containing all z_indices which have std within 20% of the minimum.

    """
    # Create a mask of where the phantom is within -300,300
    m = (im>-300) & (im<300)
    #binary operations to reduce some possible incosistencies
    m = binary_fill_holes(m)
    m = binary_erosion(m, np.ones([1,3,3]),iterations=10)
    #create a masked array
    mim = np.ma.masked_array(im,~m)
    z_counts = (~mim.mask).sum(axis=(1,2))
    mm = z_counts<(z_counts.max()/2)
    z_std = mim.std(axis=(1,2))
    z_std.mask = mm
    return np.ma.where(z_std < z_std.min()*1.2)[0]

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
    nnps = nps * calculate_normalisation(pixel_size, regions[0].shape, len(regions))
    
    return nnps


#%% CT amalgamation function

def calculate_ct_nnps(im: np.array,
                      pixel_size:typing.Iterable[float],
                      n_regions: int=16,
                      radius: int=120,
                      region_size: int=64):
    """
    Calculate the NNPS for a 2d or 3d CT noise image, sampling
    regions in a circular pattern. Normalisation not properly validated.

    Parameters
    ----------
    im : np.array
        2d or 3d array, showing a uniform circular object.
    pixel_size : tuple
        [y,x] pixel dimensions.
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
    #show_region_coords(im, region_coords, region_size*2//3)
    
    return calculate_nnps_from_region_coords(im, region_coords, region_size, pixel_size)

def calculate_planar_nnps(im: np.array,
                          pixel_size: typing.Iterable[float],
                      region_size: int=128,
                      region_overlap: float=0.5):
    """
    Calculate the NNPS for a 2d noise image, sampling
    regions in a circular pattern. Normalisation not properly validated.

    Parameters
    ----------
    im : np.array
        2d or 3d array, showing a uniform circular object.
    pixel_size : iterable
        [y, x] pixel dimensions.
    region_size : int, optional
        Width/height of the regions, in pixels. The default is 128.
    region_overlap : float, optional
        How much each region should overlap with the adjacent ones. The default is 0.5.

    Returns
    -------
    nnps : np.array
        normalised noise power spectrum.

    """
    #find_phantom_center(im)
    region_coords=get_planar_region_coords(im, region_size, region_overlap)
    show_region_coords(im, region_coords, region_size*2//3)
    
    return calculate_nnps_from_region_coords(im, region_coords, region_size, pixel_size)


def automate_ct_phantom_nnps(im,d, **kwargs):
    phantom_location = iu.localise_phantom(im)
    xr = phantom_location['x_range']
    nps_radius = int((xr[1]-xr[0])/2*.5)
    z = analyse_phantom_find_homogeneity(im)
    nps = calculate_ct_nnps(im=im[z,], pixel_size=d.PixelSpacing, radius=nps_radius, **kwargs)
    nps_output = analyse_nnps(nps, d.PixelSpacing[0])
    return nps_output

#%%

if __name__ == '__main__':
    
    # Example for getting the nnps analysis results using the automatic functionality
    phantom_directory = 'images/nps/ctp'
    im, d, end_points = load_ct_folder(phantom_directory)
    nnps_results = automate_ct_phantom_nnps(im,d)
    plt.plot(nnps_results['frequency'],nnps_results['radial'])
    import pandas as pd
    pd.DataFrame(nnps_results).to_excel('output.xlsx')
    
    # Example for producing the nnps then caluclating the results  
    im, d, end_points = load_ct_folder('images/nps/')
    #im = im[:,:,0]
    ct_nnps = calculate_ct_nnps(im,
                                d.PixelSpacing,
                                n_regions = 16, # Number of regions to sample NNPS from. Suggest at least 8, but more is fine.
                                radius=120,# radius in pixels from the centre of the image. Should be clearly within the homogeneity window.
                                region_side=64 # how big each region should be. Don't want to get too close to the centre or edge of the phantom
                                )
    ct_results = analyse_nnps(ct_nnps, d.PixelSpacing[0])
    
    
    d = pydicom.dcmread('images/MammoDaily1.IMA')
    region_size=128
    region_overlap=1.1
    mg_nnps = calculate_planar_nnps(d.pixel_array.copy(), 0.085, region_size, region_overlap)
    mg_results = analyse_nnps(mg_nnps, 0.085)
    
