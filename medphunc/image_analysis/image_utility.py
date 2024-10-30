# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:30:48 2020

Selection of general functions for interacting with diagnostic images, including 
segmentation, filtration, roi, statistics and others.
Might need to be refactored into multiple modules in the future

@author: Chris Williams
"""
import numpy as np

from scipy.ndimage import binary_fill_holes, morphology, binary_opening, binary_closing, binary_erosion, binary_dilation
from scipy.signal import wiener
from scipy import ndimage
from scipy import spatial
from matplotlib import pyplot as plt
import scipy
from skimage import measure
from medphunc.image_io import ct

import cv2

from medphunc.image_analysis import radial_data

import logging

#%% logging
log_verbosity = 1
def log_plot(x, y, priority = 2, **kwargs):
    if priority <= log_verbosity:
        fig, ax = plt.subplots()
        ax.plot(x,y, **kwargs)
        fig.show()


#%% Segmentation functions

def get_contours(m):
    m = m.copy()
    m[m>0]=255
    m = m.astype(np.uint8)
    contours, hierarchy = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # calculate area and equivalent circle diameter for the largest contour (assumed to be the patient without table or clothing)

    return contours

def find_largest_segmented_object_contour(m):
    """Find the largest segmented contour from a mask. 2d only."""
    contours = get_contours(m)
    contour = max(contours, key=lambda a: cv2.contourArea(a))
    return contour

def find_largest_segmented_object(m):
    """ Find the largest segmented object from a mask. Works in 3d."""
    t, n = measure.label(m, return_num=True)
    if m.sum() == 0:
        raise(ValueError('Cannot segment empty mask'))
    if n > 1:
        vals, counts = np.unique(t.flatten(), return_counts=True)
        i = counts[1:].argsort()[-1] # Get index for the most common object, excluding background
        t = t==vals[i+1] # Make a mask where the measured label is equal to the most common object
    return t


def draw_object_contours(draw, seg, color=(0,255,0), line_weight=2):
    """
    Draw the outlines of the supplied segmentation on the supplied drawable array

    Parameters
    ----------
    draw : np.array (np.uint8)
        drawable array of same size as seg. 2d.
    seg : np.array (boolean)
        Array of segmentation data. 2d.
    color : TYPE, tuple
        RGB color definition. The default is (0,255,0).
    line_weight : TYPE, int
        line thickness. The default is 2.

    Returns
    -------
    np.array containing the drawn data

    """
    c = get_contours(seg)
    draw = cv2.drawContours(draw, c,-1,color, line_weight)
    return draw
    

        

def draw_object_contours_3d(im, seg, window_level = (300,80), **kwargs):
    """
    Draw the outlines of the supplied segmentation on the supplied 3d array

    Parameters
    ----------
    draw : np.array (np.uint8)
        drawable array of same size as seg. 3d.
    seg : np.array (boolean)
        Array of segmentation data. 3d.
    color : TYPE, tuple
        RGB color definition. The default is (0,255,0).
    line_weight : TYPE, int
        line thickness. The default is 2.

    Returns
    -------
    np.array containing the drawn data

    """
    seg = find_largest_segmented_object(seg)
    drawstack = np.zeros((*im.shape, 3)).astype(np.uint8)
    for k in range(im.shape[2]):
        draw = drawstack[:,:,k,:].copy()
        s = seg[:,:,k]
        try:
            
            draw = draw_object_contours(draw, s, **kwargs)
        except Exception as e:
            print(e)
        drawstack[:,:,k,:] = draw
    
    cim = im.reshape((*im.shape,1))
    cim = np.repeat(cim,3,axis=3)
    cim = apply_window(cim, window_level, unit_range=True)
    drawmask = drawstack.sum(axis=3) > 0.5
    drawmask = drawmask[:,:,:,np.newaxis]
    drawmask = drawmask.repeat(3,axis=3)
    tcim = cim.copy()
    tcim[drawmask] = 0
    tcim = tcim + drawstack/255
    
    return tcim

def localise_segmented_object(seg):
    """
    Describe the location of the segmented object provided.

    Parameters
    ----------
    seg : np.array
        2d or 3d segmented object.

    Returns
    -------
    output : dict
        dictionary containing positional information about the segmented object
        keys: center, anterior_pint, y_range, x_range
            
    """
    Z = np.array(np.where(seg))
    
    mins = Z.min(axis=1)
    maxs = Z.max(axis=1)
    center = (maxs + mins) // 2
    
    #Silly Sorcery to get the most anterior point
    ZZ = Z[:,Z[0,:]==Z[0,0]]
    anterior_point = ZZ.min(1)
    
    output = {'center':center,
              'anterior_point':anterior_point
            }
    
    dims = ['z','y','x']
    if mins.shape[0] == 2:
        dims.pop(0)
        
    for i, dim in enumerate(dims):
        output[dim+'_range'] = np.array([mins[i],maxs[i]])
    return output



def weighted_mean_position(weighted_seg, intercept=-1000):
    """
    Generates the weighted center of the object image provided, assuming that
    pixel value is proportional to density.

    Parameters
    ----------
    weighted_seg : np.array
        2d image array, with any non-interesting information set to zero.
    intercept : numeric
        intercept for the linear pixel values. i.e. -1000 for CT images in HU

    Returns
    -------
    means : np.array
        index of the mean location within the image.

    """
    im = (weighted_seg - intercept).copy()
    
    #create a meshgrid
    ranges = [np.arange(im.shape[i]) for i in range(len(im.shape))]
    meshes = np.meshgrid(*ranges, indexing ='ij')
    
    # Calculate the weighted mean position of the segmented object, assuming
    # linear, increasing relationship between pixel value and density
    means = [(im * mesh).sum()/im.sum() for mesh in meshes]
    
    return np.array(means)


def cv_roi_to_index(r):
    return [int(r[1]), int(r[1]+r[3]), int(r[0]), int(r[0]+r[2])]

def apply_cv_roi(im, r):
    i = cv_roi_to_index(r)
    if len(im.shape) == 3:
        return im[:, i[0]:i[1], i[2]:i[3]].copy()
    
    return im[i[0]:i[1], i[2]:i[3]].copy()


def segment_to_rectangle(m, step_size=3):
    """Somewhat awful, moderately inefficient function for fitting a rectangle
    to the inside of a mask region. But it's the best we have.
    """
    
    ivals = [0,
             0,
             0,
             0]
    set_ivals = [False, False, False, False]
    
    def opt_func(m, ivals):
        return (~m)[ivals[0]:m.shape[0]-ivals[1],ivals[2]:m.shape[1]-ivals[3]].sum()
    
    # Loop through minimising the optimisation value by moving the rectangle in
    # on each side by step size, until there are no outside values left
    optval = opt_func(m, ivals)
    while optval > 0:
        for i, val in enumerate(ivals):
            if set_ivals[i] == True:
                continue
            ivals[i]+=step_size
            oldval = optval
            optval = opt_func(m,ivals)
            if oldval==optval:
                set_ivals[i] = True
                print(f'set {i}')
        print(optval)
        
    rm = np.zeros(m.shape)
    rm[ivals[0]:m.shape[0]-ivals[1],ivals[2]:m.shape[1]-ivals[3]] = True
    return rm


#%% CT functions


def localise_phantom(im, threshold = None):
    """
    Show the position of a phantom in an image. Originally written for 
    circular phantoms. Works in 3d, expects dim order z, y, x
    """
    if not threshold:
        threshold = im.mean()
    seg = im > threshold
    seg = binary_fill_holes(seg)
    
    seg = find_largest_segmented_object(seg)
    
    output = localise_segmented_object(seg)
    center = output['center']
    
    #Create a debug image
    logging=False
    if logging:
        for x in output['x_range']:
            seg[:,x-2:x+2] = 1
        for y in output['y_range']:
            seg[y-2:y+2,:] = 1
        seg[center[0]-20:center[0]+20,center[1]-5:center[1]+5] = 0
        seg[center[0]-5:center[0]+5,center[1]-20:center[1]+20] = 0
        
        #log.image(log.Level.INFO, seg*.99)
    
    return output

def ctshow(im, ax=None):
    "Quick, dirty function for showing a CT plot, designed to 'usually' look ok"
    tim = im[im>-100]
    window = tim.max()-tim.min()
    level = tim.min() + window
    tim = apply_window(im, (window,level))
    if ax:
        ax.imshow(tim)
        return ax
    else:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(tim)
        fig.show()

#%% Planar functions

def localise_homogenous_region(im):
    homo_region = im[im.shape[0]//2-8:im.shape[0]//2+8,im.shape[1]//2-8:im.shape[1]//2+8]
    homo_mean = homo_region.mean()
    homo_std = homo_region.std()
    m = (np.abs((im - homo_mean) / homo_std) < 5)
    m=binary_closing(m,iterations=3)
    m = segment_to_rectangle(m)
    return localise_segmented_object(m)
    
    
  

#%% Radial functions
    
#Calculate the radial average for a 2d image
def calculate_radial_average(im, pixel_size):
    """
    Calculates the radial average for a 2d image.
    x and y pixel spacing must be equal.

    Parameters
    ----------
    im : np.array
        2d image. x and y pixel spacing should be equal.
    pixel_size : float
        x / y pixel spacing.

    Returns
    -------
    frequency : np.array
        1d array with frequency values.
    rad_nps : np.array
        1d array with nps values.

    """
    #compute the radial average for the data
    aggregated_radial_data = radial_data.radial_data(im,rmax=64)
    frequency = aggregated_radial_data.r * pixel_size / 10
    radial_mean = aggregated_radial_data.mean
    frequency = frequency[radial_mean==radial_mean]
    radial_mean = radial_mean[radial_mean==radial_mean]

    return frequency, radial_mean


#%% Window functions

def apply_window(im, window_level=None,
                 unit_range=True):
    """
    Apply a windowing function to an image array.

    Parameters
    ----------
    im : np.array
        any numpy array, but an image is preferred.
    window_level : str or tuple, optional
        Set the window width and window centre in two ints. The default is False.
        Useful values for CT may be (300,120).
        
    unit_range : boolean, optional
        scale down to a range from 0 to 1. The default is True.

    Returns
    -------
    im : np.array
        a nice windowed image array.

    """
    window_lookup = {
   # head and neck
        'brain': (80, 40),
        'subdural': (200, 75),
        'stroke': (8, 32),
        'stroke wide': (40, 40),
        'temporal bones': (2800, 600),
        'soft tissues': (400, 30),
   # chest
        'lungs': (1500, -600),
        'mediastinum': (350, 50),
   # abdomen
        'soft tissues': (400, 50),
        'liver': (150, 30),
   # spine
        'spine soft tissues': (250, 50),
        'spine bone': (1800, 400)
        }
    
    if window_level is None:
        level = (im.max() + im.min())/2
        window = (im.max() - im.min())/2
    elif type(window_level) is str:
        window_level = window_lookup[window_level]
        window = window_level[0]
        level = window_level[1]
    else:
        window = window_level[0]
        level = window_level[1]
    im = (im - level)/window
    im[im < -1] = -1
    im[im > 1] = 1
    if unit_range:
        im = (im+1) / 2
    else:
        im = im*window + level
    return im


def detrend_region(region): #TODO
    "Function for removing an overarching trend in a region. Not properly implemented."
    #flat_region = scipy.ndimage.median_filter(region, footprint=2)
    #region = region-flat_region
    region = region - region.mean()
    return region

#%% ROI functions
    

def mask_point(im, buffer=2):
    """Create a mask around the highest pixel value in an image, with a buffer"""
    mask = np.zeros(im.shape)
    mask = im==im.max()
    mask = morphology.binary_dilation(mask, iterations = buffer)
    return mask


# Return all integers within val_range of i
def get_nearby_integers(i, val_range = 1.5):
    vals = np.arange(np.floor(i - val_range), np.ceil(i + val_range + 1)).astype(int)
    return vals


def extract_patch_around_point(im, point, roi_size):
    """
    Extract a rectangular patch centered on a point. Note definitions are y,x
    as per numpy indexing

    Parameters
    ----------
    im : numpy array
        image to be subset.
    coords : tuple
        y, x index around which to extract a patch from the image.
    roi_size : tuple
        Height, width of the point.

    Returns
    -------
    extracted patch as a numpy array.

    """
    y = point[0]
    x = point[1]
    minx = max(x-roi_size[1]//2, 0)
    maxx = min(x+roi_size[1]//2, im.shape[1])
    miny = max(y-roi_size[0]//2, 0)
    maxy = min(y+roi_size[0]//2, im.shape[0])
    patch = im[miny:maxy, minx:maxx]
    return patch

def Rslice(rect):
    y = slice(rect[0], rect[0] + rect[2])
    x = slice(rect[1], rect[1] + rect[3])
    return x, y


def shift_columns_by_subpixel(ROI, pixel_shifts):
    """
    Shift all the pixels in ROI, by an amount set in pixel_shifts

    Parameters
    ----------
    ROI : np.array
        ROI of pixels that need to be shifted.
    pixel_shifts : array of floats
        array explaining how much each column of ROI needs to be shifted.

    Returns
    -------
    ROI : np.array
        shifted pixels.

    """
    ROI = ROI.copy()
    for i in range(ROI.shape[1]):
        ROI[:,i] = scipy.ndimage.interpolation.shift(ROI[:,i], pixel_shifts[i])
    return ROI


def interp_first_index(X, value, search_from='center'):

    Z = np.where(np.convolve([1,1,1],X > value)==2)[0]
    
    if search_from=='center':
        x = np.abs(Z-X.shape[0]/2).argmin()
    elif search_from=='top':
        x = Z.argmin()
    elif search_from=='bottom':
        x = Z.argmax()
    else:
        raise ValueError(f'search_from must be in "center", "top", "bottom", was {search_from}')
    i = Z[x]
    XX = X[i-4:i+2]
    out_index = np.interp(value, XX, range(len(XX)))
    if np.interp(i+out_index-4, range(len(X)),X)/value > 1.1:
        logging.warning(f'Failed to align a line profile')
    return i+out_index-4

def align_line_profiles(ROI, midpoint=None):
    """
    Align a set of line profiles arranged in columns, centering at the half-max

    Parameters
    ----------
    ROI : np.array
        ROI containing line profiles.

    Returns
    -------
    output : np.array
        ROI with profiles aligned at the half-maximum.

    """
    if not midpoint:
        midpoint = (ROI.max() + ROI.min())/2
    
    smoothed = np.apply_along_axis(wiener, 0, ROI)
    indexes = np.apply_along_axis(interp_first_index, 0, smoothed, midpoint)
    shifts = ROI.shape[0]//2 - indexes
    output = shift_columns_by_subpixel(ROI, shifts)
    #log.image(log.Level.TRACE, apply_window(output))
    return output


def interactive_select_roi(im, window_level=None):
    if window_level is not None:
        im = apply_window(im, window_level)
    
    r = cv2.selectROI(im)
    roi = apply_cv_roi(im, r)
    return roi


#%% MTF functions

def plot_mtf(MTF_results, plot_name = ''):
    """
    Plot MTF, show the 50th and 10th ntile

    Parameters
    ----------
    MTF_results : dictionary
        Dictionary of dictionaries. Each subdictionary contains two keys, 'MTF'
        and 'frequency'
    plot_name : string, optional
        Title for the plot.

    Returns
    -------
    fig : pyplot figure
        Figure showing the resolution and cutoffs.

    """
    
    fig, ax = plt.subplots()
    ax.set_xlabel('lp/mm')
    ax.set_title(plot_name + ' MTF')
    
    colors = ['black', 'blue', 'red', 'green']
    for k, v in MTF_results.items():
        ax.plot(v['frequency'], v['MTF'], label = k+' MTF')
        res = calculate_resolution_from_MTF(v['frequency'], v['MTF'])
        print(k+' Resolution: ')
        print(res)
        color = colors.pop(0)
        for i, (kk, vv) in enumerate(res.items()):
            x1, y1 = vv, float(kk.split('\\')[0])/100
            x2, y2 = vv, 0
            if i == 0:
                label_value = k+' resolution cutoffs'
            else:
                label_value = None
            ax.plot([x1, x2], [y1, y2], color=color, marker = None,
                    linestyle='dashed', label=label_value)

    ax.legend()
    fig.show()
    return fig

def mtf_from_lsf(lsf, axis=-1):
    """
    Take the FT of a LSF to generate an MTF
    
    Args:
        lsf

    Returns:
        One dimensional FFT, or array of 1d FFTs
    """
    mtf = np.abs(np.fft.fft(lsf, axis=axis))
    mtf = np.split(mtf, 2, axis=axis)
    return mtf

def mtf_from_esf(esf, axis=-1):
    '''
    Convert an Edge Spread Function (or series of ESFs to an MTF(s))
    
    
    '''
    
    lsf = np.gradient(esf, axis=axis)
    mtf = mtf_from_lsf(lsf, axis=axis)[0]
    return mtf


def profile_line(image, start_coords, end_coords, *,
                 spacing=1, order=0, endpoint=True):
    coords = []
    n_points = int(np.ceil(spatial.distance.euclidean(start_coords, end_coords)
                           / spacing))
    for s, e in zip(start_coords, end_coords):
        coords.append(np.linspace(s, e, n_points, endpoint=endpoint))
    profile = ndimage.map_coordinates(image, coords, order=order)
    return profile


def spatial_frequency_for_mtf(n, spacings):
    """
    Return the corresponding spatial frequencies for an MTF

    Parameters
    ----------
    n : int
        Length of the original line sample.
    spacings : np.array or float
        spacings in the original line samples.

    Returns
    -------
    np.array
        Frequency array, in unit lp/<whatever unit spacing is in>.

    """
    try:
        spac_iter = iter(spacings)
    except TypeError:
        spacings = [spacings]
    
    freqs = [scipy.fft.fftfreq(n, spacing)[:n//2] for spacing in spacings]
    return np.vstack(freqs).squeeze()

def get_values_around_cutoff(x, y, cutoff):
    """Return 2 arrays of 2 numbers in x and y, the corresponding to 
    before and after y falls below the cutoff value"""
    
    try:
        i = np.where(y < cutoff)[0][0]
    except:
        print('MTF does not go below cutoff values')
        i = -1
    return x[i-1:i+1], y[i-1:i+1]

def interpolate_array_at_first_cutoff(x, y, cutoff):
    x, y = get_values_around_cutoff(x, y, cutoff)
    val = np.interp(cutoff, y[::-1], x[::-1])
    return val

def calculate_resolution_from_MTF(f, MTF):
    #Note: cannot use interpolation easily, since the MTF can go up and down
    #Ideally, this solution would use some form of interpolation though
    output = {}
    output['50\%'] = interpolate_array_at_first_cutoff(f, MTF, 0.5)
    output['10\%'] = interpolate_array_at_first_cutoff(f, MTF, 0.1)
    return output

#%% Statistic functions
    
def window_variance(img, window_length):
    img = img.astype(float)
    wmean, wsqrmean = (cv2.boxFilter(x, -1, (window_length, window_length),
    borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
    var = wsqrmean - wmean*wmean
    var[var<0] = 0
    return var

def window_std(img, window_length):
    return np.sqrt(window_variance(img, window_length))


