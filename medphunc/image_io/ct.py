# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:21:41 2020

@author: WILLCX
"""


import pydicom
import numpy as np
import sys
import glob
from collections import Counter


import logging
logger = logging.getLogger(__name__)


def rescale_ct_image(im, rescale_slope, rescale_intercept):
    "Rescale a ct image by applying the rescale intecept and slope"
    
    #if type(im) != np.array:
    #    raise ValueError('Must supply an image in the form of a numpy array')
    output = im.astype(float)*float(rescale_slope) + float(rescale_intercept)
    return output


def rescale_ct_dicom(d):
    "Convert a pydicom CT dataset to an array, applying the rescale intercept"
    if type(d) != pydicom.dataset.FileDataset:
        raise ValueError('Must supply a pydicom dataset')
    output = rescale_ct_image(d.pixel_array, d.RescaleSlope, d.RescaleIntercept)
    return output


def load_ct_folder(folder, return_tags='middle'):
    """
    

    Parameters
    ----------
    folder : string
        string describing the location of a folder.
    return_tags : string, optional, ['first', 'last', 'middle']
        which dicom file to return with the image structure. Default is the central

    Returns
    -------
    img3d : np.array
        3d array containing CT data.
    d : pydicom.Dataset 
        details of one file, for metadata

    """
    
    # load the DICOM files
    files = []
    logger.info('glob: {}'.format(folder))
    for fname in glob.glob(folder+'*', recursive=False):
        logger.debug("loading: {}".format(fname))
        files.append(pydicom.read_file(fname))
    
    logger.info("file count: {}".format(len(files)))
    
    # skip files with no SliceLocation (eg scout views)
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, 'InstanceNumber') and hasattr(f, 'PixelData'):
            slices.append(f)
        else:
            skipcount = skipcount + 1
    
    if len(slices) == 0:
        raise(FileNotFoundError('No dicom image files found in '+folder))
    
    logger.info("skipped, no SliceLocation: {}".format(skipcount))
    
  
    #get most frequent pixel array shape for all slices
    img_shapes = [s.pixel_array.shape for s in slices]
    c = Counter(img_shapes)
    img_2d_shape = c.most_common()[0][0]
      
    #Drop anything  with a different array size to most common
    slices = [s for s in slices if s.pixel_array.shape==img_2d_shape]
    
    # ensure they are in the correct order
    slices = sorted(slices, key=lambda s: (s.SeriesInstanceUID, s.SliceThickness, int(s.InstanceNumber)))

    
    # pixel aspects, assuming all slices are the same
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness
    ax_aspect = ps[1]/ps[0]
    sag_aspect = ps[1]/ss
    cor_aspect = ss/ps[0]
    

    
    # create 3D array
    img_shape = (len(slices), *img_2d_shape)
    img3d = np.zeros(img_shape)
    
    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        if img2d.shape == img_2d_shape:
            img3d[i, :, :] = img2d
        else:
            logger.debug('found file with wrong pixel array size, skipping')
        
    #Rescale the metadata according to the intercept
    img3d = rescale_ct_image(img3d, files[0].RescaleSlope, files[0].RescaleIntercept)
        
    if return_tags == 'middle':
        return_d = slices[len(slices)//2]
    elif return_tags == 'first':
        return_d = slices[0]
    elif return_tags == 'last':
        return_d = slices[-1]       
    else:
        raise(ValueError("return_tag not in ['middle', 'first','last']"))
    
    #Get the end points 
    start_point = np.array(slices[0].ImagePositionPatient)[::-1]
    end_point = np.array(slices[-1].ImagePositionPatient)[::-1]
    
    
    # Return the 3d array, and a dicom file for extracting metadata
    return img3d, return_d, (start_point, end_point)

if __name__ == '__main__':
    folder = sys.argv[0]
    load_ct_folder(folder)