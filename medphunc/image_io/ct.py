# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:21:41 2020

@author: WILLCX
"""


import pydicom
import numpy as np
import sys
import glob


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
    print('glob: {}'.format(folder))
    for fname in glob.glob(folder+'*', recursive=False):
        print("loading: {}".format(fname))
        files.append(pydicom.read_file(fname))
    
    print("file count: {}".format(len(files)))
    
    # skip files with no SliceLocation (eg scout views)
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, 'InstanceNumber'):
            slices.append(f)
        else:
            skipcount = skipcount + 1
    
    print("skipped, no SliceLocation: {}".format(skipcount))
    
    # ensure they are in the correct order
    slices = sorted(slices, key=lambda s: (s.SeriesInstanceUID, s.SliceThickness, int(s.InstanceNumber)))
    
    
    
    # pixel aspects, assuming all slices are the same
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness
    ax_aspect = ps[1]/ps[0]
    sag_aspect = ps[1]/ss
    cor_aspect = ss/ps[0]
    
    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)
    
    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        if img2d.shape == img3d.shape[:2]:
            img3d[:, :, i] = img2d
        else:
            print('found file with wrong pixel array size, skipping')
    
    #Rearrange so that axis order is [z,y,x]
    img3d = np.moveaxis(img3d, 2, 0)
    
    #Rescale the metadata according to the intercept
    img3d = rescale_ct_image(img3d, files[0].RescaleSlope, files[0].RescaleIntercept)
    
    p0 = np.array(slices[0].ImagePositionPatient)
    p1 = np.array(slices[-1].ImagePositionPatient)
    
    if return_tags == 'middle':
        return_d = slices[len(slices)//2]
    elif return_tags == 'first':
        return_d = slices[0]
    elif return_tags == 'last':
        return_d = slices[-1]       
    else:
        raise(ValueError("return_tag not in ['middle', 'first','last']"))
    
    # Return the 3d array, and a dicom file for extracting metadata
    return img3d, return_d, (p0[::-1], p1[::-1])

if __name__ == '__main__':
    folder = sys.argv[0]
    load_ct_folder(folder)