# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:12:35 2020

@author: willcx
"""


from matplotlib import pyplot as plt
import pandas as pd
import pathlib
import nibabel
import imageio
import numpy as np
import glob
from scipy.stats import mode
import logging
from medphunc.image_analysis import image_utility as iu
from medphunc.image_io import export_tools

logger = logging.getLogger(__name__)

#%%

def _get_im_fns(im_folder):
    im_fns = glob.glob(im_folder+'/*.gz')
    return im_fns

def _get_seg_fns(im_fns, seg_folder):
    return [seg_folder+'/'+s.split('\\')[-1][:-12]+'.nii.gz' for s in im_fns]

def _get_all_fns(im_folder, seg_folder=None, output_folder=None):
    im_fns = _get_im_fns(im_folder)
    if seg_folder is None:
        seg_folder = str(pathlib.Path(im_folder) / 'labels')
    seg_fns = _get_seg_fns(im_fns, seg_folder)
    if output_folder is None:
        output_folder = str(pathlib.Path(im_folder) / 'overlays')
    output_fns = _get_seg_fns(im_fns, output_folder)
    output_fns = [fn[:-7]+'_overlay'+'.nii.gz' for fn in output_fns]
    return im_fns, seg_fns, output_fns


#%% Create images showing segmented regions, save to the cwd
    
def save_segmentation_overlays(im_fns, seg_fns, output_fns, seg_value=666):
    """
    Draw segmented region onto images and save to disk

    Parameters
    ----------
    im_fns : list
        list of filenames pointing to image files.
    seg_fns : list
        list of filenames pointing to segmentations, in the same order as im_fns.
    output_fns : list
        list of filenames to save, in the same order as the other inputs.
        Should have the .nii or .nii.gz suffix, and will override any other
    seg_value : int
        pixel value that the segmented regions will be set to

    Returns
    -------
    None.

    """
    try:
        for i, (ifn, sfn, ofn) in enumerate(zip(im_fns, seg_fns, output_fns)):
        
            n = nibabel.load(ifn)
            im = n.get_fdata().copy()
            seg = nibabel.load(sfn).get_fdata()
            if ofn[-3:] not in ['.gz','nii']:
                ofn = ofn+'.nii'
            im[seg>0.5] = seg_value
            out = nibabel.Nifti1Image(im, n.affine, n.header)
            nibabel.save(out, ofn)
            logger.debug('Saved image at {ifn} and segmentation {sfn} as {ofn}')
    except FileNotFoundError:
        logger.error(f'Failed to parse {ifn}, {sfn}, giving up')
        pass


def save_segmentation_contour_overlays(im_fns, seg_fns, output_fns):
    """
    Write outline of contours to images and save to disk

    Parameters
    ----------
    im_fns : list
        list of filenames pointing to image files.
    seg_fns : list
        list of filenames pointing to segmentations, in the same order as im_fns.
    output_fns : list
        list of filenames to save, in the same order as the other inputs.
        Should have the .tif suffix, and will override any other.

    Returns
    -------
    None.

    """
    try:
        for i, (ifn, sfn, ofn) in enumerate(zip(im_fns, seg_fns, output_fns)):
        
            n = nibabel.load(ifn)
            im = n.get_fdata().copy()
            seg = nibabel.load(sfn).get_fdata()
            if ofn[-4:] not in ['.tif']:
                ofn = ofn+'.tif'
            iu.apply_window(im, (300,80), unit_range=True)
            out = iu.draw_object_contours_3d(im, seg)
            imageio.volwrite(ofn, out)
            #export_tools.save_image_stack(out, ofn)
            logger.debug('Saved contour overlays as {ofn}')
    except FileNotFoundError:
        logger.error(f'Failed to parse {ifn}, {sfn}, giving up')
        pass  


#%% Show all images 
def show_all_slices(im):
    
    for i in range(im.shape[2]):
        plt.imshow(im[:,:,i])
        plt.show()
#show_all_slices(pat)

#%% Extract pixel info from segmented regions
    
def extract_segmented_pixel_data(im_fns, seg_fns):
    """
    Extract values from images and returns as a dataframe

    Parameters
    ----------
    im_fns : list
        list of image filenames. filenames must be 'path/patientXXXXX_otherinfo.nii(.gz)
    seg_fns : TYPE
        list of segmentation filenames. filenames must be 'path/patientXXXXX_otherinfo.nii(.gz)

    Returns
    -------
    dataframe
        median, mean, mode, and ID for all supplied images.

    """
    try:
        results = []
        for i, (pfn, sfn) in enumerate(zip(im_fns, seg_fns)):
            try:
                npat = nibabel.load(pfn)
                pat = npat.get_fdata().copy()
                seg = nibabel.load(sfn).get_fdata()
                
                vals = pat[seg>0.5]
                output = {}
                output['PatientID'] = pfn.split('patient')[-1].split('_')[0]
                output['seg_med'] = np.median(vals)
                output['seg_mean'] = np.mean(vals)
                output['seg_mode'] = mode(vals)[0][0]
                results.append(output)
            except Exception as e:
                output['error'] = str(e)
                continue
                
    except FileNotFoundError:
        pass
    return pd.DataFrame(results)


#%%
    
def folder_to_overlays(im_folder, seg_folder=None, output_folder=None):
    im_fns, seg_fns, output_fns = _get_all_fns(im_folder, seg_folder, output_folder)
    save_segmentation_overlays(im_fns, seg_fns, output_fns)
    

def folder_to_segmented_pixel_data(im_folder, seg_folder):
    im_fns, seg_fns, __ = _get_all_fns(im_folder, seg_folder)
    df = extract_segmented_pixel_data(im_fns, seg_fns)
    return df

#%%



#%%

if __name__ == '__main__':
    #im_folder= 'C:/shared/Projects/nnUNet/nnunet/nnUNet_base/nnUNet_raw/liverseg/imagesTs'
    #seg_folder='C:/shared/Projects/nnUNet/nnunet/nnUNet_base/nnUNet_raw/liverseg/imagesTs/labelled'
    #output_folder = 'C:/shared/Projects/nnUNet/nnunet/nnUNet_base/nnUNet_raw/liverseg/imagesTs'
    
    im_folder= 'C:/shared/niftidump/kub/ax'
    seg_folder='C:/shared/niftidump/kub/ax/labels'
    output_folder = None
    
