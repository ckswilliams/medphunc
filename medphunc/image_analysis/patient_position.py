# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 11:30:22 2020

@author: willcx
"""

import pandas as pd
from medphunc.image_analysis import image_utility as iu
import numpy as np
from scipy.ndimage import binary_opening, binary_fill_holes
from medphunc.misc import dicom_coordinate_transform

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#%% functions

def measure_image_offset(im, d, result_detail='key', end_points=None):#, method='rad', average='mean'):
    """
    Measure the position of the patient within the gantry as an offset between
    the patient's center and the gantry isocenter. Uses several methods to
    define the patient center.

    Parameters
    ----------
    im : np.array
        numpy array containing a CT image stack.
    d : pydicom.Dataset
        dicom metadata for the above image stack. Preferably from the central slice.
    result_detail : str, optional
        How much detail to show in the output. The default is "key", other options are "full", "partial"
    end_points : patient position for the first and last slice in the stack, used to calculate slope

    Returns
    -------
    results : dict
        Dictionary containing the values calculated by the algorithm

    """
    logger.debug(f'Measuring object offset from isocenter for image stack of shape {im.shape}')
    results = {}

    # Get slice by slice patient position using three methods
    
    # method 1 full object segmented center
    logger.debug(f'Segmenting largest object')
    m = im > -300
    m = binary_opening(m, iterations=5)
    m = iu.find_largest_segmented_object(m)
    #m = binary_fill_holes(m) #Not really necessary
    
    logger.debug(f'Localising segmented object')
    center = iu.localise_segmented_object(m)['center']
    
    full_center = dicom_coordinate_transform.dcm_pixel_coordinate(center[1], center[2], d)
    results['full_y'], results['full_x'] = (full_center[1] , full_center[2])
    
    
    logger.debug(f'Calculating centers')
    # method 2 object center by slice
    centers_ij = []
    for j in range(m.shape[0]):
        mm = m[j,]
        try:
            centers_ij.append(iu.localise_segmented_object(mm)['center'])
        except ValueError:
            centers_ij.append(np.array([np.nan, np.nan]))
    
        
    imm = im.copy()
    imm[~(m>0.5)] = -1000

    logger.debug(f'Calculating radiological centers')
    rad_centers_ij = []
    for j in range(imm.shape[0]):
        try:
            rad_centers_ij.append(iu.weighted_mean_position(imm[j,]))
        except ValueError:
            rad_centers_ij.append(np.array([np.nan, np.nan]))
    
    
    centers = [dicom_coordinate_transform.dcm_pixel_coordinate(c[0],c[1],d) for c in centers_ij]
    rad_centers = [dicom_coordinate_transform.dcm_pixel_coordinate(c[0],c[1],d) for c in rad_centers_ij]
    
    centers_ij = [np.array([0, x[0],x[1]]) for x in centers_ij]
    rad_centers_ij = [np.array([0, x[0],x[1]]) for x in rad_centers_ij]
    
    
    temp_df = pd.DataFrame({'center':centers,'radiological_center':rad_centers, 'center_ij':centers_ij, 'radiological_center_ij':rad_centers_ij})
    for j, k in enumerate(['z','y','x']):
        for t in ['center','radiological_center', 'center_ij', 'radiological_center_ij']:
            temp_df[t+'_'+k] = temp_df[t].str[j]
    temp_df = temp_df.iloc[:,4:]
    
    
    
    
    if end_points is not None:
        logger.debug(f'End points supplied - applying adjustment')
        slope = end_points[1] - end_points[0]
        slope_x = slope[2]/slope[0]
        slope_y = slope[1]/slope[0]
        
        slice_thickness = float(d.SliceThickness)
        
        z_positions = np.arange(len(centers))-len(centers)//2
        z_positions = z_positions * slice_thickness
        x_adjust = z_positions * slope_x
        y_adjust = z_positions * slope_y
        temp_df.center_x = temp_df.center_x + x_adjust
        temp_df.radiological_center_x = temp_df.radiological_center_x + x_adjust
        temp_df.center_y = temp_df.center_y + y_adjust
        temp_df.radiological_center_y = temp_df.radiological_center_y + y_adjust
    

    if result_detail =='full':
        logger.debug(f'Compiling full result detail')
        results['slice_data'] = temp_df.to_dict()

    
    # Calculate some summary information
    logger.debug(f'Generating summary info')
    meds = temp_df.apply(np.nanmedian)
    means = temp_df.apply(np.nanmean)
    means.index += '_mean'
    meds.index += '_median'
    results = {**results, **means.to_dict(), **meds.to_dict()}

    
    # Calculate optimimum center position
    logger.debug(f'Calculating optimum position for center')
    pixel_spacing = np.array(d.PixelSpacing).astype(np.float64)
    
    image_position = np.array(d[('0020','0032')].value)
    yx_position = image_position[:-1][::-1]
    
    table_height = float(d.TableHeight)
    
    #Fine tuning to account for differences between manufacturers
    if d.Manufacturer == 'SIEMENS':
        logger.debug(f'Detected SIEMENS - including table height in calculations')
        yx_optimum = np.array([-table_height, 0]) #Siemens base the image position on the table height
    else:
        yx_optimum = np.array([0,0]) #otherwise set to 0
        
    #Fine tuning to account for patient position
    if d.PatientPosition in ['FFS', 'HFS']:
        logger.debug(f'Detected patient position:{d.PatientPosition},does not require adjustment')
        pass # No need to change anything for FFS (default for most scans)
    elif d.PatientPosition in ['FFP']:
        logger.debug(f'Detected patient position:{d.PatientPosition}, reversing Y direction')
        yx_optimum[0]*=-1 # For prone, we need to reverse the y direction. Do we need to deal with x? %todo
    else:
        raise(ValueError('Couldnt account for patient position of type :' + d.PatientPosition ))

    
    
    yx_center = np.array(im.shape[1:])/2
    key_results = {}
    
    for method in ['center','radiological_center','full']:
        for averaging in ['mean', 'median']:
            
        
            if method=='full':
                yx_patient = np.array([results[method+'_y'], results[method+'_x']])
            else:
                yx_patient = np.array([results[method+'_y'+'_'+averaging], results[method+'_x'+'_'+averaging]])
            yx_patient_offset_mm = pixel_spacing * (yx_center - yx_patient)
                
            yx_delta = yx_patient - yx_optimum
            
            key_results[method+'_'+averaging+'_position_offset_y'] = yx_delta[0]
            key_results[method+'_'+averaging+'_position_offset_x'] = yx_delta[1]


    if result_detail != 'key':
        key_results = {**key_results, **results}

    return key_results



#%%






