# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 11:30:22 2020

@author: willcx
"""

import pandas as pd
from medphunc.image_analysis import image_utility as iu
import numpy as np
from scipy.ndimage import binary_opening, binary_fill_holes


#%% functions

def measure_image_offset(im, d, result_detail='key'):#, method='rad', average='mean'):
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

    Returns
    -------
    results : dict
        Dictionary containing the values calculated by the algorithm

    """
    
    results = {}

    # Get slice by slice patient position using three methods
    
    # method 1 full object segmented center
    
    m = im > -300
    m = binary_opening(m, iterations=5)
    m = iu.find_largest_segmented_object(m)
    #m = binary_fill_holes(m) #Not really necessary
    
    
    center = iu.localise_segmented_object(m)['center']
    results['full_y'] = center[1]
    results['full_x'] = center[2]
    
    
    # method 2 object center by slice
    centers = []
    for j in range(m.shape[0]):
        mm = m[j,]
        try:
            centers.append(iu.localise_segmented_object(mm)['center'])
        except ValueError:
            centers.append(np.nan)

        
    imm = im.copy()
    imm[~m] = -1000

    rad_centers = []
    for j in range(imm.shape[0]):
        try:
            rad_centers.append(iu.weighted_mean_position(imm[j,]))
        except ValueError:
            rad_centers.append(np.nan)
    
    
    temp_df = pd.DataFrame({'center':centers,'radiological_center':rad_centers})
    for j, k in enumerate(['y','x']):
        for t in ['center','radiological_center']:
            temp_df[t+'_'+k] = temp_df[t].str[j]
    temp_df = temp_df.iloc[:,2:]

    if result_detail =='full':
        results['slice_data'] = temp_df.to_dict()
        
    # Calculate some summary information
    
    means = temp_df.apply(np.nanmedian)
    meds = temp_df.apply(np.nanmean)
    means.index += '_mean'
    meds.index += '_median'
    results = {**results, **means.to_dict(), **meds.to_dict()}

    pixel_spacing = np.array(d.PixelSpacing).astype(np.float64)    
    
    image_position = np.array(d[('0020','0032')].value)
    yx_position = image_position[:-1][::-1]
    
    table_height = float(d.TableHeight)
    
    #Fine tuning to account for differences between manufacturers
    if d.Manufacturer == 'SIEMENS':
        yx_optimum = np.array([-table_height, 0]) #Siemens base the image position on the table height
    else:
        yx_optimum = np.array([0,0]) #otherwise set to 0
        
    #Fine tuning to account for patient position
    if d.PatientPosition in ['FFS', 'HFS']:
        pass # No need to change anything for FFS (default for most scans)
    elif d.PatientPosition in ['FFP']:
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
        
            yx_size = np.array(d.pixel_array.shape).astype(np.float64)
                
            yx_delta = yx_position + pixel_spacing*yx_size/2 - yx_optimum - yx_patient_offset_mm
            
            key_results[method+'_'+averaging+'_position_offset_y'] = yx_delta[0]
            key_results[method+'_'+averaging+'_position_offset_x'] = yx_delta[1]


    if result_detail != 'key':
        key_results = {**key_results, **results}

    return key_results



#%%






