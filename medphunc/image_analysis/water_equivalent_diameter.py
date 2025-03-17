#!/usr/bin/env python3
#
# water_equivalent_diameter
#
# Calculates water equivalent area (Aw) and water equivalent circle diameter (Dw) for
# CT DICOM images, as proposed by:
#
#     McCollough C, Bakalyar DM, Bostani M, Brady S, Boedeker K, Boone JM, Chen-Mayer HH,
#     Christianson OI, Leng S, Li B, McNitt-Gray MF. Use of water equivalent diameter for
#     calculating patient size and size-specific dose estimates (SSDE) in CT: The Report of
#     AAPM Task Group 220. AAPM report. 2014 Sep;2014:6.
#
# Requirements:
# cv2, numpy, pydicom (pip3 install opencv-python numpy pydicom)
#
# Usage:
# >>> import DICOMwaterequivalent
# >>> DICOMwaterequivalent(filename, threshold, window)
#        filename:  DICOM file,
#        threshold: ROI contour threshold level,
#        window:    Optional, view window for output image, as tuple (ww,wl). No image will
#                   be outputted if omitted.
#
#        example: DICOMwaterequivalent('in.dcm', -350, (1000,40))
#
# Returns:
# Tuple containing:
#   water equivalent area (Aw) in mm² (float),
#   water equivalent diameter (Dw) in mm (float),
#   ROI area in mm² (float),
#   ROI equivalent circle diameter in mm (a circle with ROI area) (float),
#   ROI hull area in mm² (float),
#   ROI hull equivalent circle diameter in mm (float),
#   result image displaying ROI and ROI hull contours (numpy array).
#
#   example: (24740.231323242188, 177.48307205659782, 27518.49097592727, 187.18341518945613, 25731.055450439453, 181.0022025481258, array([[[0, 0, 0], ... ]]], dtype=uint8)) )
#
# Copyright:
# Based on published software:
# 2018 Caspar Verhey (caspar@verhey.net), MIT License (see MIT licence for details LICENSE)
# 
import os
import cv2
import numpy as np
import math
import pydicom
import pandas as pd
from scipy import ndimage
from typing import List


from medphunc.misc import ssde
from medphunc.image_io import ct
from medphunc.image_analysis import image_utility as iu

import logging
logger = logging.getLogger(__name__)



#%% Class refactor
class WED:
    """
    Calculate WED from either a single image, or a CT image stack.
    
    The method from_folder() should be used for loading a folder of CT images
    
    
    """
    wed = None
    ssde = None
    wed_results = {}
    im = None
    scale = None
    threshold = None
    window = None
    region = None
    method = 'centre'
    verbose=False
    
    def __init__(self, im, scale, threshold=-300,window=False, region='body',
                 verbose=False, method='center'):
        self.im = im
        self.scale=scale
        self.threshold=threshold
        self.window=window
        self.region=region
        self.verbose=verbose
        self.method=method
        self.calculate_wed()
        self.calculate_ssde()
        

    
    @classmethod
    def from_image(cls, im, scale, threshold=-300, window=False, region='body'):
        return cls(im, scale, threshold, window, region)
        
    
    @classmethod
    def from_folder(cls, folder, threshold=-300, region=None):
        vol, dcm, end_points = ct.load_ct_folder(folder)
        return cls.from_volume(vol, dcm, threshold, region)
    
    
    @classmethod
    def from_volume(cls, volume, dcm, threshold=-300, region=None, verbose=False, method='centre'):

        if region is None:
            try:
                if 'Body' in dcm.CTDIPhantomTypeCodeSequence[0].CodeMeaning:
                    region = 'body'
                elif 'Head' in dcm.CTDIPhantomTypeCodeSequence[0].CodeMeaning:
                    region='head'
            except AttributeError:
                if dcm.get('BodyPartExamined') == 'HEAD':
                    region='head'
                else:
                    region='body'
                    logger.warning('region not found in dicom and not provided'+ 
                                   ' manually - assumed body phantom')
        elif region not in ['body', 'head']:
            raise(ValueError(f'region must be one of [body,head], {region} was passed'))
            
        
        c = cls(volume, np.prod(dcm.PixelSpacing), threshold=threshold, region=region, window=None, verbose=verbose, method=method)

        return c
    
    
    @classmethod
    def from_dicom_objects(cls, dcm_objects, threshold=-300, region=None):
        im, dcm, end_points = ct.load_ct_dicoms(dcm_objects)
        return cls.from_volume(im, dcm, threshold, region)
        
    
    def calculate_wed(self):
        if len(self.im.shape) == 3:
            self.wed_results = wed_from_volume(self.im, self.scale, self.threshold, self.window, self.verbose, self.method)
            if self.method=='full':
                self.wed_results['water_equiv_circle_diam'] = self.wed_results['mean_wed']
        else:
            self.wed_results = wed_from_image(self.im, self.scale, self.threshold, self.window)
        self.wed = self.wed_results['water_equiv_circle_diam']/10
    
    
    def calculate_ssde(self):
        self.ssde = ssde.ssde_from_wed(self.wed_results['water_equiv_circle_diam']/10, self.region).iloc[0]
                  
    def __repr__(self):
        return f'Water equivalent diameter calculations \nWED: {self.wed} cm\nSSDE: {self.ssde}'
    
    
def wed_from_volume(vol, scale, threshold=-300, window=False, verbose=False, method='centre'):
    if method=='centre':
        im = vol[vol.shape[0]//2,]
        output = wed_from_image(im, scale, threshold, window, verbose)
    elif method=='full':
        wed_results = []
        for i in range(vol.shape[0]):
            wed_results.append(wed_from_image(vol[i,], scale, threshold, window, verbose))
        weds = np.array([o['water_equiv_circle_diam'] for o in wed_results])
        output = {'median_wed': np.median(weds),'max_wed': np.max(weds), 'min_wed':np.min(weds), 'mean_wed':np.mean(weds)}
        if verbose:
            output['wed_slice_results'] = wed_results
    else:
        raise(ValueError('method argument not one of "centre","full" (provided %s)' % method))
    return output
    
    
def wed_from_image(im, scale, threshold = -300, window = False, verbose=False):
    '''
    Calculate the water equivalent diameter from a CT image.
    
    Written for axial slices.

    Parameters
    ----------
    im : numpy array
        2d axial CT slice, anatomy must be surrounded by air.
    threshold : int
        threshold value to separate air from human in image.
    scale : float
        Pixel size in image, in mm^2/pixel.
    window : tuple
        Trigger return of debugging image, set (WW, WL) in debugging image

    Returns
    -------
    output : dictionary
        dictionary containing wed information. Of particular note, 
        water_equiv_circle_diam contains the main output,
        image_overlay contains a debugging image that can be used to validate
        segmentation/contouring

    '''
    # map ww/wl for contour detection (filter_img)
    thresh = ((im > threshold)*255).astype(np.uint8)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # calculate area and equivalent circle diameter for the largest contour (assumed to be the patient without table or clothing)
    # Assume scale is in mm2
    contour = max(contours, key=lambda a: cv2.contourArea(a))
    area = cv2.contourArea(contour) * scale
    equiv_circle_diam = 2.0*math.sqrt(area/math.pi)

    hull = cv2.convexHull(contour)
    hullarea = cv2.contourArea(hull) * scale
    hullequiv = 2.0*(hullarea/math.pi)**0.5

    # create mask of largest contour
    mask_img = np.zeros((im.shape), np.uint8)
    cv2.drawContours(mask_img,[contour],0,255,-1)

    # calculate mean HU of mask area
    roi_mean_hu = cv2.mean(im, mask=mask_img)[0]

    


    # calculate water equivalent area (Aw) and water equivalent circle diameter (Dw)
    # 
    water_equiv_area = 0.001 * roi_mean_hu * area + area
    water_equiv_circle_diam = 2.0 * math.sqrt(water_equiv_area/math.pi)

    if window:
        # map ww/wl to human-viewable image (view_img)
        remap = lambda t: 255.0 * (1.0 * t - (window[1] - 0.5 * window[0])) / window[0] # create LUT function; window[0]: ww, window[1]: wl
        view_img = np.array([remap(row) for row in im]) # rescale
        view_img = np.clip(view_img, 0, 255) # limit to 8 bit
        view_img = view_img.astype(np.uint8) # set color depth
        view_img = cv2.cvtColor(view_img, cv2.COLOR_GRAY2RGB) # add RBG channels

        # create overlay to draw on human-viewable image (to be added as transparent layer)
        overlay_img = np.copy(view_img)

        # draw contour 3px wide on overlay layer, merge layers with transparency
        cv2.drawContours(overlay_img, [hull], -1, (0,255,255), 2, cv2.LINE_AA)
        cv2.drawContours(overlay_img, [contour], -1, (0,255,0), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay_img, 0.40, view_img, 1 - 0.40, 0, view_img)
        
        # add text
        cv2.putText(view_img, "patient:", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(view_img, "patient:", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.putText(view_img, "{:.0f} mm^2, circle d = {:.0f} mm".format(area,equiv_circle_diam), (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(view_img, "{:.0f} mm^2, circle d = {:.0f} mm".format(area,equiv_circle_diam), (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        cv2.putText(view_img, "water eq.:", (10,36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(view_img, "water eq.:", (10,36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.putText(view_img, "{:.0f} mm^2, circle d = {:.0f} mm".format(water_equiv_area, water_equiv_circle_diam), (100,36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(view_img, "{:.0f} mm^2, circle d = {:.0f} mm".format(water_equiv_area, water_equiv_circle_diam), (100,36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        cv2.putText(view_img, "hull:", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(view_img, "hull:", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200), 1, cv2.LINE_AA)
        cv2.putText(view_img, "{:.0f} mm^2, circle d = {:.0f} mm".format(hullarea, hullequiv), (100,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(view_img, "{:.0f} mm^2, circle d = {:.0f} mm".format(hullarea, hullequiv), (100,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200), 1, cv2.LINE_AA)
    else:
        view_img = False
       
    output = {'area':area,
              'equiv_circle_diam':equiv_circle_diam,
              'water_equiv_area':water_equiv_area,
              'water_equiv_circle_diam':water_equiv_circle_diam,
              'hull_area':hullarea,
              'hull_equiv':hullequiv
        }
    if verbose:
        output['image_overlay']:view_img
    return output

def wed_from_dicom_file(dicom_filename, threshold = -300, window = False):
    d = pydicom.dcmread(dicom_filename)
    return wed_from_dicom(d, threshold, window)


def wed_from_dicom(dicom_pydicom, threshold = -300, window = False):
    '''
    Calculate the water equivalent diameter from a CT image.
    
    Written for axial slices.

    Parameters
    ----------
    dicom_pydicom : pydicom dataset
        Must containt 2d axial CT slice, anatomy must be surrounded by air.
    threshold : int
        threshold value to separate air from human in image.
    scale : float
        Pixel size in image, in mm^2/pixel.

    Returns
    -------
    output : dictionary
        dictionary containing wed information. Of particular note, 
        water_equiv_circle_diam contains the main output,
        image_overlay contains a debugging image that can be used to validate
        segmentation/contouring

    '''

    im = dicom_pydicom.pixel_array # dicom pixel values as 2D numpy pixel array
    pydicom.pixel_data_handlers.apply_rescale(im, dicom_pydicom)

    # determine pixel area in mm²/px²
    scale = dicom_pydicom.PixelSpacing[0] * dicom_pydicom.PixelSpacing[1]
    
    return wed_from_image(im, scale, threshold, window)

def get_wed(dicom):
    '''input agnostic wrapping function that returns only WED'''
    if type(dicom) != pydicom.dataset.FileDataset:
        d = pydicom.dcmread(dicom)
    
    output = wed_from_dicom(d)
    wed = output['water_equiv_circle_diam']
    return wed


#%%

def check_localiser_orientation(d_scout):
    if d_scout.PatientOrientation[0] in ['A','P']:
        orientation = 'LAT'
        sum_axis = 1
    elif d_scout.PatientOrientation[1] in ['A','P']:
        orientation = 'LAT'
        sum_axis = 0
    elif d_scout.PatientOrientation[0] in ['L','R']:
        orientation = 'AP'
        sum_axis = 1
    elif d_scout.PatientOrientation[1] in ['L','R']:
        orientation = 'AP'
        sum_axis = 0
    else:
        raise(NotImplementedError('orientation not allowed'))
    return orientation, sum_axis


def preprocess_image(input_image):
    im = input_image.copy()

    x = np.where(im[im.shape[0]//2,:] > -1000)[0]
    y = np.where(im[:,im.shape[1]//2] > -1000)[0]
    im = im[y.min():y.max(),x.min():x.max()]
    
    return im



def Dw_eq(im, axis, cal_m, cal_c, d, n):
    Aw = (im.mean(axis=axis) * cal_m + cal_c) * d * n
    Dw = (Aw * 4 / np.pi)**0.5
    return Dw.mean()


device_lookup_table = {'default':{'AP':[1.6433, -11.622]},
                       'WSTCRKPRIMESP':{'AP':[1.366469,-6.732]},
                       'TCHB5CT1PRISM':{'AP':[1.47953, -6.2701]},
                       'TCHB5CT2PRISM':{'AP':[1.4589, 1.153]},
                       'TCHB5CT3PRIME':{'AP':[1.1887, 2.7153]},
                       'AQSCN':{'AP':[1.1189, 20.3209]},
                       'AQPRIMESCAN':{'AP':[1.29950, 3.41784]}
                       }


def measure_scout_wed(d, z_index_min=0, z_index_max=0):
    
    equipment_id = d.StationName
    if equipment_id not in device_lookup_table:
        equipment_id = 'default'
    orientation, sum_axis = check_localiser_orientation(d)
    cal_params = device_lookup_table[equipment_id][orientation]
    
    im = preprocess_image(d.pixel_array)
    if z_index_max<1:
        z_index_max = im.shape[1-sum_axis]
    if z_index_min < 0:
        z_index_min = 0
    if z_index_max > im.shape[1-sum_axis]:
        z_index_max = im.shape[1-sum_axis]
    if z_index_min == z_index_max:
        z_index_min = 0
        z_index_max = im.shape[1-sum_axis]
    if z_index_min > z_index_max:
        z_index_min, z_index_max = z_index_max, z_index_min
    im = im[z_index_min:z_index_max,:]
    Dw = Dw_eq(im, sum_axis, cal_params[0], cal_params[1], d=d.PixelSpacing[sum_axis], n=im.shape[sum_axis])
    return Dw


#%% scout_wed_calibration


def check_wonky_axial(d):
    "If it's not down-the-line, just find something else."
    ordinals = d.ImageOrientationPatient
    nominals = [1.00000, 0.00000, 0.00000, 0.00000, 1.00000, 0.00000]
    for ordinal, nominal in zip(ordinals, nominals):
        if abs(abs(ordinal)-nominal > 0.05):
            return True
            raise(ValueError('The image provided as an axial is not oriented perpendicular to the patient z plane'))


def get_localiser_z_index_of_axial_slice(d_axial: pydicom.dataset.FileDataset,
                                         d_scout: pydicom.dataset.FileDataset) -> int:
    "Check where the supplied axial dicom falls within the supplied scout dicom."
    
    orientation, sum_axis = check_localiser_orientation(d_scout)
    
    if orientation=='AP':
        pass
    elif orientation == 'LAT':
        pass
    else:
        raise(NotImplementedError('Only scouts oriented as either LAT or AP are permitted'))
    
    scout_z_start = float(d_scout.ImagePositionPatient[2])
    scout_spacing = float(d_scout.PixelSpacing[1-sum_axis])
    z_axial = float(d_axial.ImagePositionPatient[2])
    z_axial_index = int((scout_z_start-z_axial)/scout_spacing)
    
    return z_axial_index



def get_relevant_localiser(localisers: list[pydicom.dataset.FileDataset],
                           sample_axial: pydicom.dataset.FileDataset) -> dict:
    relevant_scouts = {}
    for d_scout in localisers:
        try:
            orientation, sum_axis = check_localiser_orientation(d_scout)
        except:
            print('non scout situation?')
            continue
        if d_scout.FrameOfReferenceUID == sample_axial.FrameOfReferenceUID:
            logging.debug('localiser with matching frame of reference')
            relevant_scouts[orientation] = d_scout
    else:
        if len(relevant_scouts) == 0:
            raise(ValueError('Scout matching the frame of reference of the axial volume not found'))
    return relevant_scouts


def detect_number_patient_voxels_at_fov_edge(im: np.array,
                                             threshold: int=300,
                                             out_of_fov_hu_number: int=-2048) -> int:
    
    ax_mask = im > threshold
    ax_mask = ndimage.binary_fill_holes(ax_mask)
    ax_mask = iu.find_largest_segmented_object(ax_mask)
    ax_mask = ndimage.binary_dilation(ax_mask,iterations=2)
    pt_near_fov_voxels_count = (im[ax_mask] == out_of_fov_hu_number).sum()
    return pt_near_fov_voxels_count



def wed_localiser_calibration_for_study(localisers: list[pydicom.dataset.FileDataset],
                           axials: list[pydicom.dataset.FileDataset]) -> List[dict]:
    
    localisers = load_dicom_list(localisers)
    axials = load_dicom_list(axials)
    
    
    test_axial = axials[len(axials)//2]
    check_wonky_axial(test_axial)
    
    relevant_scouts = get_relevant_localiser(localisers, test_axial)
    
    scout_Aws = {}
    for orientation, d_scout in relevant_scouts.items():
        orientation, sum_axis = check_localiser_orientation(d_scout)
        scout_im = preprocess_image(d_scout.pixel_array)
        scout_Aws[orientation] = scout_im.mean(axis=sum_axis)

        
    output = []
    for d_axial in axials:
        instance_data = {}
        
        # axial content
        pt_near_fov_voxels = detect_number_patient_voxels_at_fov_edge(d_axial.pixel_array)
        wed = water_equivalent_diameter.WED.from_dicom_objects([d_axial])
        Aw_ax =(wed.wed*10)**2*np.pi/4
        Aw_ax_norm = Aw_ax / scout_im.shape[sum_axis] / d_scout.PixelSpacing[sum_axis]
        
        #localiser data
        for orientation, d_scout in relevant_scouts.items():
            try:
                scout_index = get_localiser_z_index_of_axial_slice(d_axial, d_scout)
                instance_data['LPV_'+orientation] = scout_Aws[orientation][scout_index]
            except IndexError as e:
                print(e)
                continue
        
        # meta data
        instance_data['accession_number'] = d_axial.AccessionNumber
        instance_data['study_description'] = d_axial.StudyDescription
        instance_data['z'] = float(d_axial.ImagePositionPatient[2])
        instance_data['scout_index'] = scout_index
        instance_data['Aw_ax'] = Aw_ax
        instance_data['Aw_ax_normalised'] = Aw_ax_norm
        instance_data['near_fov_voxels'] = pt_near_fov_voxels
        instance_data['slice_location'] = float(d_axial.SliceLocation)
        output.append(instance_data)
    
    return output



def get_calibration_parameters(df: pd.DataFrame, calibration_orientation: str='AP') -> (float, float):
    m, c = np.polyfit(df['LPV_'+calibration_orientation],df.Aw_ax_normalised, 1)
    return m, c


def load_dicom_list(ds):
    out_ds = []
    for d in ds:
        if type(d) is not pydicom.dataset.FileDataset:
            out_ds.append(pydicom.dcmread(d))
        else:
            out_ds.append(d)
    return ds



def calibrate_device_scouts(localisers: list[list[os.PathLike | pydicom.dataset.FileDataset]],
                     axials:list[list[os.PathLike | pydicom.dataset.FileDataset]],
                     calibration_orientation = 'AP',
                     tight_fov_threshold=50):
    
    output = []
    for ds_scouts, ds_axials in zip(localisers,axials):
        output = output + wed_localiser_calibration_for_study(ds_scouts, ds_axials)
    
    return process_wed_calibration_results(output)
    


def process_wed_calibration_results(calibration_results, tight_fov_threshold=50, calibration_orientation = 'AP'):
    df = pd.DataFrame(calibration_results)
    df = df.loc[df.near_fov_voxels < tight_fov_threshold,:]
    m, c = get_calibration_parameters(df, calibration_orientation)
    return df, (m, c)


    

#%% PACS scripts for getting data

def wed_from_scout_via_accession_number(accession_number: str) -> float:
    from medphunc.pacs import thanks
    from medphunc.pacs import sorting
    
    t_study = thanks.Thank('study',AccessionNumber=accession_number)
    t_study.find()
    t_series = t_study.drill_down(0,find=True)
    ds_scout = sorting.get_scouts(t_series)
    ds_axial = sorting.get_first_last_axial_slices(t_series)

    relevant_localisers = get_relevant_localiser(ds_scout, ds_axial[0])
    d_scout = relevant_localisers['AP']
    
    axial_indices = [get_localiser_z_index_of_axial_slice(d, d_scout) for d in ds_axial]
    return measure_scout_wed(d_scout, axial_indices[0], axial_indices[1])



def wed_calibration_data_from_accession_number(accession_number: str) -> List[dict]:
    from medphunc.pacs import thanks
    from medphunc.pacs import sorting
    
    t_study = thanks.Thank('study',AccessionNumber=accession_number)
    t_study.find()
    t_series = t_study.drill_down(0,find=True)
    ds_scout = sorting.get_scouts(t_series)
    axial_index = sorting.get_axial_index(t_series)
    ds_axial = t_series.retrieve_or_move_and_retrieve(axial_index)[0]
    
    return wed_localiser_calibration_for_study(ds_scout, ds_axial)


#%%

if __name__ == "__main__":

    import sys
    try:
        filename = sys.argv[1]
        threshold = int(sys.argv[2])
    except:
        raise AttributeError('\n\nUsage:\n$ DICOMwaterequivalent.py filename threshold\nRead source code for details.')


    result = wed_from_dicom_file(filename, threshold, (1600,-400))

    # cv2.imwrite('out.png', result[6]) # to write numpy image as file
    print(result[0:6], flush=True)                # results[0:6] = (Aw, Dw, Ap, Dp, Aph, Dph)
    cv2.imshow('DICOMwaterequivalent', result[6]) # results[6] = numpy image, press any key in graphical window to close
    cv2.waitKey(0)
    cv2.destroyAllWindows()