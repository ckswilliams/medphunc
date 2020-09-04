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

import cv2
import numpy as np
import math
import pydicom
from medphunc.misc import ssde
from medphunc.image_io import ct

import logging
logger = logging.getLogger(__name__)



#%% Class refactor
class wed:
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
    
    def __init__(self, im, scale, threshold=-300, window=False, region='body'):
        self.im = im
        self.scale=scale
        self.threshold=threshold
        self.window=window
        self.region=region
        self.calculate_wed()
        self.calculate_ssde()

    
    @classmethod
    def from_image(cls, im, scale, threshold=-300, window=False, region='body'):
        return cls(im, scale, threshold, window, region)
        
    @classmethod
    def from_folder(cls, folder, threshold=-300, region=None):
        vol, dcm, end_points = ct.load_ct_folder(folder)
        im = vol[vol.shape[0]//2,]
        dcm = dcm
        scale = dcm.PixelSpacing[0]*dcm.PixelSpacing[1]
        threshold = threshold
        try:
            window = (dcm.WindowWidth[0], dcm.WindowCenter[0])
            
        except TypeError:
            window = (80,300)
        
        if region is None:
            try:
                if 'Body' in dcm.CTDIPhantomTypeCodeSequence[0].CodeMeaning:
                    region = 'body'
                elif 'Head' in dcm.CTDIPhantomTypeCodeSequence[0].CodeMeaning:
                    region='head'
            except AttributeError:
                if dcm.BodyPartExamined == 'HEAD':
                    region='head'
                else:
                    region='body'
                    logger.warning('region not found in dicom and not provided manually - assumed body phantom')
        elif region not in ['body', 'head']:
            raise(ValueError(f'region must be one of [body,head], {region} was passed'))
        c = cls(im, scale, threshold, window, region)
        c.dcm = dcm
        return c
    
    
    def calculate_wed(self):
        self.wed_results = wed_from_image(self.im, self.scale, self.threshold, self.window)
        self.wed = self.wed_results['water_equiv_circle_diam']/10
    
    
    def calculate_ssde(self):
        self.ssde = ssde.ssde_from_wed(self.wed_results['water_equiv_circle_diam']/10, self.region)
                  
    def __repr__(self):
        return f'Water equivalent diameter calculations \nWED: {self.wed} cm\nSSDE: {self.ssde.iloc[0]}'
    
def wed_from_image(im, scale, threshold = -300, window = False):
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
              'hull_equiv':hullequiv,
              'image_overlay':view_img
        }
    return output

def wed_from_dicom_file(dicom_filename, threshold = -300, window = False):
    d = pydicom.read_file(dicom_filename)
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
    im = im - 1000.0 # remap scale 0:... to HU -1000:...

    # determine pixel area in mm²/px²
    scale = dicom_pydicom.PixelSpacing[0] * dicom_pydicom.PixelSpacing[1]
    
    return wed_from_image(im, scale, threshold, window)

def get_wed(dicom):
    '''input agnostic wrapping function that returns only WED'''
    if type(dicom) != pydicom.dataset.FileDataset:
        d = pydicom.read_file(dicom)
    
    output = wed_from_dicom(d)
    wed = output['water_equiv_circle_diam']
    return wed

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