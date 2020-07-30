# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:11:25 2020

@author: willcx
"""

from medphunc.image_analysis import image_utility as iu
from medphunc.image_io import ct
import numpy as np



def calculate_width_largest_peak(X):
    
    X = X.copy()
    
    peak = X.max()
    background = X.min()
    threshold = (peak + background) / 2
    
    mm = X > threshold

    inner_index = np.where(np.convolve(mm, [1,1,1],'same') == 2)[0]

    lefty = interpolate_at_index(threshold, X, inner_index[0])
    righty = interpolate_at_index(threshold, X, inner_index[1])

    return righty - lefty


def find_fwhm_in_dicom(d, roi_def=None):
    if d.Modality == 'CT':
        window_level = (300,120)
        im = ct.rescale_ct_dicom(d)
    else:
        window_level = None
        im = d.pixel_array
    if roi_def==None:
        roi = iu.interactive_select_roi(im, window_level)
    else:
        roi = iu.apply_cv_roi(im, roi_def)
    fwhm = np.array(calculate_fwhm(roi))
    fwhm = fwhm * np.array(d.PixelSpacing)
    return fwhm


def select_roi_calculate_fwhm(im, window_level = (300,120)):
    roi = iu.interactive_select_roi(im, window_level)
    return calculate_fwhm(roi)


def calculate_fwhm(roi):
    m = roi > ((roi.max()-roi.min()) / 2 + roi.min())
    
    y,x = np.where(m)
    
    Y = roi[:,x.min():x.max()].max(axis=1)
    X = roi[y.min():y.max(),:].max(axis=0)

    X = calculate_width_largest_peak(X)
    Y = calculate_width_largest_peak(Y)

    return Y, X


def calculate_catphan_slice_thickness(d, roi_def = None):
    out = find_fwhm_in_dicom(d, roi_def)
    i = np.argmax(out)
    
    slope_ratio = np.tan(23/180*np.pi)
    
    return out[i] * slope_ratio


#%%

def interpolate_at_index(threshold, X, i, direction=None):
    if direction=='left':
        j = i - 1
    elif direction=='right':
        j = 1 + 1
    else:
        if X[i-1] < threshold:
            j = i-1
        elif X[i+1] < threshold:
            j=i+1
        else:
            raise ValueError(f'Array does not cross threshold ({threshold}) around supplied index ({X[i-1:i+2]})')
    ij = [i,j]
    XX = X[[i,j]]
    ij.sort()
    XX.sort()
    
    return np.interp(threshold, XX, ij)


#%%
    
output = []
for dfn in dfns:
    d = pydicom.read_file(dfn)
    result = {}
    try:
        result['slice_thickness'] = calculate_catphan_slice_thickness(d, roi)
    except:
        result['error'] = 'truely'
    result['nominal_slice_thickness'] = float(d.SliceThickness)
    result['series'] = d.SeriesDescription
    result['series_uid'] = d.SeriesInstanceUID
    result['kvp'] = d.KVP
    result['num'] = int(d.InstanceNumber)
    result['fn'] = dfn
    output.append(result)
    

test = pd.DataFrame(output)
test = test[test.error!=test.error]
test = test[test.slice_thickness/test.nominal_slice_thickness < 5]

#px.box(x=test.nominal_slice_thickness, y=test.slice_thickness/test.nominal_slice_thickness)
#px.box(x=test.nominal_slice_thickness, y=test.slice_thickness)


ttest = test[test.nominal_slice_thickness == 0.625]


#%%


dfnss = glob.glob(r'C:\shared\dicomdump\PHYSICS_TEST - PHYSICS_TEST_RPCT2_2020\20200213 - none\120 catphan 0_625mm\*')


output = []
for dfn in dfnss:
    d = pydicom.read_file(dfn)
    result = {}
    try:
        result['slice_thickness'] = calculate_catphan_slice_thickness(d, r)
    except:
        result['error'] = 'truely'
    result['nominal_slice_thickness'] = float(d.SliceThickness)
    result['series'] = d.SeriesDescription
    result['series_uid'] = d.SeriesInstanceUID
    result['kvp'] = d.KVP
    result['num'] = int(d.InstanceNumber)
    result['fn'] = dfn
    output.append(result)
    

test = pd.DataFrame(output)
test = test[test.error!=test.error]
test = test[test.slice_thickness/test.nominal_slice_thickness < 5]





test = test[test.nominal_slice_thickness == 0.625]

test = test[(test.num < 145) & (test.num > 100)]


plt.plot(test.num, test.slice_thickness/test.nominal_slice_thickness,'.')

px.box(x=test.nominal_slice_thickness, y=test.slice_thickness/test.nominal_slice_thickness)
px.box(x=test.nominal_slice_thickness, y=test.slice_thickness)

