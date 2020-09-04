# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 13:20:40 2020

@author: willcx
"""

import numpy as np


def construct_dicom_coordinate_transform(pixel_spacing,
                                         image_orientation,
                                         image_position):
    
    M = np.zeros((4,4))
    M[:3,0] = np.array(image_orientation[3:])*pixel_spacing[0]
    M[:3,1] = np.array(image_orientation[:3])*pixel_spacing[1]
    M[:3,3] = np.array(image_position)
    M[3,3] = 1
    return M

def calculate_pixel_coordinate(row, col,
                               pixel_spacing,
                               image_orientation,
                               image_position
                               ):
    
    M = construct_dicom_coordinate_transform(pixel_spacing,
                                         image_orientation,
                                         image_position)
    
    pp = np.array([row, col, 0, 1])
    P = np.dot(M, pp)
    return P[:3][::-1]

def dcm_pixel_coordinate(row,col,dcm):
    
    return calculate_pixel_coordinate(row, col,
                               dcm.PixelSpacing, dcm.ImageOrientationPatient,
                               dcm.ImagePositionPatient)
        

def slice_translate(row, col, pixel_spacing, slice_thickness, 
                    image_orientation, image_position):
    pass
    
    
def dcm_slice_translate(row, col, pixel_spacing, slice_thickness, 
                    image_orientation, image_position):
    pass    



#%%

if __name__ == '__main__':

    pixel_spacing = df.meta[9].PixelSpacing
    image_orientation = df.meta[9].ImageOrientationPatient
    image_position = df.meta[9].ImagePositionPatient
    
    
    pplr = calculate_pixel_coordinate(512,723,pixel_spacing,
                                             image_orientation,
                                             image_position)
    
    ppul = calculate_pixel_coordinate(0,0,pixel_spacing,
                                             image_orientation,
                                             image_position)
    
    
