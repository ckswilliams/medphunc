# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:35:58 2020

@author: willcx
"""

import os
import cv2
import numpy as np
import imageio

def save_image_stack(im, output_folder = 'temp'):
    os.makedirs(output_folder, exist_ok = True)
    for k in range(im.shape[2]):
        cv2.imwrite(f'{output_folder}/overlay_{str(k).zfill(4)}.png', im[:,:,k,]*255)


def flexible_save_image_stack(im, file_prefix='overlay', output_folder = 'temp',
                              im_format = 'tif',
                              im_type = np.int16):
    os.makedirs(output_folder, exist_ok = True)
    #im = im.astype(im_type)
    
    for k in range(im.shape[2]):
        imageio.imsave(f'{output_folder}/{file_prefix}_{str(k).zfill(4)}.{im_format}', im[:,:,k,])