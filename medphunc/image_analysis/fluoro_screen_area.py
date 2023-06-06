# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:18:41 2023

@author: chris williams
"""


import imageio

from skimage import measure
#from medphunc.image_analysis import image_utility as iu
from matplotlib import pyplot as plt
import numpy as np
import cv2
import pathlib

FRAME_BUFFER = 5
PIXEL_SCALE_MM = 0.5
THRESHOLD = None


#%% copy/paste from medphunc.image_analysis
def get_contours(m):
    m = m.copy()
    m[m>0]=255
    m = m.astype(np.uint8)
    contours, hierarchy = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # calculate area and equivalent circle diameter for the largest contour (assumed to be the patient without table or clothing)

    return contours

def draw_object_contours(draw, seg, color=(0,255,0), line_weight=2):
    """
    Draw the outlines of the supplied segmentation on the supplied drawable array

    Parameters
    ----------
    draw : np.array (np.uint8)
        drawable array of same size as seg. 2d.
    seg : np.array (boolean)
        Array of segmentation data. 2d.
    color : TYPE, tuple
        RGB color definition. The default is (0,255,0).
    line_weight : TYPE, int
        line thickness. The default is 2.

    Returns
    -------
    np.array containing the drawn data

    """
    c = get_contours(seg)
    draw = cv2.drawContours(draw, c,-1,color, line_weight)
    return draw

#%%


def plot_image(arr, plot_dir, plot_name):
    "Plot and maybe save an array. Only saves if plot_dir is not None"
    fig, ax = plt.subplots()
    ax.imshow(arr)
    ax.axis('off')
    ax.set_title(plot_name)
    if plot_dir is not None:
        plot_dir = pathlib.Path(plot_dir)
        fig.savefig(plot_dir/plot_name)
    fig.show()


def image_background_comparison(video_file:str,
        background_time:float,
        fluoro_time:float,
        frame_buffer: int=FRAME_BUFFER,
        pixel_scale:float=PIXEL_SCALE_MM,
        threshold:float=THRESHOLD,
        save_figure_dir: str=None):
    
    im = imageio.get_reader(video_file)
    
    fps = float(im.get_meta_data()['fps'])
    
    background_index = fps*background_time
    fluoro_index = fps*fluoro_time
    
    fluoro_frames = range(int(fluoro_index-frame_buffer),int(fluoro_index+frame_buffer))
    fluoro_image = np.array([im.get_data(i) for i in fluoro_frames]).max(axis=0).mean(axis=2)
    
    
    background_image = im.get_data(int(background_index))
    
    
    if threshold is None:
        #estimate the threshold
        threshold = (fluoro_image.max()-fluoro_image.mean())/2 + fluoro_image.mean()
    
    larr, numregions = measure.label(fluoro_image>threshold, return_num=True)
    if numregions>1:
        
        print('The number of segmented regions exceeded 1, the image might not be suitable')
        plot_image(larr, save_figure_dir, 'fluoro_segmentation.png')
        plot_image(background_image, save_figure_dir, 'fluoro_background.png')
        input('Review the plotted images and try changing the fluoro time and background time if they look wrong. Enter to continue')
        raise(ValueError('The number of segmented regions exceeded 1, the image might not be suitable'))
    
    # show the segmented region
    plot_image(larr, save_figure_dir, 'fluoro_segmentation.png')

    plot_image(background_image, save_figure_dir, 'fluoro_background.png')

    
    region=measure.regionprops(larr)[0]
    
    contour_overlay = draw_object_contours(background_image, larr)
    
    
    area_text = f'Area: {region.area*PIXEL_SCALE_MM**2} mm^2'
    print(area_text)
    width_text = f'Width: {region.image.shape[1]*PIXEL_SCALE_MM} mm'
    print(width_text)
    height_text = f'Height: {region.image.shape[0]*PIXEL_SCALE_MM} mm'
    print(height_text)
    
    
    fig, ax = plt.subplots()
    
    ax.imshow(contour_overlay)
    ax.annotate(('\n').join([area_text, width_text, height_text]), xy=(0.01,0), xycoords='axes fraction', color='blue', va='bottom',ha='left')
    ax.axis('off')
    
    fig.show()
    if save_figure_dir is not None:
        fig.savefig(save_figure_dir+'fluoro_contour_overlay.png')
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('video_filename')
    parser.add_argument('fluoro_time',type=float, help='Time in s that the exposure occurs')
    parser.add_argument('background_time', type=float, help='Time in s that the background image is displayed')
    parser.add_argument('--pixel_scale', default=PIXEL_SCALE_MM, type=float, help='Pixel length. Script assumes square pixels')
    parser.add_argument('--threshold',default=None, type=float, help='Manual pixel value threshold.')
    parser.add_argument('--frame_buffer', default=5, type=int, )
    parser.add_argument('--save_figure_directory', default=None)
    args = parser.parse_args()

    
    image_background_comparison(video_file=args.video_filename,
            background_time=args.background_time,
            fluoro_time=float(args.fluoro_time),
            frame_buffer=int(args.frame_buffer),
            pixel_scale=float(args.pixel_scale),
            threshold=args.threshold,
            save_figure_dir=args.save_figure_directory)
    
    input('Press enter to close')
    
    