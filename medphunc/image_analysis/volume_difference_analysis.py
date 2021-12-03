# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 11:16:29 2021

@author: willcx
"""


from medphunc.image_analysis import image_utility as iu
from medphunc.image_io import ct
import numpy as np
import pandas as pd
import os

#from medphunc.parsers import generic_dict_parser
from medphunc.parsers import parse_dicom
import imageio

import cv2

from matplotlib import pyplot as plt

import pathlib


#%%


def load_dataset(folder):
    """
    Load a folder 

    Parameters
    ----------
    folder : TYPE
        DESCRIPTION.

    Returns
    -------
    output : TYPE
        DESCRIPTION.

    """
    folder = pathlib.Path(folder)
    series_folders = folder.glob('*/')
    
    output = []
    
    for series_folder in series_folders:
        im, d, endpoints = ct.load_ct_folder(series_folder)
        series_data = {}
        series_data['im'] = im
        series_data['d'] = d
        series_data['endpoints'] = endpoints
        series_data['folder_name'] = str(series_folder)
        series_data['series_name'] = series_folder.name
        output.append(series_data)
        
    return output
    

def process_dataset(data, baseline_search_term = 'baseline'):
    
    baseline_index = [baseline_search_term in d['series_name'].lower() for d in data].index(True)
    for baseline_index, d in enumerate(data):
        if baseline_search_term.lower() in d['series_name'].lower():
            break
    else:
        raise(FileNotFoundError('The requested baseline search term was not found in any of the series'))

    baseline_im = data[baseline_index]['im']
    for i in range(len(data)):
        data[i]['difference_map'] = data[i]['im'] - baseline_im
        data[i]['difference_map_mean'] = data[i]['difference_map'].mean()
        data[i]['difference_map_abs_mean'] = np.abs(data[i]['difference_map']).mean()
        data[i]['dicom_metadata'] = parse_dicom.extract_metadata(data[i]['d'])
    #return data


def extract_roi_data(data, roi=None, z_index=None):
    if z_index is None:
        z_index=int(input('Input an int to extract ROI data from -> '))
    if roi is None:
        roi = cv2.selectROI(data[0]['im'][z_index,])
        print(roi)
        
    for i in range(len(data)):
        data[i]['im_roi_std'] = iu.apply_cv_roi(data[i]['im'][z_index,], roi).std()
    #return data
    
    #noisevals = [iu.apply_cv_roi(dat['im'][62,], roi).std() for dat in data]    



def plot_difference_map(im, title=None):
    fig, ax = plt.subplots()
    i = ax.imshow(im, cmap='bwr', vmin = -100, vmax = 100)
    fig.colorbar(i, ax=ax)
    ax.set_title(title)
    return fig, ax


def plot_differences(data, z_specify = None, z_number=None, save_folder = None):
    for i in range(len(data)):
        s = data[i]['difference_map']
        z_length = data[i]['im'].shape[0]
        if z_specify is not None:
            if type(z_specify) is int:
                z_specify = [z_specify]
            for z in z_specify:
                fig, ax = plot_difference_map(s[z], title=f'kernel:{data[i]["series_name"]}, z:{z_specify}')
                if save_folder is not None:
                    fig.savefig(save_folder+f'/kern{data[i]["series_name"]}_z{z_specify}.png')
        if z_number is not None:
            for j in range(1,z_length,z_length//z_number):
                fig, ax = plot_difference_map(s[j], title=f'kernel:{data[i]["series_name"]}, z:{j}')
                if save_folder is not None:
                    fig.savefig(save_folder+f'/kern{data[i]["series_name"]}_z{j}.png')
        
def save_differences(data, save_folder):
    for i in range(len(data)):
        imageio.volwrite(f'{save_folder}/{data[i]["series_name"]}_diffmap.tif', data[i]['difference_map'])



def compile_results(data, omit_complex=True):
    results = pd.DataFrame.from_records(data)
    results = pd.concat([results, pd.DataFrame(results['dicom_metadata'].tolist())], axis=1)
    
    if omit_complex:
        complex_columns = ['im','d','difference_map', 'dicom_metadata']
        results.drop(columns=complex_columns).to_clipboard()
    return results

def volume_difference_analysis(folder, baseline_search_term='fbp',
                        output_folder = None, roi = None, z_roi = None,
                        number_plots = 16,
                        save_differences=False):
    """
    Look through a folder containing reconstructions. Select one series as a baseline,
    based on the folder names. Compare all series to this baseline, 
    showing a variety of plots and images as desired.

    Parameters
    ----------
    folder : TYPE
        Folder containing all reconstructions to be compared.
    baseline_search_term : str, optional
        How to identify the series which is the baseline. Defaults to 'FBP'
    output_folder : TYPE, optional
        Output folder. If not set, will only output to the console. The default is None.
    roi : TYPE, tuple(int,int,int,int)
        CV ROI. If not set, will create a pop up. The default is None. If not set, skip noise extraction.
    z_roi : TYPE, int
        z level at which to extract . The default is None.
    number_plots : int, optional
        How many plots to create for each series. The default is 16.
    save_plots : boolean, optional
        Whether to save plots. The default is True.
    save_differences : boolean, optional
        Whether to save difference maps. The default is True.

    Returns
    -------
    data : pd.DataFrame
        Compliation of all the results.

    """
    data = load_dataset(folder)
    process_dataset(data, baseline_search_term)
    if roi:
        extract_roi_data(data, roi, z_roi)
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
    plot_differences(data, z_roi, number_plots, output_folder)
    if save_differences:
        save_differences(data, output_folder)
    
    return data, compile_results(data)
    
    

#%%

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visual inspection and data extraction for multiple reconstructions of one dataset')
    parser.add_argument('--folder', type=str, help='Folder containing a folder for each recon.')
    parser.add_argument('--z_roi', type=int, help = 'z_level for the ROI. Blank will require input.')
    parser.add_argument('--roi', type=int, nargs=4,
                        help='OpenCV ROI definition. Not specifying will create a pop-up requiring selection')
    
    parser.add_argument('--save_difference_folder', type=str, help='If specified, difference maps will be saved to this folder')
    parser.add_argument('--save_plot_folder', type=str, help='If specified, plots will be saved to this folder')
    args = parser.parse_args()

    data = load_dataset(args.folder)
    process_dataset(data)
    extract_roi_data(data, args.roi, args.z_roi)
    
    if args.save_plot_folder:
        plot_differences(data, save_folder = args.save_plot_folder)
    if args.save_difference_folder:
        save_differences(data, args.save_difference_folder)
    