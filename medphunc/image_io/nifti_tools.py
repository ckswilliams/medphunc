# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:21:41 2020

@author: WILLCX
"""


import dicom2nifti
#import glob
from pathlib import Path
import os
import logging

def find_directory_endpoints(base_dir):
    """Walk through the directory, returnging only folders that contain files
    but don't contain folders
    """
    return [Path(folder[0]) for folder in os.walk(base_dir) if (folder[1] == []) and (folder[2] != [])]


def convert_dicom_directory_to_nifti(input_dir, output_dir):
    """
    Process all the subdirectories containing dicom files into nifti files

    Parameters
    ----------
    input_dir : str or path
        folder containing all the files that need to be processed.
    output_dir : str or path
        folder where the nifti files will be saved.

    Returns
    -------
    None.

    """
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok = True)
    folder_list = find_directory_endpoints(input_dir)
    
    
    
    for folder in folder_list:
        logging.warning(f'Converting to niftii: {folder}')
        output_file = folder.parts[len(input_dir.parts):]
        output_file = output_dir / Path('/'.join(output_file))
        output_file = output_file.with_suffix('.nii.gz')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        logging.warning(f'Created directory: {output_file.parent}')
        
        try:
            dicom2nifti.dicom_series_to_nifti(folder, output_file, reorient_nifti=True)
        except Exception as e:
            logging.warning(f'Could not convert contents of {folder} to nifti, because: {e}\nThis may be acceptable behaviour')
            continue
    

if __name__ == '__main__':
    input_dir = 'C:/shared/dicomdump'
    output_dir = 'C:/shared/niftidump'
    
    convert_dicom_directory_to_nifti(input_dir, output_dir)