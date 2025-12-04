# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:52:49 2020

@author: willcx
"""

import pydicom
import logging
from pathlib import Path
import os
import copy
import typing

logger = logging.getLogger(__name__)


#%%


def decompress_file(fn):
    fn = Path(fn)
    d = pydicom.dcmread(fn.as_posix())
    if d.file_meta.TransferSyntaxUID != '1.2.840.10008.1.2.1':
        try:
            d.decompress()
        except:
            raise(ValueError(f'Opened but could not decompress: {fn}, with transfer syntax: {d.file_meta.TransferSyntaxUID}'))
    else:
        raise(ValueError('File found was already decompressed'))
    logger.debug('Decompressed file. Saving.')
    d.save_as(fn.as_posix())


def decompress_file_list(fns):
    for fn in fns:
        try:
            decompress_file(fn)
        except (ValueError, FileNotFoundError, PermissionError) as e:
            logger.warning(e)


def decompress_folder(base_dir):
    fns = Path(base_dir).glob('**/*')
    decompress_file_list(fns)


#%% unpack_tomography

def unpack_tomography(fn: typing.Union[os.PathLike, str], specify_frames: typing.Optional[typing.Iterable]) -> None:
    """
    Unpack a single-dicom tomography file into a directory of individual slices.
    Such a directory can be opened in Image J.
    Primarily designed for mammography tomo images that need to be unpacked into slices for measuring MTF.
    Automatically creates a directory in the same folder as the original dicom file and puts the slices into it.

    Parameters
    ----------
    fn : typing.Union[os.PathLike, str]
        filename for a tomography dicom file.
    specify_frames : typing.Optional[typing.Iterable]
        If you don't want all the slices, include the specific slices that you want to extract here.
        Don't use the python 0 index, use 1 for the first slice etc. to align with what the 
        typical situation where slice 1 is 1 mm above the breast support, etc.


    """
    fn = Path(fn)
    d = pydicom.dcmread(fn.as_posix())
    if len(d.pixel_array.shape) < 3:
        raise(ValueError('The supplied dicom file has only 2 dimensions and is therefore not an cannot be unpacked into slices'))


    output_dir = fn.parent / (fn.name + '_slices')
    output_dir.mkdir(exist_ok=True)
    
    if specify_frames is None:
        frames = range(1,int(d.NumberOfFrames)+1)
    else:
        frames = [frame-1 for frame in specify_frames]
        
    out_d = copy.deepcopy(d)
    out_d.PixelData = None
    out_d.NumberOfFrames = '1'
    
    for frame in frames:
        out_d.PerFrameFunctionalGroupsSequence = [d.PerFrameFunctionalGroupsSequence[frame]]
        out_d.PixelData = d.pixel_array[[frame],]
        try:
            out_d.PixelSpacing = out_d.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
        except AttributeError:
            out_d.PixelSpacing = out_d.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
        out_d.save_as(output_dir / f'{fn.name}_{str(frame+1).zfill(3)}.dcm')



#%%

if __name__ == '__main__':
    base_dir = 'C:/shared/dicomdump'
    decompress_folder(base_dir)
