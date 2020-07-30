# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:52:49 2020

@author: willcx
"""

import pydicom
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


#%%
def decompress_file_list(fns):
    for fn in fns:
        try:
            d = pydicom.read_file(fn.as_posix())
        except:
            logger.debug(f'Could not open: {fn}')
            continue
        if d.file_meta.TransferSyntaxUID != '1.2.840.10008.1.2.1':
            try:
                d.decompress()
            except:
                logger.warning(f'Opened but could not decompress: {fn}, with transfer syntax: {d.file_meta.TransferSyntaxUID}')
                continue
            logger.debug(f'Found and decompressed: {fn}')
            d.save_as(fn.as_posix())
        else:
            logger.debug(f'Found already uncompressed file: {fn}')
            


def decompress_folder(base_dir):
    fns = Path(base_dir).glob('**/*')
    decompress_file_list(fns)
    
#%%

if __name__ == '__main__':
    base_dir = 'C:/shared/dicomdump'
    decompress_folder(base_dir)
