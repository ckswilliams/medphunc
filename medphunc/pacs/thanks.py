# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 12:03:12 2021

@author: WILLCX
"""

import pyorthanc

from pydicom.filebase import DicomBytesIO
import pydicom

from typing import Type, List, Tuple
import numpy as np
import os
from medphunc.image_io import ct
import ssl

from medphunc.pacs import pacsify as pi


# %%

class Orthancs(pyorthanc.Orthanc):

    def __init__(self, orthanc_info):
        url = orthanc_info['url']
        
        context = False
        if 'verify' in orthanc_info:
            context = ssl.create_default_context(cafile=orthanc_info['verify'])
        if 'cert' in orthanc_info:
            # Load the .crt and .key files
            if not context:
                context = ssl.create_default_context()
            context.load_cert_chain(certfile=orthanc_info['cert'][0],
                                    keyfile=orthanc_info['cert'][1])
        super().__init__(url, verify=context)

    def set_orthanc_url(self, url):
        self._orthanc_url = url


orthanc_config = pi.NETWORK_INFO['orthanc']
default_orthanc = os.environ.get('MEDPHUNC-ORTHANCDEFAULT')
if not default_orthanc:
    default_orthanc = orthanc_config['default']
orthanc_info = orthanc_config[default_orthanc]

orthanc = Orthancs(orthanc_info)
if 'username' in orthanc_info:
    orthanc.setup_credentials(orthanc_info['username'], orthanc_info['password'])
orthanc.timeout=5 # Retain default value. Change manually if longer timeout is required.


# %%

def retrieve_orthanc_instance(instance_oid: str) -> pydicom.Dataset:
    """
    Retrieve a single instance from Orthanc

    Parameters
    ----------
    instance_oid : str
        A string corresponding to the id for an instance in Orthanc.

    Returns
    -------
    pydicom.Dataset containing the requested dicom data.
    
    """
    return pydicom.dcmread(DicomBytesIO(orthanc.get_instances_id_file(instance_oid)))


def retrieve_orthanc_series(series_oid: str) -> List[pydicom.Dataset]:
    instance_oids = orthanc.get_series_id(series_oid)['Instances']
    return [retrieve_orthanc_instance(oid) for oid in instance_oids]


def retrieve_orthanc_volume_data(
        orthanc_instance_ids: List[str]
) -> Tuple[Type[np.ndarray], pydicom.Dataset, List[float]]:
    """
    Retrieve all instances in a volume image set in a list from Orthanc, returning 
    the volume and a single dicom dataset to provide metadata.

    Parameters
    ----------
    orthanc_instance_ids : List[str]
        list of orthnac instance ids.

    Returns
    -------
    Tuple[np.typing.ArrayLike, pydicom.Dataset, List[float]]
        tuple containing the requested image volume and metadata

    """
    dcms = [retrieve_orthanc_instance(i) for i in orthanc_instance_ids]
    return ct.load_ct_dicoms(dcms)


def check_series_status(series_uid):
    series_oids = orthanc.post_tools_find({"Level": 'Series',
                                  'Query': {'SeriesInstanceUID': series_uid}})
    if len(series_oids) == 1:
        return True
    elif len(series_oids) == 0:
        return False
    else:
        raise (ValueError('Multiple series found corresponding to single series uid %s', series_uid))


def retrieve_series(series_uid):
    """
    Directly retrieve an entire series from Orthanc from the series uid

    Parameters
    ----------
    series_uid : TYPE
        DESCRIPTION.

    Returns
    -------
    Tuple[np.typing.ArrayLike, pydicom.Dataset, List[float]]
        Tuple containing the requested image volume and metadata

    """
    orthanc_instance_ids = orthanc.post_tools_find({"Level": 'Instance',
                                           'Query': {'SeriesInstanceUID': series_uid}})

    if len(orthanc_instance_ids) > 1:
        im, d, meta = retrieve_orthanc_volume_data(orthanc_instance_ids)
    elif len(orthanc_instance_ids) == 1:
        d = retrieve_orthanc_instance(orthanc_instance_ids[0])
        im = d.pixel_array
        meta = {}
    else:
        raise (ValueError("Requested series has no instances"))

    return im, d, meta


def retrieve_sop_instance(sop_uid):
    orthanc_instance_ids = orthanc.post_tools_find({"Level": 'Instance',
                                           'Query': {'SOPInstanceUID': sop_uid}})
    if len(orthanc_instance_ids)==0:
        raise(ValueError('Requested SOP instance is not available in orthanc'))
    d = retrieve_orthanc_instance(orthanc_instance_ids[0])
    try:
        im = d.pixel_array
    except AttributeError:
        im = None
    meta = {}
    return im, d, meta


class Thank(pi.RDSR):

    def query_orthanc(self, qrl = None, **kwargs):
        if qrl is None:
            qrl = self.QueryRetrieveLevel.title()
        else:
            if qrl not in ['STUDY','SERIES','INSTANCE','IMAGE']:
                raise(ValueError('qrl not one of STUDY, SERIES, or INSTANCE/IMAGE'))
        if qrl == 'Image':
            qrl = 'Instance'
        query = {}
        for s in self:
            if s.keyword == 'QueryRetrieveLevel':
                continue
            if s.is_empty:
                continue
            if type(s.value) is pydicom.multival.MultiValue:
                value = '\\'.join(s.value)
            else:
                value = s.value
            query[s.keyword] = value
        query = {**query, **kwargs}

        return orthanc.post_tools_find({'Level': qrl, 'Query': query})


    def retrieve_oids(self, orthanc_ids):
        if self.QueryRetrieveLevel == 'IMAGE':
            return [retrieve_orthanc_instance(oid) for oid in orthanc_ids]

        if self.QueryRetrieveLevel == 'SERIES':
            output = []
            for series_oid in orthanc_ids:
                output.append(retrieve_orthanc_series(series_oid))
            return output

        if self.QueryRetrieveLevel == 'STUDY':
            output = []
            for study_oid in orthanc_ids:
                study_info = orthanc.get_studies_id(study_oid)
                series_oids = study_info['Series']
                output.append([retrieve_orthanc_series(series_oid) for series_oid in series_oids])
            return output
            # raise(NotImplementedError('Not yet available on a study searchset'))


    def retrieve_all_data(self):
        orthanc_ids = self.query_orthanc()
        return self.retrieve_oids(orthanc_ids)

    
    def retrieve(self, retrieve_index=None):
        retrieve_object = self.copy()
        if retrieve_index is not None:
            if self.query_level == 'study':
                retrieve_object.StudyInstanceUID = self.result.StudyInstanceUID[retrieve_index]
            if self.query_level == 'series':
                retrieve_object.SeriesInstanceUID = self.result.SeriesInstanceUID[retrieve_index]
            if self.query_level == 'instance':
                retrieve_object.SOPInstanceUID = self.result.SOPInstanceUID[retrieve_index]
        for key in self.keys():
            if retrieve_object[key].value == '':
                retrieve_object.pop(key)
        orthanc_ids = retrieve_object.query_orthanc()
        return self.retrieve_oids(orthanc_ids)
    
                

    def retrieve_one_instance_all_series(self):
        sop_uids = self.move_one_instance_all_series()
        if type(sop_uids) is str:
            sop_uids = [sop_uids]
        sop_uids = list(np.array(sop_uids).flat)
        return [retrieve_sop_instance(suid) for suid in sop_uids]
    
    
    def retrieve_rdsrs(self):
        instance_oids = self.query_orthanc(qrl='INSTANCE', SOPClassUID = '1.2.840.10008.5.1.4.1.1.88.67')
        return [retrieve_orthanc_instance(oid) for oid in instance_oids]
    
    
    def retrieve_or_move_and_retrieve(self, retrieve_index=None):
        """
        Method for reducing PACS load by assuming that data is in the accessible
        orthanc already, and only looking in PACS if orthanc is empty.
        
        However, it will cause issues when used in a situation where the orthanc 
        contains only some of the data you want.
        
        Could be refined so that if searching on the series or instance level
        """
        ds = self.retrieve(retrieve_index)
        if len(ds) == 0:
            self.move(retrieve_index)
            ds = self.retrieve(retrieve_index)
        return ds
    
   