# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:55:20 2020

@author: WILLCX
"""

# -*- coding: utf-8 -*-
"""
Query Script
Set of functions and tools for querying PACS
Default settings for QH included.
"""

from pydicom import Dataset
import pydicom
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian, ExplicitVRBigEndian
from pynetdicom.sop_class import XRayRadiationDoseSRStorage, DigitalXRayImageStorageForPresentation
from pynetdicom import AE, StoragePresentationContexts
from pynetdicom import QueryRetrievePresentationContexts
from pynetdicom import VerificationPresentationContexts
from pynetdicom import StoragePresentationContexts

import datetime
import pandas as pd
import json
import pathlib
import copy

import os

import logging
logger = logging.getLogger(__name__)

# %%
import pynetdicom

if pynetdicom.__version__ >= '1.5':
    # from pynetdicom.sop_class import PatientRootQueryRetrieveInformationModelFind
    from pynetdicom.sop_class import PatientRootQueryRetrieveInformationModelMove
    from pynetdicom.sop_class import StudyRootQueryRetrieveInformationModelFind
    from pynetdicom.sop_class import StudyRootQueryRetrieveInformationModelMove
else:
    PatientRootQueryRetrieveInformationModelFind = 'P'
    PatientRootQueryRetrieveInformationModelMove = 'P'
    StudyRootQueryRetrieveInformationModelFind = 'S'
    StudyRootQueryRetrieveInformationModelMove = 'S'

# %%

# Load the PACS config
pacsconfig_path = os.environ.get('MEDPHUNC-PACSCONFIG')
if not pacsconfig_path:
    logger.warning("MEDPHUNC-PACSCONFIG environment variable not set - loading aeinfo.json from package directory")
    pacsconfig_path = pathlib.Path(__file__).parent.absolute() / 'aeinfo.json'
with open(pacsconfig_path, 'r') as f:
    NETWORK_INFO = json.load(f)


# %% AE wrapper class
class AEInfo:
    aet = None
    address = None
    port = None

    def __init__(self, name=None, default=None,
                 aet='DEFAULT', address='127.0.0.1', port=104):
        if default:
            self.set_from_default(default)
        elif name:
            self.set_from_saved(name)
        else:
            self.set_ae_info(aet, address, port)

    def __str__(self):
        return f'AET:{self.aet}\nAddress:{self.address}\nPort:{self.port}'

    def __repr__(self):
        return repr(f'AET:{self.aet}, Address:{self.address}, Port:{self.port}')

    def set_ae_info(self, aet, address, port):
        self.aet = aet
        self.address = address
        self.port = port
        try:
            make_my_ae()
        except NameError:
            pass
        global assoc
        assoc = None

    def set_from_saved(self, name):
        self.set_ae_info(**NETWORK_INFO['dicom'][name])

    def set_from_default(self, default):
        self.set_from_saved(NETWORK_INFO['dicom']['default'][default])

    def interactive_set_ae_from_saved(self):
        choices = list(NETWORK_INFO['dicom'].keys())
        for i, k in enumerate(choices):
            print(f'{i} - {k}')
        choice = input('Input an integer:\n')
        self.set_from_saved(choices[int(choice)])


# Create the default AE objects
default_my = os.environ.get('MEDPHUNC-PACSDEFAULT-MY')
if default_my:
    MY = AEInfo(name=default_my)
else:
    MY = AEInfo(default='me')
    logger.warning("MEDPHUNC-PACSDEFAULT-MY environment variable not set - " +
                   "choosing default local AE from definition in config file")

default_remote = os.environ.get('MEDPHUNC-PACSDEFAULT-REMOTE')
if default_remote:
    REMOTE = AEInfo(name=default_remote)
else:
    REMOTE = AEInfo(default='remote')
    logger.info("MEDPHUNC-PACSDEFAULT-MY environment variable not set " +
                   "- choosing default remote AE from definition in config file")
assoc = None


# %% Utility functions

def test_assoc(assoc):
    if 'is_alive' in dir(assoc):
        return assoc.is_alive()
    else:
        return assoc.isAlive()

def make_my_ae():
    global ae
    ae = AE(MY.aet)
    ae.requested_contexts = QueryRetrievePresentationContexts
    return ae
ae = None
make_my_ae()


def ensure_assoc():
    global assoc

    try:
        if test_assoc(assoc):
            return assoc
    except:
        pass
    assoc = ae.associate(REMOTE.address, REMOTE.port, ae_title=REMOTE.aet, max_pdu=32764)

    if test_assoc(assoc):
        return assoc
    else:
        logger.warning('Association failed')
        return assoc


def run_query(generator):
    x = []
    for a in generator:
        x.append(a)
    return x


# %% storage and ping functions, that use a different instance of ae and assoc
store_assoc = None

store_assoc_contexts = [[XRayRadiationDoseSRStorage, ImplicitVRLittleEndian],
                        [DigitalXRayImageStorageForPresentation, ImplicitVRLittleEndian]
                       ]
                        
def ensure_store_assoc(force = False):
    global store_assoc
    try:
        if test_assoc(store_assoc):
            if not force:
                return store_assoc
    except:
        pass
    store_ae = AE(MY.aet)
    for context in store_assoc_contexts:
        store_ae.add_requested_context(context[0], context[1])

    store_assoc = ae.associate(REMOTE.address, REMOTE.port, ae_title=REMOTE.aet, max_pdu=32764)
    if test_assoc(store_assoc):
        return store_assoc
    else:
        logger.warning('Association failed')
        return store_assoc


                
def do_store(d):
    ensure_store_assoc()
    try:
        generator = store_assoc.send_c_store(d)
        return run_query(generator)
    except ValueError as e:
        # Value errors are likely to occur when the context is not correct for the storage attempt. This won't happen for every SCP.
        logger.debug('ValueError occurred: %s, retrying with additional contexts', e)
        store_assoc_contexts.append([d.file_meta.MediaStorageSOPClassUID,
                            d.file_meta.TransferSyntaxUID])
        ensure_store_assoc(force=True)
        generator = store_assoc.send_c_store(d)
        return run_query(generator)



def do_ping():
    ping_ae = AE(MY.aet)
    ping_ae.requested_contexts = VerificationPresentationContexts
    assoc = ping_ae.associate(REMOTE.address, REMOTE.port, ae_title=REMOTE.aet, max_pdu=32764)
    if test_assoc(assoc):
        generator = assoc.send_c_echo()
        return run_query(generator)
    else:
        return 'Failed due to no assoc'


# %% Basic functions


def do_find(d):
    """
    Send search query using the supplied dataset

    Parameters
    ----------
    d : pydicom Dataset
        Dataset containin a valid dicom search.

    Returns
    -------
    x : List
        List of datasets corresponding to the search.
        
    """
    ensure_assoc()
    if d.QueryRetrieveLevel == 'PATIENT':
        generator = assoc.send_c_find(d, query_model=StudyRootQueryRetrieveInformationModelFind)
    else:
        generator = assoc.send_c_find(d, query_model=StudyRootQueryRetrieveInformationModelFind)
    x = run_query(generator)
    return x


def do_move(d):
    """
    Send move query using the supplied dataset.

    Parameters
    ----------
    d : pydicom Dataset
        Dataset containin a valid dicom move specification.

    Returns
    -------
        Results of the move request.
        
    """
    ensure_assoc()
    if d.QueryRetrieveLevel == 'PATIENT':
        generator = assoc.send_c_move(d, MY.aet, query_model=PatientRootQueryRetrieveInformationModelMove)
    else:
        generator = assoc.send_c_move(d, MY.aet, query_model=StudyRootQueryRetrieveInformationModelMove)
    x = run_query(generator)
    return x[-1]


def dfind(d):
    """
    Send search query using the supplied dataset, returning a pandas dataframe.

    Parameters
    ----------
    d : pydicom Dataset
        Dataset containin a valid dicom search.

    Returns
    -------
        Dataframe containing search results.
        
    """
    x = do_find(d)
    return process_find_results(x)


def process_find_results(x):
    if len(x) == 1:
        if x[0][0].Status == 0:
            return pd.DataFrame({})
        else:
            raise (ValueError(x[0][0]))
    else:
        return query_results_to_dataframe(x)


# %% Tricksy utility functions

def accession_to_study_uid(accession_number):
    search_results = search_accession(accession_number)
    if search_results is not None:
        return search_results.study_instance_uid
    else:
        return


def search_accession(accession_number):
    ss = SearchSet('study', AccessionNumber=accession_number)
    r = ss.find()
    if r.shape[0] == 1:
        return r.loc[0, :]
    elif r.shape[0] > 1:
        raise (ValueError('Multiple studies found with this accession number'))
    else:
        return


def study_from_patient_and_fuzzy_date(patient_id, nominal_study_date,
                                      date_deltas=(0, -1, 1, -2, 2),
                                      modality=None,
                                      study_description=None):
    daydelta = pd.to_timedelta(1, unit='day')
    nominal_study_date = pd.to_datetime(nominal_study_date)

    for date_delta in date_deltas:
        search_date = nominal_study_date + date_delta * daydelta
        search_date = search_date.strftime('%Y%m%d')
        ss = SearchSet('study', PatientID=patient_id,
                       StudyDate=search_date,
                       ModalitiesInStudy=modality,
                       StudyDescription=study_description
                       )
        r = ss.find()
        if r.shape[0] != 0:
            return {'delta_found': date_delta,
                    'study_instance_uid': r.StudyInstanceUID[0],
                    'search_results': r.iloc[0, :]}



# %% SearchSet. Create query datasets. Not for move requests.

class SearchSet(pydicom.Dataset):
    drill = {}
    drill_result = {}
    drill_merge = pd.DataFrame({})
    result = pd.DataFrame({})
    query_level = None

    def __init__(self, query_level='series', **kwargs):
        self.query_level = query_level.lower()
        super().__init__()
        if 'find' in kwargs:
            find = kwargs.pop('find')
        else:
            find = False
        self.initialise_searchable_tags(query_level)
        self.set_tags(**kwargs)
        if find:
            self.find()

    searchable_tags = {
        'study': [

            # Required for moves (some systems)
            'StudyInstanceUID',
            # Optional
            'StudyDate',
            'PatientName',
            'PatientID',
            'PatientBirthDate',
            'AccessionNumber',
            'StudyDescription',
            'StudyID',
            'StudyTime',
            'StudyInstanceUID',
            'NumberOfStudyRelatedSeries',
            'StationName',
            'SpecificCharacterSet',
            'ModalitiesInStudy',
            'Manufacturer',
            'ManufacturerModelName',
            'InstitutionName'
        ],
        'series': [
            # Required (some systems)
            'StudyInstanceUID',
            # Required for moves (some systems)
            'SeriesInstanceUID',
            # Optional
            'SeriesDescription',
            'SeriesNumber',
            'Modality',
            'NumberOfSeriesRelatedInstances',
            'StationName',
            'SpecificCharacterSet',
            'SeriesDate',
            'AccessionNumber',
            'SeriesTime',
            'Manufacturer',
            'ManufacturerModelName'
        ],
        'instance': [
            # Required (some systems)
            'SeriesInstanceUID',
            'StudyInstanceUID',
            # Required for moves (some systems)
            'SOPInstanceUID',
            # Optional
            'AccessionNumber',
            'StationName',
            'SeriesDate',
            'SeriesTime',
            'SOPClassUID',
            'InstanceNumber',
            'SpecificCharacterSet',
            'StudyInstanceUID'
        ],
        'patient': [
            'PatientName',
            'PatientSex',
            'PatientID',
            'AccessionNumber'
        ]
    }

    def initialise_searchable_tags(self, query_level):
        query_level = self.query_level
        if query_level == 'study':
            self.QueryRetrieveLevel = 'STUDY'
            tags = self.searchable_tags['study']
        elif query_level == 'series':
            self.QueryRetrieveLevel = 'SERIES'
            tags = self.searchable_tags['series']

        elif query_level in ['image', 'instance']:
            self.QueryRetrieveLevel = "IMAGE"
            tags = self.searchable_tags['instance']

        elif query_level == 'patient':  # todo find optional tags
            self.QueryRetrieveLevel = 'PATIENT'
            tags = self.searchable_tags['patient']
        else:
            raise NotImplementedError('The selected query level does not exist: %s', query_level)

        for tag in tags:
            self.write_tag_by_keyword(tag, '')

    def write_tag_by_keyword(self, keyword, value):
        tag = pydicom.datadict.tag_for_keyword(keyword)
        # ttag = pydicom.tag.Tag('StudyInstanceUID')
        ddat = pydicom.datadict.DicomDictionary[tag]
        self.add_new(tag, ddat[0], value)

    def set_tags(self, **kwargs):
        for k, v in kwargs.items():
            try:
                self.write_tag_by_keyword(k, v)
            except KeyError:
                print(f'could not write tag: "{k}" with value: "{v}"')

    def find(self):
        self.dcm_result = do_find(self)
        self.result = process_find_results(self.dcm_result)
        return self.result

    def move(self, index=None):
        dd = self.copy()
        if index is not None:
            if self.query_level == 'study':
                dd.StudyInstanceUID = self.result.StudyInstanceUID[index]
            if self.query_level == 'series':
                dd.SeriesInstanceUID = self.result.SeriesInstanceUID[index]
            if self.query_level == 'instance':
                dd.SOPInstanceUID = self.result.SOPInstanceUID[index]
        for key in self.keys():
            if dd[key].value == '':
                dd.pop(key)
        return do_move(dd), dd

    def copy(self):
        return copy.deepcopy(self)

    def drill_down(self, i=None, find=False):
        if i is None:
            print(self.result)

            i = int(input('Select index of item to drill down -> '))
        drill_result = self.result.loc[i, :]
        if self.query_level == 'study':
            search_tag = {'StudyInstanceUID': drill_result.StudyInstanceUID}
            level = 'series'
        elif self.query_level == 'series':
            level = 'instance'
            search_tag = {'StudyInstanceUID': drill_result.StudyInstanceUID,
                          'SeriesInstanceUID': drill_result.SeriesInstanceUID}
        elif self.query_level == 'instance':
            raise (ValueError("Can't drill down from image level"))
        else:
            raise (ValueError("Query retrieve level not properly set"))
        drill_result = self._drill(level, search_tag, find)
        self.drill[i] = drill_result
        if find:
            self.drill_result[i] = drill_result.result.copy()
            self.drill_result[i]['drill_index'] = i
        return drill_result

    def drill_all(self, find=True):
        """Drill down on any item in the self.result dataframe.
        Removing items from the result dataframe before calling this function
        can improve efficiency!
        
        """
        drill_all = [self.drill_down(i, find) for i in self.result.index]
        if find and drill_all:
            drill_concat = pd.concat(self.drill_result)
            self.drill_merge = pd.merge(self.result, drill_concat, how='left', on='StudyInstanceUID')
            self.drill_merge.columns = self.drill_merge.columns.str.replace('_x', '')
            self.drill_merge = self.drill_merge.loc[:, ~self.drill_merge.columns.str.contains('_y')]
        return drill_all

    def move_one_instance_all_series(self):
        "For every result for the current search, drill down to all series and retrieve a single dicom object from each"
        # If we're on the image level, search and return a single instance
        if self.QueryRetrieveLevel == 'IMAGE':
            self.SOPInstanceUID = self.result.SOPInstanceUID.iloc[0]
            logger.debug('At the image level, moving SOP instance %s', self.SOPInstanceUID)
            print(self)
            self.move()
            return self.SOPInstanceUID

        # If we're not on the image level, drill down
        logger.debug('Found %s items to drill down to', self.result.shape[0])
        output = []
        for i in self.result.index:
            logger.debug('At the %s level, drilling down to item %s', self.QueryRetrieveLevel, i)
            ss = self.drill_down(i)
            ss.find()
            output.append(ss.move_one_instance_all_series())
        return output

    @classmethod
    def _drill(cls, level, search_tags, find):
        search_tags['find'] = find
        drilled = cls(level, **search_tags)
        return drilled
    
    @classmethod
    def from_study_uid_or_accession(cls,
                                    study_uid = None,
                                    accession_number = None):
        if study_uid is not None:
            return cls('series', StudyInstanceUID=study_uid)
        elif accession_number is not None:
            t_study = cls('study',AccessionNumber=accession_number)
            t_study.find()
            return cls('series', StudyInstanceUID=t_study.result.StudyInstanceUID[0])
        else:
            raise(ValueError('Need to supply one of study_instance_uid or accession_number'))


# %% RDSR functions
class RDSR(SearchSet):
    """
    Separate class for searches that specifically revolve around RDSRs.
    %todo Should this just be combined with SearchSet?
    """
    def find_rdsrs(self):
        """
        Try to move all RDSR within the selected search results

        On the study level, finds all studies with SR objects and performs a series level search.
        On the series level, finds all series which are tagged SR
        On the instance level, finds all instances which match the RDSR SOP Class
        Yields an object of the same class for each object. These can be moved
        with .move(). The 'result' dataframe retains the instance level 
        search result.

        Returns
        -------
        list of RDSR objects

        """
        

        if self.QueryRetrieveLevel == 'STUDY':
            study_with_srs = (self.result.ModalitiesInStudy.explode() == 'SR').index.unique()
            logger.debug('%s studies containing SRs found', len(study_with_srs))
            # study_uids = self.result.StudyInstanceUID.loc[self.result.ModalitiesInStudy.apply(lambda x: 'SR' in x)]
            for study_index in study_with_srs:
                r = self.drill_down(study_index)
                logger.debug('Now searching in study %s', self.result.AccessionNumber.loc[study_index])
                r.find()
                yield from r.find_rdsrs()

        if self.QueryRetrieveLevel == 'SERIES':
            """functions for finding RDSRS out of a list of series data"""
            try:
                # siemens artis
                sr_series_indices = self.result.loc[self.result.Modality == 'SR'].index
                logger.debug('%s series containing SRs found', len(sr_series_indices))
            except IndexError:
                logger.warning('No SRs in study, find failed')
                return
            for sr_index in sr_series_indices:
                series_mover = self.drill_down(sr_index)
                series_mover.find()
                yield from series_mover.find_rdsrs()

        
        if self.QueryRetrieveLevel == 'IMAGE':
            try:
                # siemens artis
                rdsr_sop_instance_uids = self.result.loc[self.result.SOPClassUID == '1.2.840.10008.5.1.4.1.1.88.67'].SOPInstanceUID
            except IndexError:
                logger.warning('No RDSRs in series, move failed')
                return
            for rdsr_sop_instance_uid in rdsr_sop_instance_uids:
                
                rdsr_instance = self.copy()
                rdsr_instance.SOPInstanceUID = rdsr_sop_instance_uid
                yield rdsr_instance
                
                
    def move_rdsrs(self):
        
        logger.debug('Attempting to find and move RDSRs')
        rdsr_instance_list = self.find_rdsrs()
        rdsr_instance_list = list(rdsr_instance_list)
        logger.info('%s movable RDSRs found. Attempting to move.', len(rdsr_instance_list))
        for rdsr_instance in rdsr_instance_list:
            rdsr_instance.move()
        
        


def make_daterange(start_date, window=5):
    """Make a dicom-friendly date range from a datetime object, over the specified duration"""
    window = datetime.timedelta(days=window)
    try:
        # If a datetime has been provided, get the date
        start_date = start_date.date()
    except AttributeError:
        pass
    end_date = start_date + window
    date_range = str(start_date) + '_' + str(end_date)
    return date_range.replace('-', '').replace('_', '-')


# %% Query result parsing functions

# todo write this function
def dataset_to_series(d):
    pass


def query_results_to_dataframe(ds):
    s = []
    for line in ds:
        try:
            line_item = line[1]
        except IndexError:
            line_item = line
        try:
            line_item.AccessionNumber
        except AttributeError:
            continue
        ld = line_item.dir()
        d = {a: line_item.get(a) for a in ld}
        s.append(d)
    df = pd.DataFrame(s)
    return df


# %%

ae = make_my_ae()

# %%


if __name__ == '__main__':
    pass
    # date_range = '20191018'
    # station_filter = 'qhtsv*'
    # search_retrieve_rdsr(date_range, station_filter)
