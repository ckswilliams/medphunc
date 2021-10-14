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
from pynetdicom import AE
from pynetdicom import QueryRetrievePresentationContexts
from pynetdicom import VerificationPresentationContexts
from pynetdicom import StoragePresentationContexts

from time import sleep
import datetime
import pandas as pd
import logging
import json
import pathlib
import copy
#from pynetdicom import debug_logger
#debug_logger()


#%%
import pynetdicom
if pynetdicom.__version__ >= '1.5':
    from pynetdicom.sop_class import PatientRootQueryRetrieveInformationModelFind
    from pynetdicom.sop_class import PatientRootQueryRetrieveInformationModelMove
    from pynetdicom.sop_class import StudyRootQueryRetrieveInformationModelFind
    from pynetdicom.sop_class import StudyRootQueryRetrieveInformationModelMove
else:
    PatientRootQueryRetrieveInformationModelFind = 'P'
    PatientRootQueryRetrieveInformationModelMove = 'P'
    StudyRootQueryRetrieveInformationModelFind = 'S'
    StudyRootQueryRetrieveInformationModelMove = 'S'
    

#%%

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

if len(logger.handlers) == 0:
    logger.addHandler(logging.StreamHandler())


#%% AE wrapper class

with open(pathlib.Path(__file__).parent.absolute() / 'aeinfo.json', 'r') as f:
    savedaeinfo = json.load(f)
    
class ae_info:
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
        
    def set_from_saved(self, name):
        self.set_ae_info(**savedaeinfo[name])
        
    def set_from_default(self, default):
        self.set_from_saved(savedaeinfo['default'][default])
        
    def interactive_set_ae_from_saved(self):
        choices = list(savedaeinfo.keys())
        for i, k in enumerate(choices):
            print(f'{i} - {k}')
        choice = input('Input an integer:\n')
        self.set_from_saved(choices[int(choice)])

MY = ae_info(default='me')
REMOTE = ae_info(default='remote')

assoc = None

#%% Utility functions

def make_my_ae():
    ae = AE(MY.aet)
    ae.requested_contexts = QueryRetrievePresentationContexts
    return ae


def ensure_assoc():
    global assoc
    def test_assoc(assoc):
        if 'is_alive' in dir(assoc):
            return assoc.is_alive()
        else:
            return assoc.isAlive()
    
    try:
        if test_assoc(assoc):
            return assoc
    except:
        pass
    assoc = ae.associate(REMOTE.address, REMOTE.port, ae_title=REMOTE.aet, max_pdu=32764)
    
    if test_assoc(assoc):
        return assoc
    else:
        print('assoc failed :(')
        return assoc


def run_query(generator):
    x = []
    for a in generator:
        x.append(a)
    return x



#%% storage and ping functions, that use a different instance of ae and assoc


def do_store(d):
    ae = AE(MY.aet)
    ae.requested_contexts = StoragePresentationContexts
    assoc = ae.associate(REMOTE.address, REMOTE.port, ae_title=REMOTE.aet, max_pdu=32764)
    generator = assoc.send_c_store(d)
    return run_query(generator)


def do_ping():
    ae = AE(MY.aet)
    ae.requested_contexts = VerificationPresentationContexts
    assoc = ae.associate(REMOTE.address, REMOTE.port, ae_title=REMOTE.aet, max_pdu=32764)
    if assoc.isAlive():
        generator = assoc.send_c_echo()
        return run_query(generator)
    else:
        return 'Failed due to no assoc'


#%% Basic functions


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
        generator = assoc.send_c_find(d,query_model = StudyRootQueryRetrieveInformationModelFind)
    else:
        generator = assoc.send_c_find(d, query_model = StudyRootQueryRetrieveInformationModelFind)
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
        generator = assoc.send_c_move(d, MY.aet, query_model = PatientRootQueryRetrieveInformationModelMove)
    else:
        generator = assoc.send_c_move(d, MY.aet, query_model = StudyRootQueryRetrieveInformationModelMove)
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
    if len(x) == 1:
        if x[0][0].Status==0:
            return pd.DataFrame()
        else:
            raise(ValueError(x[0][0]))
    else:
        return query_results_to_dataframe(x)

#%% Tricksy utility functions

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
        return r.loc[0,:]
    elif r.shape[0] > 1:
        raise(ValueError('Multiple studies found with this accession number'))
    else:
        return

def study_from_patient_and_fuzzy_date(patient_id, nominal_study_date,
                                         date_deltas=[0,-1,1,-2,2],
                                      modality=None,
                                      study_description=None):
    
    daydelta = pd.to_timedelta(1, unit='day')
    nominal_study_date = pd.to_datetime(nominal_study_date)
    
    for date_delta in date_deltas:
        search_date = nominal_study_date + date_delta*daydelta
        search_date = search_date.strftime('%Y%m%d')
        ss = SearchSet('study', PatientID=patient_id,
                       StudyDate = search_date,
                       ModalitiesInStudy=modality,
                       StudyDescription = study_description
                       )
        r = ss.find()
        if r.shape[0] != 0:
                return {'delta_found':date_delta,
                    'study_instance_uid': r.StudyInstanceUID[0],
                    'search_results':r.iloc[0,:]}
        

    

#%% Basic movement functions

def move_sop_instance(sop_instance_uid):
    d = Dataset()
    d.QueryRetrieveLevel = 'IMAGE'
    d.SOPInstanceUID = sop_instance_uid
    do_move(d)

def move_sop_instance_list(sop_instance_uids):
    for sop_instance in sop_instance_uids:
        move_sop_instance(sop_instance)


def move_series(series_uid):
    d = Dataset()
    d.QueryRetrieveLevel = 'SERIES'
    d.SeriesInstanceUID = series_uid
    return do_move(d)


def move_series_list(series_uid_list):
    for series_instance_uid in series_uid_list:
        move_series(series_instance_uid)


def move_study_uid(study_uid, one_per_series = False):
    """Move a full study, one_per_series doesn't work yet"""
    d=Dataset()
    d.QueryRetrieveLevel = 'STUDY'
    d.StudyInstanceUID = study_uid
    x = do_move(d)
    return x

def move_accession_number(accession_number, one_per_series = True):
    if one_per_series:
        print('Getting metadata for study')
        d = make_dataset('image')
        d.AccessionNumber = accession_number
        d.QueryRetrieveLevel = 'IMAGE'
        x = do_find(d)
        x = query_results_to_dataframe(x)
        y = x.groupby('SeriesInstanceUID').SOPInstanceUID.first()
        print('Grabbing images')
        for sop_instance_uid in y:
            move_sop_instance(sop_instance_uid)
    else:
        d = Dataset()
        d.QueryRetrieveLevel = 'STUDY'
        d.AccessionNumber = accession_number
        do_move(d)

#%% Search functions (for pacs with series/image level query restrictions)

def find_series_from_study(StudyInstanceUID = '', AccessionNumber = '', **kwargs):
    t = locals()
    tt = t.pop('kwargs')
    t = {**t, **tt}
    d = make_dataset('study', **t)
    x = do_find(d)
    
    t['StudyInstanceUID'] = x[0][1].StudyInstanceUID
    
    d = make_dataset('series', **t)
    xx = do_find(d)
    return xx


def find_images_from_study(StudyInstanceUID = '', AccessionNumber = '', **kwargs):
    x = find_series_from_study(StudyInstanceUID, AccessionNumber, **kwargs)
    x = x[:-1]
    results = []
    for xx in x:
        kwargs['StudyInstanceUID'] = xx[1].StudyInstanceUID
        kwargs['SeriesInstanceUID'] = xx[1].SeriesInstanceUID
        d = make_dataset('image', **kwargs)
        z = do_find(d)
        results = results + z[:-1]
    return results


def find_images_from_series():
    pass

#%% SearchSet. Create query datasets. Not for move requests.

class SearchSet(pydicom.Dataset):
    
    def __init__(self, query_level='series', **kwargs):
        super(SearchSet, self).__init__()
        self.initialise_searchable_tags(query_level)
        self.set_tags(**kwargs)

    def initialise_searchable_tags(self, query_level):

        if query_level=='study':
            self.QueryRetrieveLevel = 'STUDY'
            self.StudyDate = ''
            #Required
            self.StudyInstanceUID = ''
            #Optional
            self.PatientName = ''
            self.PatientID = ''
            self.PatientBirthDate = ''
            self.AccessionNumber = ''
            self.StudyDescription = ''
            self.StudyID = ''
            self.StudyTime = ''
            self.StudyInstanceUID = ''
            self.NumberOfStudyRelatedSeries = ''
            self.StationName = ''
            self.SpecificCharacterSet = ''
            self.ModalitiesInStudy = ''
            self.Manufacturer = ''
            self.ManufacturerModelName = ''
            self.InstitutionName = ''
    
        if query_level=='series':
            self.QueryRetrieveLevel = 'SERIES'
            self.SeriesDescription = ''
            self.SeriesNumber = ''
            self.SeriesInstanceUID = ''
            self.Modality = ''
            self.NumberOfSeriesRelatedInstances = ''
            self.StationName = ''
            self.SpecificCharacterSet = ''
            self.SeriesDate = ''
            self.AccessionNumber = ''
            self.SeriesTime = ''
            self.Manufacturer = ''
            self.ManufacturerModelName = ''
            self.SOPClassUID = ''
    
        if query_level=='image':
            self.QueryRetrieveLevel = "IMAGE"
            #Required
            self.SeriesInstanceUID = ''
            #Optional
            self.StudyInstanceUID = ''
            self.SOPInstanceUID = ''
            self.AccessionNumber = ''
            self.StationName = ''
            self.SeriesDate = ''
            self.SeriesTime = ''
            self.SOPClassUID = ''
            self.InstanceNumber = ''
            self.SpecificCharacterSet = ''
            self.StudyInstanceUID = ''
    
        if query_level=='patient': #todo find optional tags
            self.QueryRetrieveLevel = 'PATIENT'
            self.PatientName = ''
            self.PatientSex = ''
            self.PatientID = ''
            self.AccessionNumber = ''
 
    def write_tag_by_keyword(self, keyword, value):
        tag = pydicom.datadict.tag_for_keyword(keyword)
        #ttag = pydicom.tag.Tag('StudyInstanceUID')
        ddat = pydicom.datadict.DicomDictionary[tag]
        self.add_new(tag, ddat[0], value)
    
    def set_tags(self, **kwargs):
        for k, v in kwargs.items():
            try:
                self.write_tag_by_keyword(k, v)
            except:
                print(f'could not write tag: "{k}" with value: "{v}"')             

    
    def find(self):
        self.result = dfind(self)
        return self.result
        
    def move(self):
        dd = self.copy()
        for key in self.keys():
            if dd[key].value == '':
                dd.pop(key)
        return do_move(dd)
    
    def copy(self):
        return copy.deepcopy(self)
        

#%% Deprecated dataset functions

def write_tag_by_keyword(d, keyword, value):
    tag = pydicom.datadict.tag_for_keyword(keyword)
    #ttag = pydicom.tag.Tag('StudyInstanceUID')
    ddat = pydicom.datadict.DicomDictionary[tag]
    d.add_new(tag, ddat[0], value)
    return d



        
def make_dataset(query_level='series', **kwargs):

    if query_level=='study':
        d = Dataset()
        d.QueryRetrieveLevel = 'STUDY'
        d.StudyDate = ''
        #Required
        d.StudyInstanceUID = ''
        #Optional
        d.PatientName = ''
        d.PatientID = ''
        d.AccessionNumber = ''
        d.StudyDescription = ''
        d.StudyID = ''
        d.StudyTime = ''
        d.StudyInstanceUID = ''
        d.NumberOfStudyRelatedSeries = ''
        d.StationName = ''
        d.SpecificCharacterSet = ''
        d.ModalitiesInStudy = ''
        d.Manufacturer = ''
        d.ManufacturerModelName = ''
        d.InstitutionName = ''

    if query_level=='series':
        d = Dataset()
        d.QueryRetrieveLevel = 'SERIES'
        d.SeriesDescription = ''
        d.SeriesNumber = ''
        d.SeriesInstanceUID = ''
        d.Modality = ''
        d.NumberOfSeriesRelatedInstances = ''
        d.StationName = ''
        d.SpecificCharacterSet = ''
        d.SeriesDate = ''
        d.AccessionNumber = ''
        d.SeriesTime = ''
        d.Manufacturer = ''
        d.ManufacturerModelName = ''
        d.SOPClassUID = ''

    if query_level=='image':
        d = Dataset()
        d.QueryRetrieveLevel = "IMAGE"
        #Required
        d.SeriesInstanceUID = ''
        #Optional
        d.StudyInstanceUID = ''
        d.SOPInstanceUID = ''
        d.AccessionNumber = ''
        d.StationName = ''
        d.SeriesDate = ''
        d.SeriesTime = ''
        d.SOPClassUID = ''
        d.InstanceNumber = ''
        d.SpecificCharacterSet = ''
        d.StudyInstanceUID = ''

    if query_level=='patient': #todo find optional tags
        d = Dataset()
        d.QueryRetrieveLevel = 'PATIENT'
        d.PatientName = ''
        d.PatientSex = ''
        d.PatientID = ''
        d.AccessionNumber = ''
        
    for k, v in kwargs.items():
        try:
            d = write_tag_by_keyword(d, k, v)
        except:
            print(f'could not write tag: "{k}" with value: {v}')
            
    return d
        
#%% RDSR functions




class RDSR(SearchSet):
       
    
    def move_rdsrs(self):
        
        if self.QueryRetrieveLevel == 'STUDY':
            study_uids = self.result.StudyInstanceUID.loc[self.result.ModalitiesInStudy.apply(lambda x: 'SR' in x)]
            for study_uid in study_uids:
                r = RDSR('series', StudyInstanceUID = study_uid)
                r.find()
                r.move_rdsrs()
             
    
        if self.QueryRetrieveLevel == 'SERIES':
            """functions for getting RDSRS out of a list of series data go here"""
            try:
                #siemens artis
                rdsr_series_uid = self.result.loc[self.result.Modality=='SR'].SeriesInstanceUID.iloc[0]
            except ValueError:
                print('No SRs in study, move failed')
                return
            series_mover = self.copy()
            series_mover.SeriesInstanceUID = rdsr_series_uid
            return series_mover.move()
            
    
    @classmethod
    def from_accession(cls, accession_number):
        return cls('study', AccessionNumber=accession_number)

    @classmethod
    def from_study_uid(cls, study_uid):
        return cls('study', StudyInstanceUID=study_uid)

def find_rdsr_series_from_study_uid(study_uid):
    d = make_dataset('series', StudyInstanceUID = study_uid)
    x = dfind(d)
    
    pass
'''


for study_uid in df.StudyInstanceUID:
    sss = SearchSet('series', StudyInstanceUID = study_uid)
    ddf = sss.find()
    series_uid = ddf.loc[ddf.SeriesNumber == 990,'SeriesInstanceUID'].iloc[0]
    sss.SeriesInstanceUID = series_uid
    print(sss.move())
    print()

for i, row in df:
    sss = pi.SearchSet('series', StudyInstance)

'''


def is_series_rdsr(study_uid, series_uid):
    pass
    
def retrieve_rdsrs_from_study_uid(study_uid):
    pass

class Multilevel:
    
    @staticmethod
    def get_rdsr_sop_uids_from_query_results(query_results):
        if len(query_results) == 1:
            return []
        df = query_results_to_dataframe(query_results)
        tdf = df[df.SOPClassUID == '1.2.840.10008.5.1.4.1.1.88.67']
        if tdf.shape[0] > 0:
            return tdf.SOPInstanceUID.tolist()
        else:
            return []
    
    @staticmethod
    def retrieve_rdsr_from_query_results(query_results):
        sop_uids = Multilevel.get_rdsr_sop_uids_from_query_results(query_results)
        logger.info('Found {} RDSR SOP UIDs'.format(len(sop_uids)))
        move_sop_instance_list(sop_uids)
        return sop_uids
    
    @staticmethod
    def retrieve_rdsr_sop_uids_from_study(study_instance_uid):
        d = make_dataset('image')
        d.StudyInstanceUID = study_instance_uid
        d.Modality = 'SR'
        x = do_find(d)
        logger.info('Found {} items which might be RDSRs'.format(len(x)-1))
        sop_uids = Multilevel.get_rdsr_sop_uids_from_query_results(x)
        return sop_uids
    
    @staticmethod
    def retrieve_rdsr_from_study(study_instance_uid):
        
        d = make_dataset('image')
        d.StudyInstanceUID = study_instance_uid
        d.Modality = 'SR'
        x = do_find(d)
        sop_uids = Multilevel.retrieve_rdsr_from_query_results(x)
        return sop_uids
    
    @staticmethod
    def retrieve_rdsr_from_accession(accession_number):
        d = make_dataset('image')
        d.AccessionNumber = str(accession_number)
        d.Modality = 'SR'
        x = do_find(d)
        sop_uids = Multilevel.retrieve_rdsr_from_query_results(x)
        return sop_uids
    
    @staticmethod
    def search_retrieve_rdsr(date_range, station_filter=''):
        d = make_dataset('image')
        d.StudyDate = date_range
        d.StationName = station_filter
        d.Modality = 'SR'
        x = do_find(d)
        logger.info('Found {} items which might be RDSRs'.format(len(x)-1))
        sop_uids = Multilevel.retrieve_rdsr_from_query_results(x)
        return x, sop_uids




#%%


def make_daterange(start_date, window = 5):
    """Make a dicom-friendly date range from a datetime object, over the specified duration"""
    window = datetime.timedelta(days=window)
    try:
        start_date = start_date.date()
    except:
        pass
    end_date=start_date + window
    date_range = str(start_date) + '_' + str(end_date)
    return date_range.replace('-','').replace('_','-')

#%% Query result parsing functions

#todo write this function
def dataset_to_series(d):
    pass


def query_results_to_dataframe(ds):
    s = []
    for line in ds:
        try:
            l = line[1]
        except:
            l = line
        try:
            l.AccessionNumber
        except:
            continue
        ld = l.dir()
        d = {a:l.get(a) for a in ld}
        s.append(d)
    df = pd.DataFrame(s)
    return df


#%% General scripts





#%% II related scripts


def get_ii_data(date_range, station_filter):
    df = query_ii_study_list(date_range, station_filter)
    move_list = query_ii_dose_capture_uid(df)
    move_series_list(move_list)


#This script finds philips and hologic secondary captures by using some wacky science
def get_secondary_captures(df):
    #Philips BV and Veradius filter
    philips = df.loc[df.Manufacturer.str.contains('Philips'),:]
    philips = philips.loc[philips.SeriesNumber==0,:]

    #Hologic Fluoroscan Insight filter
    hologic = df.loc[df.Manufacturer.str.contains('Hologic'),:]
    hologic = hologic.loc[hologic.SeriesTime=='']
    hologic = hologic.loc[hologic.Modality=='RF']
    return pd.concat([philips, hologic])


# Queries PACS and retrieves dataframe containing studies that meet the supplied
# filters
def query_ii_study_list(date_range, station_filter):
    dfs = []
    for m in ['OT', 'RF']:

        d = make_dataset('study')
        d.ModalitiesInStudy = [m]
        d.StudyDate = date_range
        d.StationName = station_filter
        d.StudyDescription = 'II*'

        f = do_find(d)
        dfs.append(query_results_to_dataframe(f))

    df = pd.concat(dfs)
    df = df.drop_duplicates('StudyInstanceUID')
    print('Found {} studies in the time period {}'.format(df.shape[0], date_range))
    return df

# Takes a dataframe containing studies, queries on a series level to find
# secondary captures, then moves them to the local AE
def query_ii_dose_capture_uid(df):
    query_list = []
    for i, r in df.iterrows():
        d = make_dataset('series')
        d.StudyInstanceUID = r.StudyInstanceUID
        ff = do_find(d)
        ddf = query_results_to_dataframe(ff)
        try:
            cdf = get_secondary_captures(ddf)
        except:
            print('skipping study {}'.format(r.SeriesInstanceUID))
            continue
        query_list = query_list + list(cdf.SeriesInstanceUID.unique())
    query_list = pd.Series(query_list).unique()
    print('Finished looking for dose capture uids, found {} series'.format(len(query_list)))
    return query_list



#%%

ae = make_my_ae()

#%%


if __name__ == '__main__':
    pass
    #date_range = '20191018'
    #station_filter = 'qhtsv*'
    #search_retrieve_rdsr(date_range, station_filter)