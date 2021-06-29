# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:03:28 2019

Short script for performing risk calculations, based on either total effective
dose or CT expo organ calculated dose.

@author: Williamch
"""

import numpy as np
import pandas as pd
from math import floor, log10
import win32com.client
from tkinter.filedialog import askopenfilename

#%% Calculate via effective dose
#In accordance with, I believe, RPS 8:
# Ultiamtely sourced from (maybe): Estimates of late radiation risks to the UK  population.  Documents  of  the  NRPB  4(4),  1993:  Table  4.8:    Estimates  of  radiation-induced fatal cancer risks in a UK population

# polynomial coefficients for risk as a function of age

inc_f =  np.array([	4.27E-09,	-9.44E-07,	7.52E-05,	-2.72E-03,	4.70E-02])
inc_m  = np.array([ 	2.34E-09,	-5.26E-07,	4.13E-05,	-1.43E-03,	2.52E-02])
mort_f = np.array([	8.69E-10,	-2.32E-07,	2.11E-05,	-8.40E-04,	1.75E-02])
mort_m = np.array([	4.68E-10,	-1.35E-07,	1.25E-05,	-4.94E-04,	1.09E-02])

def get_risk(dose, age, risk_coefc = mort_m):
    p1 = np.poly1d(risk_coefc)
    return p1(age)*dose/100

def get_risk_string(dose, age, risk_coefc):
    r = get_risk(dose, age, risk_coefc)
    rr = round_to_2(1/r)
    return '1:%s' % ('%d' % rr)


def do_effective_dose_risk_calc():
    input_dose = float(input('Input dose in mGy\n>>'))
    age_range = select_age_range()
    ages = np.arange(age_range[0], age_range[1], 5)
    f_risks = 1/get_risk(input_dose, ages, inc_f)
    m_risks = 1/get_risk(input_dose, ages, inc_m)
    df = combine_m_f_risks(ages, f_risks, m_risks)
    return df

#%% Calculate via organ dose

# Load CT Expo spreadsheet

def load_dose_data(fn):
    input("This will cause (minor) issues with open Excel workbooks. Please close then press Enter to continue...")
    xlApp = win32com.client.Dispatch("Excel.Application")
    xlApp.Visible = False
    xlwb = xlApp.Workbooks.Open(fn, Password='VelvetSweatshop')
    xlws = xlwb.Sheets('Berechnung')
    row = 25
    col=9
    
    
    dose_dic = {}
    for i in range(15):
        for j in range(2):
            dose_dic[xlws.Cells(row+i, col+j*2).text] = xlws.Cells(row+i, col+1+j*2).value
    xlwb.Close(False)
    del(xlApp)
    del(xlwb)
    del(xlws)
    return dose_dic

def combine_ct_expo_workbooks(fns):
    dose_dics = {}
    for fn in fns:
        dose_dics[fn] = load_dose_data(fn)
    combined_dose = pd.DataFrame(dose_dics).sum(axis=1).to_dict()
    return combined_dose


#%% Generic class for risk calculations

class Risk:
    
    age = []
    gender = ''
    risk = pd.DataFrame()
    odds = pd.DataFrame()
    
    def __init__(self, age_range=None, gender='a'):
        "Initialse with age as a numeric or list of numerics, and gender one of 'm', 'f', 'a' "
        if not age_range:
            age_range = [0,100]
        
        c = ['male', 'female']
        if gender == 'm':
            c.pop(0)
        elif gender == 'f':
            c.pop(1)
        
        self.risk = pd.DataFrame(index=age_range, columns = c)
        self.age_range = np.array(age_range)
        self.gender = gender
        
        self.calculate_risk()
        if gender == 'a':
            self.combine_risks()
        elif gender == 'm':
            self.risk = self.risk.loc[:,'male']
        elif gender == 'f':
            self.risk = self.risk.loc[:,'female']
        self.calculate_odds()
        self.risk = self.risk.loc[(self.risk.index >= self.age_range[0]) & 
                                  (self.risk.index <= self.age_range[1]),:]



    def choose_age_range(self):
        a0 = int(input('Choose lower age bracket: '))
        amax = int(input('Choose upper age bracket: '))
        return [a0, amax]

    def calculate_risk(self):
        "Method for calculating risk that should be overwritten"
        raise(ValueError('This class needs to be extended with a valid risk calulation method before use'))
        self.risk.male = 0.1
        self.risk.female = 0.2
    
    def combine_risks(self):
        self.risk['average'] = (self.risk.male + self.risk.female)/2

    def round_to_2(x):
        return round(x, 1-int(floor(log10(abs(x)))))

        
    def to_clipboard(self):
        self.risk.applymap(float_formatter).T.to_clipboard()
    
    def calculate_odds(self):
        odds = 1/self.risk
        odds.columns = odds.columns.str.title() + ' risk'
        self.odds = odds.applymap(float_formatter)
        return self.odds
    
    def calculate_cohort_odds(self):
        if self.gender == 'a':
            return float_formatter(1/self.risk["average"].mean())
        else:
            return float_formatter(1/self.risk.iloc[:,0].mean())
    
    def individual_risk_interactive(self):
        patient_age = int(input('Input patient age: '))
        patient_sex = input('Patient sex (m/f) : ')
        if patient_sex == 'm':
            dat = mdat
        elif patient_sex == 'f':
            dat = fdat
        ssde_rescale = input('SSDE rescale factor (enter=1) : ')
        if ssde_rescale=='':
            ssde_rescale=1
        else:
            ssde_rescale=float(ssde_rescale)
        
        return self.individual_risk(patient_age, patient_sex, ssde_rescale)
    
        
    def individual_risk(self, age, gender, ssde_rescale = 1):
        risk = self.risk.copy()
        try:
            age_data = risk.loc[age,:]
        except:
            risk.loc[age,:] = np.nan
            risk = risk.interpolate(method='index')
            age_data = risk.loc[age,:]
        odds = 1/age_data
        odds.index = odds.index.str.title() + ' risk'
        odds = odds.apply(float_formatter)
        return age_data, odds

    def __str__(self):
        return str(self.calculate_odds())
    
    def __repr__(self):
        return str(self.calculate_odds())


#%% Organ import functions


class OrganRisk(Risk):
    
    #organs should be converted into a pd.Series with these index labels
    organs = ['Stomach', 'Low. Large int.', 'Liver', 'Lungs', 'Breasts',
              'Uterus', 'Ovaries', 'Bladder', 'Other', 'Thyroid', 'Bone marrow'] 
    
    organ_dose = pd.DataFrame(index=organs, columns = ['male', 'female'])
    
    def __init__(self, organ_dose, age_range=None):
        
        if organ_dose.shape[1]==1:
            gender = organ_dose.columns.str[0][0]
        else:
            gender = 'a'
        self.organ_dose = organ_dose
        super(OrganRisk, self).__init__(age_range, gender)

        
    
    # Methods for constructing based on different sources of organ dose.
    
    @classmethod
    def from_ctexpo(cls, age_range=None, male_fns=None, female_fns=None):
        
        if not (male_fns or female_fns):
            male_fns = OrganRisk.select_source_fns('(Male)')
            female_fns = OrganRisk.select_source_fns('(Female)')
        
        male_dose = combine_ct_expo_workbooks(male_fns)
        male_dose.name = 'male'
        female_dose = combine_ct_expo_workbooks(female_fns)
        female_dose.name = 'female'
        dats = [dat for dat in [male_dose, female_dose] if not dat.empty]
        organ_dose = pd.concat(dats,axis=1)
        
        organ_dose.loc['Low. Large int.',:] += organ_dose.loc['Upp. large int.',:]# + data.loc['Low. Large int','dose']
        organ_dose = organ_dose[organ_dose.index != 'Upp. Large int']
        organ_dose.loc['Other', :] = organ_dose.loc[~organ_dose.index.isin(OrganRisk.organs), :].mean(axis=0)
        
        return cls(organ_dose, age_range)
    
    
    @staticmethod
    def load_ctexpo_dose_data(fn):
        input("This will cause (minor) issues with open Excel workbooks. Please close then press Enter to continue...")
        xlApp = win32com.client.Dispatch("Excel.Application")
        xlApp.Visible = False
        xlwb = xlApp.Workbooks.Open(fn, Password='VelvetSweatshop')
        xlws = xlwb.Sheets('Berechnung')
        row = 25
        col=9
        
        
        dose_dic = {}
        for i in range(15):
            for j in range(2):
                dose_dic[xlws.Cells(row+i, col+j*2).text] = xlws.Cells(row+i, col+1+j*2).value
        xlwb.Close(False)
        del(xlApp)
        del(xlwb)
        del(xlws)
        return dose_dic
    
    @staticmethod
    def combine_ct_expo_workbooks(fns):
        dose_dics = {}
        for fn in fns:
            dose_dics[fn] = OrganRisk.load_ctexpo_dose_data(fn)
        combined_dose = pd.DataFrame(dose_dics).sum(axis=1)
        return combined_dose
    
    @staticmethod
    def select_source_fns(text=''):
        fns = []
        fn = 'hi'
        while fn != '':
            fn = askopenfilename(title=f'Choose dose source file {text} (Cancel to finish choosing)')
            fns.append(fn)
        return fns
    
    
    
    @classmethod
    def from_pcxmc(cls, age_range=None, male_fns=None, female_fns=None):
        
        if not (male_fns or female_fns):
            male_fns = OrganRisk.select_source_fns(' - PCXMC .gy, Male')
            female_fns = OrganRisk.select_source_fns(' - PCXMC .gy, Female')
        
        doses = []
        if male_fns:
            male_dose = [OrganRisk.load_pcxmc_data(fn) for fn in male_fns]
            male_dose = pd.concat(male_dose,axis=1).sum(axis=1)
            male_dose.name = 'male'
            doses.append(male_dose)
        
        if female_fns:
            female_dose = [OrganRisk.load_pcxmc_data(fn) for fn in female_fns]
            female_dose = pd.concat(female_dose,axis=1).sum(axis=1)
            female_dose.name = 'female'
            doses.append(female_dose)
        
        organ_dose = pd.concat(doses, axis=1)
        
        return cls(organ_dose,age_range)
        

    @staticmethod
    def load_pcxmc_data(fn):
        with open(fn,'r') as f:
            s = f.readlines()
        
        df = pd.DataFrame(s)
        df = df.iloc[22:,:]
        s = df.iloc[:,0].str.split(' {4,40}')
        df = pd.DataFrame({'organ':s.str[1].str.strip(),
        'dose':s.str[2].str.strip(),
        'uncertainty':s.str[3].str.strip()})
        df = df.set_index('organ')
        df = df.loc[:'Uterus',:]
        s = df.dose
        s=s.astype(float)
        s.index = s.index.str.replace('Active bone marrow', 'Bone marrow')
        s.index = s.index.str.replace('Colon (Large intestine)', 'Low. Large int.')
        s.index = s.index.str.replace('Urinary bladder', 'Bladder')
        
        s.loc['Other'] = s.loc[~s.index.isin(OrganRisk.organs)].mean()
        
        return s


#fns =['M:/MedPhy/General/^Ethics/2020/5-MAY/ATG-017-001_EAC_x/CT-Expo v2.5 (E).xls']
#o = OrganRisk.combine_ct_expo_workbooks(fns)
# o = OrganRisk(tt,[30,40])
# print(o)


#%% BEIR VII organ dose method


#BEIR VII Table 12D-1
class BEIR(OrganRisk):
    
    mdat=pd.DataFrame({'Age':[0,5,10,15,20,30,40,50,60,70,80],
    'Stomach':[76,65,55,46,40,28,27,25,20,14,7],
    'Low. Large int.':[336,285,241,204,173,125,122,113,94,65,30],
    'Liver':[61,50,43,36,30,22,21,19,14,8,3],
    'Lungs':[314,261,216,180,149,105,104,101,89,65,34],
    'Prostate':[93,80,67,57,48,35,35,33,26,14,5],
    'Bladder':[209,177,150,127,108,79,79,76,66,47,23],
    'Other':[1123,672,503,394,312,198,172,140,98,57,23],
    'Thyroid':[115,76,50,33,21,9,3,1,0.3,0.1,0],
    'Bone marrow':[237,149,120,105,96,84,84,84,82,73,48]})
    
    fdat=pd.DataFrame({'Age':[0,5,10,15,20,30,40,50,60,70,80],
    'Stomach':[101,85,72,61,52,36,35,32,27,19,11],
    'Low. Large int.':[220,187,158,134,114,82,79,73,62,45,23],
    'Liver':[28,23,20,16,14,10,10,9,7,5,2],
    'Lungs':[733,608,504,417,346,242,240,230,201,147,77],
    'Breasts':[1171,914,712,553,429,253,141,70,31,12,4],
    'Uterus':[50,42,36,30,26,18,16,13,9,5,2],
    'Ovaries':[104,87,73,60,50,34,31,25,18,11,5],
    'Bladder':[212,180,152,129,109,79,78,74,64,47,24],
    'Other':[1339,719,523,409,323,207,181,148,109,68,30],
    'Thyroid':[634,419,275,178,113,41,14,4,1,0.3,0],
    'Bone marrow':[185,112,86,76,71,63,62,62,57,51,37]})
    
    fdat = fdat.set_index('Age')
    mdat = mdat.set_index('Age')
    

    def calculate_risk(self):
        risks = []
        try:
            m_risk = (self.organ_dose.male*self.mdat).sum(axis=1)/100000
            m_risk.name = 'male'
            risks.append(m_risk)
        except:
            pass
        try:
            f_risk = (self.organ_dose.female*self.fdat).sum(axis=1)/100000
            f_risk.name = 'female'
            risks.append(f_risk)
        except:
            pass
        self.risk = pd.concat(risks,axis=1)

# b=BEIR(tt,[30,50])   
# print(b)

# bb = BEIR.from_pcxmc(male_fns = [fn])
# print(bb)

# bb.individual_risk(55,'m',1)

#%%


class RPS8Risk(Risk):

    risk_tables = {
        'incidence':{
            'female':np.array([	4.27E-09,-9.44E-07,	7.52E-05,-2.72E-03,	4.70E-02]),
            'male':np.array([ 2.34E-09,-5.26E-07,	4.13E-05,-1.43E-03,	2.52E-02])},
        'mortality':{
            'female':np.array([	8.69E-10,-2.32E-07,	2.11E-05,-8.40E-04,	1.75E-02]),
            'male':np.array([	4.68E-10,-1.35E-07,	1.25E-05,-4.94E-04,	1.09E-02])}
        }
    risk_metric = 'incidence'

    def __init__(self, effective_dose, age_range, gender='a', risk_metric='incidence'):
        self.risk_metric = risk_metric
        self.effective_dose = effective_dose
        super().__init__(age_range, gender)
    

    def calculate_risk(self):

        ages = np.arange(self.age_range[0], self.age_range[1]+1, 5,  )
        risk_table = self.risk_tables[self.risk_metric]
        f_risks = self.get_risk(self.effective_dose, ages, risk_table['female'])
        m_risks = self.get_risk(self.effective_dose, ages, risk_table['male'])
        self.risk = pd.DataFrame({'female':f_risks,'male':m_risks}, index=ages)
        

    @staticmethod
    def get_risk(dose, age, risk_coef):
        p1 = np.poly1d(risk_coef)
        return p1(age)*dose/100
    
    @staticmethod
    def get_risk_string(dose, age, risk_coef):
        r = get_risk(dose, age, risk_coef)
        rr = round_to_2(1/r)
        return '1:%s' % ('%d' % rr)



#%%



    
def select_ct_expo_spreadsheet(text):
    fn = askopenfilename(title=text)
    #fn = 'C:/Users/WilliamCh/Desktop/research risk assesment for RB/CT-Expo v2.5 rb ATLAS.xls'
    return fn


    
def float_formatter(x):
    return '1 in %s' % ('%d' % float('%.2g' % x))
        

    
#%%
    
def risk_by_age(age, dose_dic, risk_dat):
    age_cat = risk_dat.index[risk_dat.index >= age][0]
    row = risk_dat.loc[age_cat,]
    data = pd.DataFrame({'lar':row.to_dict(), 'dose':dose_dic})
    data.loc['Low. Large int.','dose'] += data.loc['Upp. large int.','dose']# + data.loc['Low. Large int','dose']
    data = data[data.index != 'Upp. Large int']
    data.loc['Other', 'dose'] = data.loc[data.lar != data.lar, 'dose'].mean()
    data = data[data.lar==data.lar]
    return 10000000/(data.lar*data.dose).sum()

def risk(ages, dose_dic, risk_dat):
    risks = [risk_by_age(age, dose_dic, risk_dat) for age in ages]
    risks = np.array(risks)
    return risks
    
def combine_m_f_risks(ages, f_risk, m_risk):
    a_risk = 1/(.5/np.array(f_risk) + .5/np.array(m_risk))
    df = pd.DataFrame([f_risk, m_risk, a_risk],columns=ages,index=['Female risk', 'Male risk', 'Average risk'])
    return df

def summarise_risks(age_range, f_dose_dic, m_dose_dic):
    ages = mdat.loc[(mdat.index >= age_range[0]) & (mdat.index <= age_range[1]),].index
    f_risk = risk(ages, f_dose_dic, fdat)
    m_risk = risk(ages, m_dose_dic, mdat)
    df = combine_m_f_risks(ages, f_risk, m_risk)
    return df
    
def round_to_2(x):
    return round(x, 1-int(floor(log10(abs(x)))))

def select_ct_expo_spreadsheet(text):
    fn = askopenfilename(title=text)
    #fn = 'C:/Users/WilliamCh/Desktop/research risk assesment for RB/CT-Expo v2.5 rb ATLAS.xls'
    return fn

def select_age_range():
    a0 = int(input('Choose lower age bracket: '))
    amax = int(input('Choose upper age bracket: '))
    return [a0, amax]

def float_formatter(x):
    return '1 in %s' % ('%d' % float('%.2g' % x))

def do_ct_expo_risk_calc():
    #fn = select_ct_expo_spreadsheet()
    age_range = select_age_range()
    male_dose_dic = load_dose_data(select_ct_expo_spreadsheet('Choose CT Expo file for male'))
    female_dose_dic = load_dose_data(select_ct_expo_spreadsheet('Choose CT Expo file for female'))
    #male_dose_dic = load_dose_data('C:/Users/WilliamCh/Desktop/research risk assesment for RB/CT-Expo v2.5 rb ATLAS male.xls')
    #female_dose_dic = load_dose_data('C:/Users/WilliamCh/Desktop/research risk assesment for RB/CT-Expo v2.5 rb ATLAS female.xls')
    df=summarise_risks(age_range, female_dose_dic, male_dose_dic)
    df = df/float(input('Use a dose multiplier of: '))
    df.applymap(float_formatter).to_clipboard()
    return df


def single_patient_ct_expo_calc():
    patient_age = int(input('Input patient age: '))
    patient_sex = input('Patient sex (m/f) : ')
    if patient_sex == 'm':
        dat = mdat
    elif patient_sex == 'f':
        dat = fdat
    ssde_rescale = input('SSDE rescale factor (enter=1) : ')
    if ssde_rescale=='':
        ssde_rescale=1
    else:
        ssde_rescale=float(ssde_rescale)
    
    age_bracket = [patient_age//10*10, patient_age//10*10+10]
    
    fn = 'go'
    fns = []
    while fn != '':
        fn = select_ct_expo_spreadsheet('Choose CT Expo file (Cancel to finish choosing)')
        fns.append(fn)
        
    dose_data = combine_ct_expo_workbooks(fns)
    dose_data = {k:v*ssde_rescale for k,v in dose_data.items()}
    
    patient_risk = risk(age_bracket, dose_data, dat)
    df = pd.DataFrame([patient_risk], index=['Risk'],columns=age_bracket)

    df.applymap(float_formatter).to_clipboard()
    
    return df
    
    
def interactive_risk():
    from tkinter.filedialog import askopenfilename
    user_input = input('Choose an option:\n1 : From total effective dose\n2 : From CT Expo organ doses\n3 : From multiple CT expo files for a single patient\n>>')
    if user_input=='1':
        output = do_effective_dose_risk_calc()        
    elif user_input == '2':
        output = do_ct_expo_risk_calc()
    elif user_input == '3':
        output = do_ct_expo_risk_calc()
    else:
        raise('Goblins have prevented your input from being acceptable')
    output.applymap(float_formatter).to_clipboard()
    print(output.applymap(float_formatter))
    print('Output added to clipboard - paste into excel or word')
    input('Press Enter to close')


#%% Calculate risk from PCXMC export using BEIR


#%%
if __name__=="__main__":
    interactive_risk()