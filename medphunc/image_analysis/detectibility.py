# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 16:02:45 2021

@author: willcx
"""


from matplotlib import pyplot as plt
import numpy as np
import typing

#from medphunc.image_analysis import nnps
#from medphunc.image_analysis import image_utility as iu
from medphunc.image_analysis import clinical_mtf, clinical_nps

import pydicom

from typing import Type


#%%

class TaskFunction:
    task_type = "cylinder"
    
    task_radius = 1.5 # mm
    task_contrast = 1 # HU
    pixel_size = 0.5 #mm
    
    pixel_reduction_factor = 5.0
    rebin_f_inc = 0.05
    loess_bandwidth = 0.02
    loess_robustness = 10
    loess_on_lsf = False
    
    freq_res = 0.01
    
    n_elements = 4096
    
    pixel_size = 0.5
    
    @property
    def nyquist(self):
        return 1/(2*self.pixel_size)

    def create_task(self):
        # Create a linspace swice as large as the task
        x_half_range = 1/self.freq_res/2
        X = np.linspace(-2*x_half_range, 2*x_half_range, self.n_elements)
        # Create square wave task in real space (equivalent to a cylinder in 2d)
        T = ((X>-self.task_radius) & (X<self.task_radius))*self.task_contrast
        # Convert to frequency space via FFT
        W = np.abs(np.fft.fft(T)) / self.n_elements /self.freq_res
        
        f = np.arange(0,self.nyquist, self.freq_res)
        W = W[:len(f)]
        self.f = f
        self.W = W
    
    def __init__(self, task_radius=1.5, task_contrast=1, pixel_size=0.5, freq_res = 0.01):
        self.task_radius = task_radius
        self.task_contrast=task_contrast
        self.pixel_size=pixel_size
        self.freq_res = freq_res
        
        self.create_task()
        
    def plot(self):
        plt.plot(self.f, self.W)


#%%

def eye_model(f_array, model='sol', **kwargs):
    if model=='sol':
        return EyeSol(f_array)
    elif model=='ricsie':
        return EyeRicSie(f_array)

class EyeModel:
    f_array=None
    E = None
    
    def __init__(self, f_array):
        self.f_array = f_array
        self.calculate_eye_model()
        
    def calculate_eye_model(self):
        "Calculation method. Must be overridden in subclasses"
        pass
    
    @classmethod
    def from_freq_cutoff(cls, freq_cutoff, freq_increment, **kwargs):
        f_array = cls.calc_freq(freq_cutoff, freq_increment)
        return cls(f_array, **kwargs)
    
    @staticmethod
    def calc_freq(freq_cutoff, freq_increment):
        return np.arange(0, freq_cutoff, freq_increment)
    
    def plot(self):
        plt.plot(self.f_array, self.E)
        

class EyeSol(EyeModel):
    
    # Viewing information
    recon_fov = 400.0 # Reconstructed field of view of image (mm)
    viewing_dist = 500.0 #The viewing distance (mm)
    display_pixel_pitch = 0.2 # The size of a pixel on the display (mm)
    
    #Image display information
    image_pixel_width = 512 # The number of pixels in the image width
    display_zoom_factor = 3.0 # The zoom applied to the image on the display
    
    # Parameters for equation
    a1 = 1.5
    a2 = 0.98
    a3 = 0.68
    
    @property
    def f_to_rho(self):
        return self.recon_fov * self.viewing_dist * np.pi / self.display_size / 180
    
    @property
    def display_size(self):
        return self.display_pixel_pitch * self.image_pixel_width * self.display_zoom_factor
        
    def calculate_eye_model(self):
        
        #angular frequency
        rho = self.f_array * self.f_to_rho
        
        e = rho**self.a1 * np.exp(-self.a2*rho**self.a3)
        
        # Normalise by dividing by max and square
        e = (e/e.max())**2
        
        self.E = e
        return e


class EyeRicSie(EyeModel):

    #Parameters for equation
    n = 1.3
    c = 3.093
    
    def calculate_eye_model(self):
        
        self.E = self.f_array ** self.n * np.exp(-self.c * self.f_array**2)
        return self.E
    
#%%
class Detectibility:
    nps: dict = None
    ttf: dict = None
    eye_model: Type[EyeModel] = None
    task_function: Type[TaskFunction] = None
    
    def __init__(
            self, nps: dict, ttf: dict,
            eye_model: Type[EyeModel], task_function: Type[TaskFunction]
            ):
        self.nps = nps
        self.ttf = ttf
        self.eye_model = eye_model
        self.task_function=task_function
        self.common_basis()
        self.calculate_detectibility()
    
    def common_basis(self):
        self.f_basis = self.task_function.f
        self.W=self.task_function.W
        # Interpolate all other functions to have the same frequency index
        self.E = np.interp(self.f_basis, self.eye_model.f_array, self.eye_model.E)
        self.TTF = np.interp(self.f_basis, self.ttf['Clinical']['frequency'], self.ttf['Clinical']['MTF'])
        self.NPS = np.interp(self.f_basis, self.nps['frequency'],self.nps['radial'])
    
    def calculate_detectibility(self):

        # calculate top and bottom of NPWE equation
        top = (self.W**2 * self.TTF**2*self.E**2)
        bottom = (self.W**2*self.TTF**2*self.NPS*self.E**4)
        # integrate and root
        # Note that 
        self.d = (top.sum()**2/bottom.sum())**0.5
        return self.d
    
    
def clinical_detectibility(im: Type[np.ndarray], d: Type[pydicom.Dataset]):
    
    
    ttf = clinical_mtf.clinical_mtf_from_dicom_metadata(im, d)
    nps = clinical_nps.automate_clinical_nnps(im, d)
    t = TaskFunction(task_radius=1.5, task_contrast=5, pixel_size=d.PixelSpacing[0])
    e = EyeSol(t.f)
    
    d = Detectibility(nps, ttf, e, t)
    return d