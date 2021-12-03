# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 09:53:17 2021

@author: willcx
"""



from medphunc.image_analysis import nnps
from medphunc.image_analysis import image_utility as iu
import scipy
from scipy import ndimage
from skimage import measure

import pandas as pd

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

import numpy as np
from scipy.optimize import curve_fit

from sklearn.covariance import EllipticEnvelope
from scipy.optimize import curve_fit

from typing import List



#%%




def extract_low_variance_patches(imv: np.ndarray) -> List:
    """
    Take an arbitrary 

    Parameters
    ----------
    imv : np.ndarray
        DESCRIPTION.

    Returns
    -------
    List
        DESCRIPTION.

    """
    
    output = []
    
    for i in range(imv.shape[0]):
        im = imv[i,]
        stdmap = iu.window_std(im, 12)
        
        #Create a mask of non-zero std values
        m = (stdmap>0)
        std_nonzeros = stdmap[m]
        
        #cast to int and find the most common value
        std_mode = scipy.stats.mode(std_nonzeros.astype(int)).mode[0]
        
        # Create a mask where std in 12x12 window is around the modal value
        m = (stdmap > std_mode*.65) & (stdmap < std_mode*1.5)
        
        # Pare back the mask a little just in case
        m = ndimage.binary_erosion(m, iterations=2)
        
        labelled_im = measure.label(m)
        measured = measure.regionprops(labelled_im,im)
        
        output = output + measured
    
    # Restrict the output, so that small and low density patches are excluded
    output = [o for o in output if o.mean_intensity > -300]
    output = [o for o in output if o.area > 150]
    return output


def linfit(x,m,c):
    return m*x + c

def detrend_region(im, m=None):
    "detrend a 2d region using a 2nd degree polynomial fit"

    polyreg = make_pipeline(
            PolynomialFeatures(degree=2),
            LinearRegression()
            )
    if m is None:
        m = im==im
    
    y, x = np.where(m)
    z = im[y,x]
    X = np.vstack([y,x]).T
    
    polyreg.fit(X, z)
    
    z_fitted = polyreg.predict(X)
    # Make a copy so we don't overwrite the supplied image data
    im = im.copy()
    # Subtract the fit from the image wherever the mask is True
    im[X[:,0], X[:,1]]-=z_fitted
    return im



def bulk_process_nps_region(regions):
    "Convert a series of noise regions (which are regionprops created from skimage) into noise spectrums"
    
    outs = []
    npss = []
    
    for r in regions:
        m = r.image
        im = r.intensity_image
        
        im = detrend_region(im, m)
        
        b = np.fft.fftshift(np.abs(np.fft.fft2(im))**2)/np.prod(im.shape)
        #b = flapflangle(im,m)
        
        nps = nnps.analyse_nnps(b,1)
        npss.append(nps)
        out = {}
        out['maxnp'] = nps['radial'].max()
        out['meanhu'] = r.mean_intensity
        out['stdval'] = r.intensity_image[r.image].std()
        outs.append(out)
    
    return outs, npss

def filter_clinical_nps_results(df):

    cols = ['meanhu', 'maxnp']
    e = EllipticEnvelope(contamination=.4)
    df['good'] = e.fit_predict(df.loc[:,cols].values)==1
    
    ddf = df.loc[df.good,:]
    return ddf


def automate_clinical_nnps(im, d):
    
    gg = extract_low_variance_patches(im[:,:,:])
    bulk_results = bulk_process_nps_region(gg)
    df=pd.DataFrame(bulk_results[0])
    ddf = filter_clinical_nps_results(df)
    

    p, cov = curve_fit(linfit, ddf.meanhu, ddf.maxnp)
    
    npsdat = [bulk_results[1][i] for i in ddf.index]
        
    fff = pd.DataFrame([n['frequency'] for n in npsdat]).values
    ffff = pd.DataFrame([n['radial'] for n in npsdat]).values
    
    
    freq = np.ma.array(fff,mask=np.isnan(fff))
    sfreq = freq.mean(axis=0)
    below_nyquist = sfreq < 2
    sfreq = sfreq[below_nyquist]
    
    nps = np.ma.array(ffff,mask=np.isnan(ffff))
    nps = nps/ddf.maxnp.values[:,np.newaxis]*p[1]
    snps = np.ma.median(nps,axis=0)
    snps = snps[below_nyquist]
    
    return {'frequency':sfreq, 'radial':snps}