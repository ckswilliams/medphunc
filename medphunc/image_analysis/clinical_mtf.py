# -*- coding: utf-8 -*-
"""
Functions for calculating the MTF of arbitrary CT images.

Created on Fri Jan 24 15:04:56 2020



@author: Chris Williams
"""

from matplotlib import pyplot as plt
import numpy as np


import pydicom

from medphunc.image_io.ct import rescale_ct_dicom
from medphunc.image_analysis import image_utility as iu
from skimage import measure
from scipy import fft


import logging

#%%
log_verbosity = 5


def log_plot(x, y, priority=2, **kwargs):
    if priority <= log_verbosity:
        fig, ax = plt.subplots()
        ax.plot(x, y, **kwargs)
        fig.show()

#%%


def smooth_esf(line_profiles, max_badbuffer=120, min_badbuffer=25):
    """
    Smooth an array of ESFs by flattening the top and bottom of the curves.
    
    1) Set any values before the maximum or after the minimum to those values
    2) generate an index array corresponding to ESFs that contain too low/too
    high values before/after the max/min respectively
    (i.e. dips before the edge, or peaks after the edge)

    Parameters
    ----------
    line_profiles : np.array
         2d array of edge functions. Each row corresponds to one line
    max_badbuffer : int, optional
        Any ESF with a minima before the first maxima of more than this is 'bad'. The default is 120.
    min_badbuffer : int, optional
        Any ESF with a maxima after the minimum value more than this is 'bad'. The default is 25.

    Returns
    -------
    line_profiles : np.array
        Array of line profiles.
    bad_profile_indices : np.array
        array of indexes corresponding to bad line profiles, which can be used
        to filter the data.

    """
    #max_badbuffer = 120
    #min_badbuffer = 25
    line_profiles = line_profiles.copy()
    wm = np.where(np.ones(line_profiles.shape))

    prof_argmax = line_profiles.argmax(axis=1)[wm[0]]
    prof_argmin = line_profiles.argmin(axis=1)[wm[0]]

    prof_max = line_profiles[wm[0], prof_argmax]
    prof_min = line_profiles[wm[0], prof_argmin]

    prof_vals = line_profiles[wm]

    left_max = wm[1] < prof_argmax
    right_min = wm[1] > prof_argmin

    # any values below the cutoff and left of the max
    # any values above the lower cutoff and right of the min
    # need to be eliminated!
    bad_profiles = (((prof_vals < (prof_max-max_badbuffer)) & left_max) +
                    (((prof_vals > (prof_min+min_badbuffer))) & right_min))

    bad_profile_indices = np.unique(wm[0][bad_profiles])

    # all values left of the max and all values right of the min
    # should be set to max/min respectively

    line_profiles[wm[0][left_max], wm[1][left_max]] = prof_max[left_max]
    line_profiles[wm[0][right_min], wm[1][right_min]] = prof_min[right_min]

    return line_profiles, bad_profile_indices

#%%


def clinical_mtf(im, pixel_spacing, profile_pixel_length=20, profile_pixel_spacing=0.5,
                 step_size=8,
                 # min above, max above, max below
                 magnitude_requirement=(-975, -50, 100),
                 bad_extrema_buffer=(120, 25),
                 threshold=-300,
                 high_hu_object=True):
    """
    Calculate the MTF for clinical volumetric patient images.
    
    Calculate MTF using the method suggested by Sanders et al.
    https://aapm.onlinelibrary.wiley.com/doi/pdf/10.1118/1.4961984
    with a little extra special sauce

    Parameters
    ----------
    im : np.array
        2d or 3d axial CT image. im[Z, Y, X] 
    pixel_spacing : tuple
        pixel size of (z, y, x) in mm.
    profile_pixel_length : int, optional
        Number of pixels each line profile should be. The default is 20.
    profile_pixel_spacing : TYPE, optional
        How frequently to sample for the line. Lowering this value reduces averaging. The default is 0.5.
    step_size : int, optional
        Used for generating the 3d mesh. See scipy.measure.marching_cubes_lewinar.
        Strongly affects processing time. The default is 8, which is appropriate
        for 512x512 volumes. For small subvolumes, the algorithm may fail unless
        reduced to 1.
    magnitude_requirement : tuple(int, int, int), optional
        Magnitude requirements(Min below, max above, max below). The default is (-975, -50, 100).
        In the future, this can be ignored by supplying None.
    bad_extrema_buffer : tuple(int, int), optional
        Not used. The default is (120,25).
    threshold : numeric, optional
        determines what counts as internal vs external to the object. 
        The default is -300, which is a reasonable discriminator between air/any tissue
    invert_threshold : boolean, optional
        False if the object is greater in HU than the background (i.e. patient in air)
        True if the object is lesser in HU (i.e. supplied cropped contrast object
                                            in a phantom)

    Returns
    -------
    output : dict
        Dictionary containing the key 'clinical', which contains 'MTF' and 'frequency'
        dict('Clinial':{'frequency':frequency, 'MTF:mtf}).

    """

    # m = im > threshold
    # if invert_threshold:
    #     m = ~m
    if high_hu_object:
        gradient_direction = 'descent'
    else:
        gradient_direction = 'ascent'

    # Create a mesh from the image
    verts, faces, norms, val = measure.marching_cubes(
        im,
        level=threshold,
        gradient_direction=gradient_direction,
        #spacing=None,
        #level=0.5,
        step_size=step_size,
        allow_degenerate=True
    )

    # Get the start and finish of each possible line profile going through
    # one of the mesh vertices
    p0s = verts-norms*profile_pixel_length/2
    p1s = verts+norms*profile_pixel_length/2

    # Calculate the angle between the
    angles = np.array([np.arccos(np.dot(n, [1, 0, 0]))
                       for n in norms])/np.pi*180
    angle_good = (angles > 70) & (angles < 110)

    #detect any points creating lines outside the image volume
    boundry_bad = ((p0s < 0) +
                   (p1s < 0) +
                   ((p0s-im.shape) > 0) +
                   ((p1s-im.shape) > 0)
                   ).any(axis=1)

    # degenerate means ??
    degenerate = ((norms**2).sum(axis=1) < 0.98)

    baddies = boundry_bad + ~angle_good + degenerate

    pixel_spacing = np.array(pixel_spacing)
    line_pixel_length = ((pixel_spacing*norms)**2).sum(axis=1)**0.5

    mesh_index = np.where(~baddies)[0]

    # make all the line profiles
    line_profiles = points_to_line_profiles(
        im, p0s[mesh_index], p1s[mesh_index], profile_pixel_spacing, profile_pixel_length)

    line_profiles = np.array(line_profiles)
    if not high_hu_object:
        line_profiles = line_profiles[:, ::-1]
    mesh_index = np.array(mesh_index)

    # Catch magnitude issues
    good_magnitude = ((line_profiles.max(axis=1) < magnitude_requirement[2]) &
                      (line_profiles.max(axis=1) > magnitude_requirement[1]) &
                      (line_profiles.min(axis=1) < magnitude_requirement[0]))

    line_profiles = line_profiles[good_magnitude]
    mesh_index = mesh_index[good_magnitude]

    line_profiles, bad_profile_indices = smooth_esf(line_profiles)

    line_profiles = np.delete(line_profiles, bad_profile_indices, axis=0)
    mesh_index = np.delete(mesh_index, bad_profile_indices, axis=0)
    line_pixel_length = line_pixel_length[mesh_index]

    #Get MTFs from ESFs
    mtfs = iu.mtf_from_esf(line_profiles)
    #Normalise to max value
    mtfs = mtfs / mtfs.max(axis=1)[:, None]

    freqs = iu.spatial_frequency_for_mtf(
        line_profiles.shape[1], line_pixel_length*profile_pixel_spacing).T

    freq, mtf = np.median(freqs, axis=1), np.median(mtfs, axis=0)

    output = {'Clinical': {'MTF': mtf,
                           'frequency': freq}}

    return output


def points_to_line_profiles(im, p0s, p1s, profile_pixel_spacing, profile_pixel_length):
    line_profiles = []
    for p0, p1 in zip(p0s, p1s):
        line = iu.profile_line(
            im, p0, p1, spacing=profile_pixel_spacing, order=1, endpoint=False)
        line_profiles.append(
            line[:int(profile_pixel_length/profile_pixel_spacing)])
    return line_profiles


def clinical_mtf_from_dicom_metadata(im, d, **kwargs):
    "Rough and ready method for calculating the clinical MTF for a patient volume and dicom header"
    pixel_spacing = np.array([d.SliceThickness, *d.PixelSpacing])
    return clinical_mtf(im, pixel_spacing, **kwargs)


#%%
def old_clinical_mtf(im, pixel_spacing, roi_size=(50, 30)):
    """
    Calculate the MTF from a patient/phantom image.
    
    Requires air on the side of
    the patient corresponding to the zero direction of the y axis.
    
    Only use this function for 2d images, which are not currently covered by
    new clinical MTF

    Parameters
    ----------
    im : np.array
        2d or 3d axial CT image. im[Z, Y, X] 
    pixel_spacing : tuple
        pixel size of (y, x) in mm.
    roi_size : tuple,
        Size of the ROI that the MTF will be calculated across. The default is (50, 30).

    Returns
    -------
    output : dictionary
        Dictionary containing the key 'clinical', which contains 'MTF' and 'frequency'

    """
    if im.ndim == 2:
        im = im[np.newaxis, ]

    mtfs = []
    fs = []

    for i in range(im.shape[0]):
        im_2d = im[i, :]

        #find head
        try:
            seg_data = iu.localise_phantom(im_2d, -300)
            if seg_data['anterior_point'][0] < 15:
                im_2d = np.pad(im_2d, 15, mode='constant',
                               constant_values=-1024)
                seg_data = seg_data = iu.localise_phantom(im_2d, -300)
        except:
            continue
        y = seg_data['anterior_point'][0]
        x = seg_data['anterior_point'][1]

        roi = iu.extract_patch_around_point(im_2d, (y, x), roi_size)
        #log.image(log.Level.TRACE, iu.apply_window(roi))

        # find the index of array element with the highest rate of change
        roi = iu.align_line_profiles(roi, -500)

        w = 10
        c = roi.shape[0]//2

        esf = roi[c-w:c+w, :].mean(axis=1)

        #log_plot(range(len(esf)), esf)

        mtf = iu.mtf_from_esf(esf)
        mtf = mtf/mtf[0]
        f = pixel_spacing[0]/2*np.arange(mtf.shape[0])
        mtfs.append(mtf)
        fs.append(f)

    mtf = np.array(mtfs)
    mtf = np.median(mtf, axis=0)
    f = np.array(fs)
    f = np.median(f, axis=0)

    output = {'Clinical': {'MTF': mtf,
                           'frequency': f}}

    return output


def old_clinical_mtf_from_dicom(fn):
    """Calculate the clinical MTF from a dicom file, across the patient/air anterior border."""
    d = pydicom.read_file(fn)
    im = rescale_ct_dicom(d)
    return clinical_mtf(im, d.PixelSpacing)

#%%


if __name__ == '__main__':
    #Testing for
    fn = 'images/catphan/'
    from medphunc.image_io import ct
    im, d, stuff = ct.load_ct_folder(fn)
    results = clinical_mtf(im, [d.SliceThickness, *d.PixelSpacing])

    im, np.hstack([d.SliceThickness, d.PixelSpacing])
    iu.plot_mtf(results)
    results = clinical_mtf(im, d.PixelSpacing)
    iu.plot_mtf(results)

    fn = 'images/catphan1.dicom'
    results = old_clinical_mtf_from_dicom(fn)
    iu.plot_mtf(results)
