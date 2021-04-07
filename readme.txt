Chris' Medical Physics Python Scripts

This package contains a variety of scripts and functions that may find a use within diagnostic imaging departments.

The scope of the package is very broad, ranging from image analysis to facilitating data transfer with PACS.

While these scripts mostly work, they are not necessarily designed for ease of use over ease of development.
Little effort has been put into user interfaces. Documentation is extremely inconsistent.
Feedback and bug reports are desirable, but should be met with no expectation of rapid turnaround.

I recommend that these scripts are used with an anaconda distribution.
To set up an anaconda environment from scratch that is compatible with this package, start with something like the following:

conda config --add channels conda-forge
conda create --name medph python=3.6 numpy pandas gdcm dicom2nifti jpeg nibabel opencv spyder scikit-image scikit-learn astropy lxml openpyxl xlrd spyder
conda activate medph
pip install rdsr-navigator pynetdicom

