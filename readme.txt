Chris' Medical Physics Python Scripts

This package contains a variety of scripts and functions that may find a use within diagnostic imaging departments.

The scope of the package is very broad, ranging from image analysis to facilitating data transfer with PACS.

While these scripts mostly work, they are not necessarily designed for ease of use over ease of development.
Little effort has been put into user interfaces. Documentation is extremely inconsistent.
Feedback and bug reports are desirable, but should be met with no expectation of rapid turnaround.

I recommend that these scripts are used with an anaconda distribution.
To set up an anaconda environment from scratch that is compatible with this package, start with something like the following:
This will probably not be kept up to date.

conda create -n medph python=3.7
conda activate medph
conda install numpy pandas pydicom 
conda install -c conda-forge gdcm dicom2nifti jpeg nibabel opencv
conda install spyder
pip install rdsr-navigator pydicom pynetdicom opencv-log lxml


