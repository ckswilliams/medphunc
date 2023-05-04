Chris' Medical Physics Python Scripts

This package contains a variety of scripts and functions that may find a use within diagnostic imaging departments.

The scope of the package is very broad, ranging from image analysis to facilitating data transfer with PACS.

While these scripts mostly work, they are not necessarily designed for ease of use over ease of development.
Little effort has been put into user interfaces. Documentation is rather inconsistent.
Please send feedback and bug reports as they arise.

I recommend that these scripts are used with an anaconda distribution. Setup instructions from scratch:

Download the package to a directory
Navigate to the directory
Create an environment using:
conda env create --file environment.yml
The default name is gp (short for general purpose). Activate the environment:
conda activate gp
Install the package:
pip install -e .
proceed
