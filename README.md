# lbFCS

## Description
Python package to evaluate [DNA-PAINT][paint] SMLM data via autocorrelation analysis for self-calibrating counting of molecules (according to ['Self-calibrating molecular counting
with DNA-PAINT'][paper]). 

![principle](/docs/figures/principle.png)

## Table of contents
* [Installation](#installation)
* [Usage](#usage)

## Installation
1. Create a [conda][conda] environment ``conda create --name lbFCS python=3.5``

2. Activate the environment ``conda activate lbFCS``

3. Install necessary packages 
    * ``conda install h5py matplotlib numba numpy scipy pyqt=4 pyyaml scikit-learn colorama tqdm spyder pandas dask spyder fastparquet pytables``
    * ``pip install lmfit``
    * ``pip install jupyterlab``

4. [Clone](https://help.github.com/en/articles/cloning-a-repository) the lbFCS repository

3. Let iPython know where it can find the lbFCS repository (for Jupyter Notebook or Spyder)
   * Browse to ~/.ipython/profile_default in your home directory
   * Create a file called startup.py with the following content
   ```
   import sys
   sys.path.append('Path to /lbFCS')
   ```
4. Install the [picasso GUI](https://github.com/jungmannlab/picasso) (standard installation).

## Usage
The following is a short guide through the necessary processing steps from DNA-PAINT raw-data to the final result according to ['Self-calibrating molecular counting
with DNA-PAINT'][paper]. For more information on the imaging modalities or sample preparation please refer to the reference. 

### 1. Localize
Localize DNA-PAINT raw-data aquired under low exitation intensities (approx. 10 kW/cm^2 with a frame rate of 5Hz used imager was labeled with Cy3B) using the [picasso.localize](https://picassosr.readthedocs.io/en/latest/localize.html) module. 
The following picture shows the standard parameters used and the typical look of our raw data. 
![localize](/docs/figures/localize.png)

The shown paramters apply to the standard conditions of our setup & sample and might hence differ under other circumstances. In general the follwing rules should apply for the selections of localization parameters with picasso.localize
   * Set the box size to the minimum length in which the PSF of your microscope fits in 
   * Set the minimum net gradient low enough so that every spot that can be detected by eye is also detected by picasso.localize.  
   * Set the correct photon conversion parameters for your camera
    
Afer you adjusted all the parameters localize by choosing >Analyze>Localize. A '_locs.hdf5' file is created in the same folder as the raw-data that is used for further analysis.

### 2. Undrift
Undrift the '_locs.hdf5' file using the [picasso.render ](https://picassosr.readthedocs.io/en/latest/render.html) module.
Therefore drag and drop the '_locs.hdf5' file into the render GUI and choose >Postprocess>Undrift by RCC with a segmentation of 500. 
![undrift](/docs/figures/undrift.png)

An undrifted '_locs_render.hdf5' file is created in the same folder as the raw-data that is used for further analysis.

### 3. Autopick
Go to the [autopick notebook](/scripts/notebooks/autopick.ipynb)

### 4. Render: Picked

### 5. Pickprops
Go to the [pickprops&filter notebook](/scripts/notebooks/pickprops&filter.ipynb)


[paint]:https://www.nature.com/articles/nprot.2017.024
[paper]:http://not-known-yet.com
[conda]:https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
