# lbFCS

## Description
Python package to evaluate [DNA-PAINT][paint] SMLM data via autocorrelation analysis for self-calibrating counting of molecules 
(according to ['Towards absolute molecular numbers in DNA-PAINT'][paper]). It requires the installation of the [picasso](https://github.com/jungmannlab/picasso) package.

** !Important!: To use the published version of the package, please refer to the 'published' branch of the repository**

<img src="/docs/figures/principle.png" alt="principle" width="700">

## Table of contents
* [Installation](#installation)
* [Usage](#usage)
* [Remarks](#remarks)

## Installation

We will first prepare the necessary conda environment:
1. Create a [conda][conda] environment ``conda create --name lbFCS python=3.7``
2. Activate the environment ``conda activate lbFCS``
3. Install necessary packages 
    * ``conda install h5py matplotlib numba numpy scipy pyqt pyyaml scikit-learn colorama tqdm=4.36.1 spyder pandas dask spyder fastparquet pytables jupyterlab``
    * ``pip install lmfit``


Since lbFCS uses some core modules of the [picasso](https://github.com/jungmannlab/picasso) package we will next install picasso into our environment
1. [Clone](https://help.github.com/en/articles/cloning-a-repository) the [picasso](https://github.com/jungmannlab/picasso) repository
2. Switch to the cloned folder ``cd picasso``
3. Install picasso into the environment ``python setup.py install``

Finally we install lbFCS into our environment

1. Leave the picasso directory ``cd ..``
2. [Clone](https://help.github.com/en/articles/cloning-a-repository) the [picasso](https://github.com/schwille-paint/lbFCS) repository
3. Switch to the cloned folder ``cd lbFCS``
4. Install lbFCS into the environment ``python setup.py install``



## Usage
The following is a short guide through the necessary processing steps from DNA-PAINT raw-data to the final result according to ['Towards absolute molecular numbers in DNA-PAINT'][paper]. For more information on the imaging modalities or sample preparation please refer to the reference. 


1. Localize the raw movie and perform drift correction [(localize_undrift notebook)](/scripts/notebooks/01_localize_undrift.ipynb)

2. Automated localization cluster detection [(autopick notebook)](/scripts/notebooks/02_autopick.ipynb)

3. Kinetic analysis and of localization clusters and automated filtering based on kinetic properties [(pickprops_filter notebook)](/scripts/notebooks/03_pickprops_filter.ipynb)

4. Final counting of molecular numbers and hybridization kinetics via concentration series evaluation [(c-series notebook)](/scripts/notebooks/04_c-series.ipynb)


### Remarks
Sometimes the notebooks cannot be rendered by GitHub. Try the following work-around:
1. Open https://nbviewer.jupyter.org/
2. Paste the link to the notbook on GitHub, e.g. https://github.com/schwille-paint/lbFCS/blob/master/scripts/notebooks/autopick.ipynb 

[paint]:https://www.nature.com/articles/nprot.2017.024
[paper]: https://pubs.acs.org/doi/abs/10.1021/acs.nanolett.9b03546
[conda]:https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
