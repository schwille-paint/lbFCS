# lbFCS

## Description
Python package to evaluate [DNA-PAINT][paint] SMLM data via autocorrelation analysis for self-calibrating counting of molecules (according to ['Self-calibrating molecular counting
with DNA-PAINT'][paper]). 

![principle](/docs/figures/principle.png)

## Table of contents
* [Installation](#installation)
* [Usage](#usage)

## Installation
1. Create a [conda][conda] environment with the following commands:
```
conda create --name lbFCS python=3.5
conda activate lbFCS
conda install h5py matplotlib numba numpy scipy pyqt=4 pyyaml scikit-learn colorama tqdm spyder pandas dask spyder fastparquet pytables
pip install lmfit
pip install jupyterlab
```

2. Clone the lbFCS repository from github

3. Let iPython know where it can find the lbFCS repository (for Jupyter Notebook or Spyder)
   * Browse to ~/.ipython/profile_default in your home directory
   * Create a file called startup.py with the following content
   ```
   import sys
   sys.path.append('Path to /lbFCS')
   ```

## Usage



[paint]:https://www.nature.com/articles/nprot.2017.024
[paper]:http://not-known-yet.com
[conda]:https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
