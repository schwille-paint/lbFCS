# lbFCS

## Description
Python package to evaluate [DNA-PAINT][paint] SMLM data via autocorrelation analysis for self-calbrating counting of molecules (according to ['Self-calibrating molecular counting
with DNA-PAINT'][paper]). 

## Installation
1. Create a [conda][conda] environment with the following commands:
```
conda create --name lbFCS python=3.5
conda install h5py matplotlib numba numpy scipy pyqt=4 pyyaml scikit-learn colorama tqdm spyder pandas dask spyder fastparquet pytables
pip install lmfit
```


* How to create path to package so that ipython can find it




[paint]:https://www.nature.com/articles/nprot.2017.024
[paper]:http://not-known-yet.com
[conda]:https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
