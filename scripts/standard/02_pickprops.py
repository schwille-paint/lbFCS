'''
Script to analyze localization traces for each localization cluster (i.e. pick).
Observables for each cluster are computed and based on these a solution for N, k_off and k_on is found.
'''
import os
import traceback
from dask.distributed import Client
import multiprocessing as mp

import picasso_addon.io as addon_io
import lbfcs.pick_combine as props

############################################################# Used imager concentrations in pM
cs = [5000]*3

############################################# Load raw data
dir_names=[]
dir_names.extend([r'C:\Data\p17.lbFCS2\21-07-13_sim']*3)

file_names=[]
file_names.extend(['sum_k2_picked.hdf5'])
file_names.extend(['sum_k4_picked.hdf5'])
file_names.extend(['sum_k6_picked.hdf5'])

############################################ Set parameters 
params={'parallel': False}

############################################# Start dask parallel computing cluster 
# try:
#     client = Client('localhost:8787')
#     print('Connecting to existing cluster...')
# except OSError:
#     props.cluster_setup_howto()

#%%

failed_path = []
paths = [ os.path.join(dir_names[i],file_names[i]) for i in range(len(file_names)) ]

for i, path in enumerate(paths):
    try:
        locs,info=addon_io.load_locs(path)
        out=props.main(locs,info,path,cs[i],**params)
    except Exception:
        traceback.print_exc()
        failed_path.extend([path])

print()    
print('Failed attempts: %i'%(len(failed_path)))
