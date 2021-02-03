import os
import traceback
from dask.distributed import Client
import multiprocessing as mp

import picasso_addon.io as addon_io
import lbfcs.pickprops as props

############################################################# Which experiment?
# Experimental setting,  like i.e constant variables like labeling, origami design etc.
setting = [12]*5
# Controlled varying experimental variable like temperature in C (21,22,23) or concentration in pM (1000, 5000)
vary    = [5000,2500,1250,625,313]

############################################# Load raw data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-01-22_Simulation/5xCTC_exp400_T23_kon30e6_4M_new_CORR1_nock/N12']*5)

file_names=[]
file_names.extend(['N12_c5000.hdf5'])
file_names.extend(['N12_c2500.hdf5'])
file_names.extend(['N12_c1250.hdf5'])
file_names.extend(['N12_c0625.hdf5'])
file_names.extend(['N12_c0313.hdf5'])


############################################ Set parameters 
params={'ignore':1,
                'parallel':True}

############################################# Start dask parallel computing cluster 
try:
    client = Client('localhost:8787')
    print('Connecting to existing cluster...')
except OSError:
    props.cluster_setup_howto()
    
#%%
              
failed_path = []
paths       = [ os.path.join(dir_names[i],file_names[i]) for i in range(len(file_names)) ]
conditions  = [(setting[i],vary[i]) for i in range(len(paths))]

for i, path in enumerate(paths):
    try:
        locs,info=addon_io.load_locs(path)
        out=props.main(locs,info,path,conditions[i],**params)
    except Exception:
        traceback.print_exc()
        failed_path.extend([path])

print()    
print('Failed attempts: %i'%(len(failed_path)))
        
