#Script to call spt.immobile_props.main()
import os
import glob
import re
import traceback
import importlib
import numpy as np
from dask.distributed import Client
import multiprocessing as mp

import picasso_addon.io as addon_io
import lbfcs.pickprops as props

importlib.reload(props)

############################################################# Which experiment?
# Experimental setting,  like i.e constant variables like labeling, origami design etc.
setting = [12]*4
# Controlled varying experimental variable like temperature in C (21,22,23) or concentration in pM (1000, 5000)
vary    = [10000,10001,10100,10101]

############################################# Load raw data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-11-04_N12_T23_Buffer-Repeat/20-11-04_FS_L']*4)

file_names=[]
file_names.extend(['id45-01_Pm2-10nM-L_p40uW_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
file_names.extend(['id45-01_Pm2-10nM-L_p40uW_2_MMStack_Pos0.ome_locs_render_picked.hdf5'])
file_names.extend(['id45-02_Pm2-10nM-L_p40uW_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
file_names.extend(['id45-02_Pm2-10nM-L_p40uW_2_MMStack_Pos0.ome_locs_render_picked.hdf5'])


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
importlib.reload(props)                 
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
        
