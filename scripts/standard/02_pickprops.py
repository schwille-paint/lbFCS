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
setting = [1]*4
# Controlled varying experimental variable like temperature in C (21,22,23) or concentration in pM (1000, 5000)
vary    = [5000,10000,20000,20000]
############################################# Load raw data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-05-30_SDS_T21/id114_5nM_p35uW_T21_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-05-30_SDS_T21/id114_10nM_p35uW_T21_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-05-30_SDS_T21/id114_20nM_p35uW_T21_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-05-30_SDS_T21/id114_20nM_p35uW_T21_2'])

file_names=[]
file_names.extend(['id114_5nM_p35uW_T21_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
file_names.extend(['id114_10nM_p35uW_T21_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
file_names.extend(['id114_20nM_p35uW_T21_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
file_names.extend(['id114_20nM_p35uW_T21_2_MMStack_Pos0.ome_locs_render_picked.hdf5'])

# dir_name=r'C:\Data\p04.lb-FCS\20-06-22_Simulation\cseries_run02'
# paths=glob.glob(os.path.join(dir_name+r'\*.hdf5'))

# paths=[path for path in paths if not bool(re.search('props',path))]
# paths=[path for path in paths if bool(re.search('koff15',path))]
# paths=[path for path in paths if bool(re.search('konc006',path))]
# paths=[path for path in paths if bool(re.search('N20',path))]

############################################ Set parameters 
params={'ignore':1,
        }

############################################# Start dask parallel computing cluster 
# try:
#     client = Client('localhost:8787')
#     print('Connecting to existing cluster...')
# except OSError:
#     props.cluster_setup_howto()
    
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
        
