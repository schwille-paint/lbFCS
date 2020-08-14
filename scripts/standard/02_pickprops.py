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

############################################################# Set concentrations
# conc=[5,10,20] # Imager concentration [nM]

############################################# Load raw data
# dir_names=[]
# dir_names.extend([r'C:\Data\p04.lb-FCS\20-07-15_Tutorial']*3)

# file_names=[]
# file_names.extend(['id64_5nM_p35uW_T23_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
# file_names.extend(['id64_10nM_p35uW_T23_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
# file_names.extend(['id64_20nM_p35uW_T23_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])

dir_name=r'C:\Data\p04.lb-FCS\20-06-22_Simulation\scan_run04'
paths=glob.glob(os.path.join(dir_name+r'\*.hdf5'))

paths=[path for path in paths if not bool(re.search('props',path))]
paths=[path for path in paths if bool(re.search('koff52',path))]
paths=[path for path in paths if bool(re.search('konc013',path))]
paths=[path for path in paths if bool(re.search('N20',path))]

conc=[1e-9]*len(paths)

############################################ Set non standard parameters 
### Valid for all evaluations
params_all={'ignore':0,
            }

## Exceptions
params_special={}

############################################# Start dask parallel computing cluster 
# try:
#     client = Client('localhost:8787')
#     print('Connecting to existing cluster...')
# except OSError:
#     props.cluster_setup_howto()
    
#%%                   
failed_path=[]
for i in range(0,len(paths)):
    ### Create path
    # path=os.path.join(dir_names[i],file_names[i])
    
    ### Set paramters for each run
    params=params_all.copy()
    for key, value in params_special.items():
        params[key]=value[i]
    
    ### Run main function
    try:
        locs,info=addon_io.load_locs(paths[i])   
        out=props.main(locs,info,paths[i],conc[i],**params)
    except Exception:
        traceback.print_exc()
        failed_path.extend([paths[i]])

print()    
print('Failed attempts: %i'%(len(failed_path)))
        
