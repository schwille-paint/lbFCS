import os
import traceback
from dask.distributed import Client
import multiprocessing as mp

import picasso_addon.io as addon_io
import lbfcs.pickprops as props

############################################################# Which experiment?
# Experimental setting,  like i.e constant variables like labeling, origami design etc.
setting = [40,41,42]
# Controlled varying experimental variable like temperature in C (21,22,23) or concentration in pM (1000, 5000)
vary = [5000]*3

############################################# Load raw data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-08_origami-checks/12_s5_5000pM_p30uW_exp400_FOVs_1/21-04-09_FS']*3)


file_names=[]
file_names.extend(['12_s5_5000pM_p30uW_exp400_FOVs_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
file_names.extend(['12_s5_5000pM_p30uW_exp400_FOVs_1_MMStack_Pos1.ome_locs_render_picked.hdf5'])
file_names.extend(['12_s5_5000pM_p30uW_exp400_FOVs_1_MMStack_Pos2.ome_locs_render_picked.hdf5'])

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

