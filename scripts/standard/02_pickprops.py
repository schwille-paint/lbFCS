import os
import traceback
from dask.distributed import Client
import multiprocessing as mp

import picasso_addon.io as addon_io
import lbfcs.pick_combine as props

############################################################# Used imager concentrations in pM
cs = [1500]

############################################# Load raw data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-15_NUPs_a20/02_s2w6_c1500_561-p40uW-s23_FOV2_1'])

file_names=[]
file_names.extend(['02_s2w6_c1500_561-p40uW-s23_FOV2_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])


############################################ Set parameters 
params={'ignore':1,
        'parallel': True}

############################################# Start dask parallel computing cluster 
try:
    client = Client('localhost:8786')
    print('Connecting to existing cluster...')
except OSError:
    props.cluster_setup_howto()

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
