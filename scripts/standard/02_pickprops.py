import os
import traceback
from dask.distributed import Client
import multiprocessing as mp

import picasso_addon.io as addon_io
import lbfcs.pick_combine as props

############################################################# Used imager concentrations in pM
cs = [2500]*5

############################################# Load raw data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-21_EGFR_2xCTC/09_w5-250pM_Cy3B-CTween-c2500_561-p50uW-s75_Pos0-4_1/Pos0_pd30'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-21_EGFR_2xCTC/09_w5-250pM_Cy3B-CTween-c2500_561-p50uW-s75_Pos0-4_1/Pos1_pd30'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-21_EGFR_2xCTC/09_w5-250pM_Cy3B-CTween-c2500_561-p50uW-s75_Pos0-4_1/Pos2_pd30'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-21_EGFR_2xCTC/09_w5-250pM_Cy3B-CTween-c2500_561-p50uW-s75_Pos0-4_1/Pos3_pd30'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-21_EGFR_2xCTC/09_w5-250pM_Cy3B-CTween-c2500_561-p50uW-s75_Pos0-4_1/Pos4_pd30'])

file_names=[]
file_names.extend(['09_w5-250pM_Cy3B-CTween-c2500_561-p50uW-s75_Pos0-4_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
file_names.extend(['09_w5-250pM_Cy3B-CTween-c2500_561-p50uW-s75_Pos0-4_1_MMStack_Pos1.ome_locs_render_picked.hdf5'])
file_names.extend(['09_w5-250pM_Cy3B-CTween-c2500_561-p50uW-s75_Pos0-4_1_MMStack_Pos2.ome_locs_render_picked.hdf5'])
file_names.extend(['09_w5-250pM_Cy3B-CTween-c2500_561-p50uW-s75_Pos0-4_1_MMStack_Pos3.ome_locs_render_picked.hdf5'])
file_names.extend(['09_w5-250pM_Cy3B-CTween-c2500_561-p50uW-s75_Pos0-4_1_MMStack_Pos4.ome_locs_render_picked.hdf5'])

############################################ Set parameters 
params={'parallel': True}

############################################# Start dask parallel computing cluster 
try:
    client = Client('localhost:8787')
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
