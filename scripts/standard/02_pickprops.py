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
setting = [12]*4+[4]*3+[1]*3
# Controlled varying experimental variable like temperature in C (21,22,23) or concentration in pM (1000, 5000)
vary    = [10000]*2+[20000]*2+[5000,10000,20000]*2
############################################# Load raw data
dir_names=[]

###################### N=12
# dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-05_N=12/id63_2-5nM_p35uW_1'])
# dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-05_N=12/id63_2-5nM_p35uW_2'])
# dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-05_N=12/id63_5nM_p35uW_1'])
# dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-05_N=12/id63_5nM_p35uW_2'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-05_N=12/id63_10nM_p35uW_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-05_N=12/id63_10nM_p35uW_2'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-05_N=12/id63_20nM_p35uW_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-05_N=12/id63_20nM_p35uW_2'])
###################### N=4
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-05-16_N=4/id125_05nM_p35uW_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-05-16_N=4/id125_10nM_p35uW_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-05-16_N=4/id125_20nM_p35uW_1'])
###################### N=1
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-05-30_SDS_T21/id114_5nM_p35uW_control_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-05-30_SDS_T21/id114_10nM_p35uW_control_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-05-30_SDS_T21/id114_20nM_p35uW_control_1'])


file_names=[]

###################### N=12
# file_names.extend(['id63_2-5nM_p35uW_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
# file_names.extend(['id63_2-5nM_p35uW_2_MMStack_Pos0.ome_locs_render_picked.hdf5'])
# file_names.extend(['id63_5nM_p35uW_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
# file_names.extend(['id63_5nM_p35uW_2_MMStack_Pos0.ome_locs_render_picked.hdf5'])
file_names.extend(['id63_10nM_p35uW_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
file_names.extend(['id63_10nM_p35uW_2_MMStack_Pos0.ome_locs_render_picked.hdf5'])
file_names.extend(['id63_20nM_p35uW_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
file_names.extend(['id63_20nM_p35uW_2_MMStack_Pos0.ome_locs_render_picked.hdf5'])
###################### N=4
file_names.extend(['id125_05nM_p35uW_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
file_names.extend(['id125_10nM_p35uW_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
file_names.extend(['id125_20nM_p35uW_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
###################### N=1
file_names.extend(['id114_5nM_p35uW_control_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
file_names.extend(['id114_10nM_p35uW_control_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
file_names.extend(['id114_20nM_p35uW_control_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])


# dir_name = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-10-09_Simulation-Ringberg/cseries_koff28_kon06'
# paths = glob.glob(os.path.join(dir_name+r'/*.hdf5'))
# paths = [path for path in paths if not bool(re.search('props',path))]

# names = [os.path.split(path)[-1] for path in paths]
# setting = [int(name.split('_')[0][1:]) for name in names]
# vary    = [int(float(name.split('_')[-1].split('.')[0][4:]) * (1e-3/(6.5e6*1e-12))) for name in names]

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
        
