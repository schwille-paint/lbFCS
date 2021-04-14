import os
import traceback
from dask.distributed import Client
import multiprocessing as mp

import picasso_addon.io as addon_io
import lbfcs.pick_combine as props

############################################################# Which experiment?
# Experimental setting,  like i.e constant variables like labeling, origami design etc.
setting = [4]
# Controlled varying experimental variable like temperature in C (21,22,23) or concentration in pM (1000, 5000)
vary = [5000]

############################################# Load raw data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-01-28_higherN-5xCTC_cseries/21-04-13_N4_JS_test'])

file_names=[]
file_names.extend(['id200_5000pM_p40uW_exp400_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])


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

#%%
import matplotlib.pyplot as plt
import numpy as np

data_init = out[1]
data = data_init.copy()


success = 99
exp = 0.4

#################### Evaluation parameters
query_str = 'success >= @success '
query_str += 'and abs(frame-M/2)*(2/M) < 0.2 '
query_str += 'and std_frame - 0.8*M/4 > 0 '
query_str += 'and N <  7.1'
query_str += 'and abs(1-eps_normstat) < 0.1'

data = data.query(query_str)


############# N = 4
bins = np.linspace(0.2,6.5,65)
field = 'N'

f = plt.figure(2,figsize = [4,3])
f.clear()
ax = f.add_subplot(111)
query_str = 'setting == 4'
ax.hist(data.query(query_str)[field],
        bins=bins,histtype='step',lw=1.5,
        ec='tomato',label=r'%i'%len(data.query(query_str)[field]),
        )

ax.legend()
ax.set_xlabel('N')
ax.set_xticks(range(7))
ax.set_xlim(0.5,6.5)
ax.set_ylabel('Occurences')