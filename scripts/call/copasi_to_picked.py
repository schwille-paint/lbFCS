# Load modules
import os #platform independent paths
import numpy as np
import importlib
# Load custom modules
import copasi_convert 
importlib.reload(copasi_convert)
#%%
# Define folder of locs_picked.hdf5 file
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/z.simulations/19-06-19_Pm2_2B07/N1_exp30']*2)

file_names=[]
file_names=[]
#file_names.extend(['N48_2_5nM.txt'])
#file_names.extend(['N48_5nM.txt'])
file_names.extend(['N1_10nM_1h_01.txt'])
file_names.extend(['N1_20nM_1h_01.txt'])

# Create full path list
path=[]
for i in range(0, len(file_names)):
    path.append(os.path.join(dir_names[i],file_names[i]))


# Set number of intervals as defined in COPASI
intervals=[120000,120000]
# Set interval_size as defined in COPASI
interval_size=0.03

for i in range(0,np.size(path)):
    arr_all,n_l,n_iter,locs=copasi_convert.copasi2locs(path[i],interval_size,intervals[i])

