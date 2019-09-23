# Load modules
import os #platform independent paths
import numpy as np
import importlib
# Load custom modules
from lbfcs import copasi_convert as copasi_convert
importlib.reload(copasi_convert)
#%%
# Define folder of locs_picked.hdf5 file
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/z.simulations/19-08-20_FS_ralf/lbfcs_copasi']*5)

file_names=[]
file_names.extend(['simulation_10.txt'])
file_names.extend(['simulation_20.txt'])
file_names.extend(['simulation_30.txt'])
file_names.extend(['simulation_40.txt'])
file_names.extend(['simulation_50.txt'])

# Create full path list
path=[]
for i in range(0, len(file_names)):
    path.append(os.path.join(dir_names[i],file_names[i]))


# Set number of intervals as defined in COPASI
intervals=[18000]*5
# Set interval_size as defined in COPASI
interval_size=0.2

for i in range(0,np.size(path)):
    locs,n_l,n_iter=copasi_convert.copasi2locs_ralf(path[i],interval_size,intervals[i])
