# Script to reduce picked file to picks in props
# In: picked & props
# Out: picked
#
############################################################## Define data paths
# Order must be: props, picked, props, picked, ...

dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p15.RiboCounting/19-12-19_NUP-GFP-Nanobody/FOV1_align']*2) 

file_names=[]
file_names.extend(['200pM_final_picked_props_ig3.hdf5'])
file_names.extend(['200pM_final_picked.hdf5'])


#################################################### Load packages
import os #platform independent paths
import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm

# Load user defined functions
import picasso_addon.io as io

############################################################## Identify & Save
#### Create list of paths
path_props=[os.path.join(dir_names[i],file_names[i]) for i in range(0,len(file_names),2)]
path_locs=[os.path.join(dir_names[i],file_names[i]) for i in np.arange(1,len(file_names),2)]

for i in range(0,len(path_locs)):
    #### Which file is treated
    print(file_names[i])
    
    #### Read in '_picked' and corresponding '_picked_props' file
    locs,info_locs=io.load_locs(path_locs[i])
    props,info_props=io.load_locs(path_props[i])
    
    #### Get groups in props
    groups=props.group.values
    
    #### Predefinition of return value 
    locs_filter=pd.DataFrame(columns=locs.columns,dtype='f4')
    
    #### Reduce groups in picked to groups in props
    for g in tqdm(groups):
        locsg=locs.loc[locs.group==g]
        
        locs_filter=pd.concat([locs_filter,locsg])
    
    
    #### Set running index
    locs_filter.set_index(np.arange(0,len(locs_filter),1),inplace=True)
    #### Set correct dtype
    locs_filter=locs_filter.astype({'frame':'u4','group':'i4'})
        
    #### Save results
    io.save_locs(path_locs[i].replace('picked','picked_identified'),
                         locs_filter,
                         info_locs,
                         mode='picasso_compatible')