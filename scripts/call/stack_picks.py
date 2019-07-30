#################################################### Load packages
import os #platform independent paths
import importlib
# Load user defined functions
import var_io
import pickprops_calls as props_call
importlib.reload(props_call)

############################################################## Define number of groups that are combined into single one
N=[2]
compress=False
############################################################## Define data paths
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/z.simulations/19-06-19_Pm2_2B07/N1_exp30']*2)

file_names=[]
file_names.extend(['N1_2-5nM_01_locs_picked_props_ig0.hdf5'])
file_names.extend(['N1_5nM_01_locs_picked_props_ig0.hdf5'])

############################################################## Define
#### Create list of paths
path_locs=[os.path.join(dir_names[i],file_names[i].replace('picked_props_ig0','picked')) for i in range(0,len(file_names))]
path_props=[os.path.join(dir_names[i],file_names[i]) for i in range(0,len(file_names))]

for i in range(0,len(path_locs)):
    #### Read in '_picked' and corresponding '_picked_props' file
    locs,info_locs=var_io.read_locs(path_locs[i])
    props,info_props=var_io.read_locs(path_props[i])
    
    for n in N:
        #### Stack
        locs_combine,info_combine=props_call.combine_picks(locs,info_locs,props,n,compress)
        #### Save
        savename_ext='_stack%i'%(n)
        var_io.save_locs(locs_combine,info_combine,path_locs[i],savename_ext=savename_ext)

