# Import modules
import os
import importlib


#### Define path to file
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-25_c-series_N12_long/id63_5nM_p35uW_1/19-06-25_JS'])
#dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-25_c-series_N12_long/id63_10nM_p35uW_1/19-06-25_JS'])
#dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-25_c-series_N12_long/id63_20nM_p35uW_1/19-06-25_PS'])

file_names=[]
file_names.extend(['id63_5nM_p35uW_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
#file_names.extend(['id63_10nM_p35uW_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
#file_names.extend(['id63_20nM_p35uW_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
#### Create list of paths
path=[os.path.join(dir_names[i],file_names[i]) for i in range(0,len(file_names))]

#%%
#### Segment
import pickprops_calls as props_call
# Reload modules
importlib.reload(props_call)

noFrames_seg=9000
for p in path:
    print(p)
    props_call.segment_time(p,noFrames_seg)