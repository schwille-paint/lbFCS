'''
Script to sum normalized traces of individual picks 
to generate picks with higher N
'''
import os

import lbfcs.analyze as analyze
import picasso.io as io
import picasso_addon.io as addon_io

#%%
dir_names = []
file_names = []


#################### Define path to _props file
dir_names.extend([r'C:\Data\p17.lbFCS2\21-07-13_sim'])
file_names.extend(['N01_c5000_picked_props.hdf5'])


#################### Define path to _picked file
dir_names.extend([r'C:\Data\p17.lbFCS2\21-07-13_sim'])
file_names.extend(['N01_c5000_picked.hdf5'])


#################### Define filter criteria
query_str = 'abs(frame-M/2)*(2/M) < 0.2 '
query_str += 'and std_frame - 0.8*M/4 > 0 '
query_str += 'and success >= 99 '
query_str += 'and N > 0.9 and N < 1.2 '
query_str += 'and occ > 0.4'


#################### Summation parameters
n = 1000
ks = [2,4,6]


#################### Saving
savedir = r'C:\Data\p17.lbFCS2\21-07-13_sim'

#%%
### Load _props and _picked
paths = [os.path.join(dir_names[i],file) for i,file in enumerate(file_names)]
props, info_props = addon_io.load_locs(paths[0])
picked, info_picked = addon_io.load_locs(paths[1])

### Query _props
props = props.query(query_str)

### Main loop: Sum & Save
for k in ks:
    ### Sum picks
    picked_sum = analyze.sum_normalized_picks(picked,props,n,k)
    
    ### Saving
    params = {'from_dir':dir_names[1],
              'from_file':file_names[1],
              'query_str':query_str,
              'n':n,
              'k':k,
              }
    info = info_picked.copy()+[params]
    io.save_locs(os.path.join(savedir,'sum_k%i_picked.hdf5'%k),
                 picked_sum.to_records(index=False),
                 info,
                 )


