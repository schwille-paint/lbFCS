import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import picasso.io as io
import lbfcs.solveseries as solve

plt.style.use('~/lbFCS/styles/paper.mplstyle')

#%%
#################### Define directories of solved series
dir_names = []

dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-17_N1-2x5xCTC_cseries/20-12-17_FS_id180'])
dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-18_N2-5xCTC_cseries/20-12-18_FS_id194'])
# dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-17_N1-2x5xCTC_cseries/20-12-17_FS_id192_sample2'])


#################### Load data
### Load obsol and info
obsol_paths = [os.path.join(dir_name,'_obsol.hdf5') for dir_name in dir_names]
obsol = pd.concat([pd.DataFrame(io.load_locs(p)[0]) for p in obsol_paths])
obsol_info = [io.load_locs(p)[1] for p in obsol_paths]
### Load file_lists
files_paths = [os.path.join(dir_name,'_files.csv') for dir_name in dir_names]
files = pd.concat([pd.read_csv(p) for p in files_paths])


#%%
####################
'''
Analyze
'''
####################
### Combine obsol and ensemble solving (old&new)
obsol_comb = solve.combine_obsol(obsol)                                     # Combined
obsol_ensemble = solve.get_obsol_ensemble(obsol,[1,1,1,0,0])    # New ensemble solving


### Preselect which measurements are taken into account
settings = [2]
dates = [20201218]

query_str = 'setting in @settings and date in @dates'

subset = obsol.query(query_str )
subset_comb = obsol_comb.query(query_str )
subset_ensemble = obsol_ensemble.query(query_str )

kon = np.mean(subset_ensemble.konc/subset_ensemble.vary)
koff = np.mean(subset_ensemble.koff)
N = (1/subset.A) * (koff/(kon*subset.vary))

bins = np.linspace(0,5,60)

f = plt.figure(0,figsize = [5,4])
f.clear()
ax = f.add_subplot(111)
ax.hist(N,bins=bins,histtype='step',ec='grey')
ax.hist(subset.N,bins=bins,histtype='step',ec='r')

