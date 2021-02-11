import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import picasso.io as io
import lbfcs.solveseries as solve
import lbfcs.visualizeseries as visualize

plt.style.use('~/lbFCS/styles/paper.mplstyle')

#%%
#################### Define directories of solved series
dir_names = []

### N=2, 5xCTC
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-18_N2-5xCTC_cseries/20-12-18_FS_id194_new'])
### N=4, 5xCTC
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-01-28_higherN-5xCTC_cseries/20-01-28_JS_N4_new'])
### N=6, 5xCTC
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-01-28_higherN-5xCTC_cseries/21-01-28_JS_N6_new'])


#################### Load data
### Load obsol and infophotons
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
query_str = 'vary <= 5000 and nn_d > 5'

### Combine
obsol_combined = solve.combine_obsol(obsol)
# print()
# print('Combined solutions:')
# solve.print_solutions(obsol_combined)

### Ensemble solution
obsol_ensemble, obsol_ensemble_combined = solve.get_obsols_ensemble(obsol.query(query_str),[1,1,0,0,0,0,0])
# print()
# print('Ensemble solution:')
# solve.print_solutions(obsol_ensemble_combined)


#################### Plot anything
bins = np.linspace(0,10,70)
# bins = 'fd'

field = 'N'
query_str = 'setting == 6 and vary >= 1250 and nn_d > 5 and N > 0'

f = plt.figure(0,figsize = [4,3])
f.clear()
ax = f.add_subplot(111)
ax.hist(obsol.query(query_str)[field],bins=bins,histtype='step')
ax.hist(obsol_ensemble.query(query_str)[field],bins=bins,histtype='step')


#################### Plot residuals
f = plt.figure(1,figsize = [4,3])
f.clear()
ax = f.add_subplot(111)
visualize.residual_violinplot_toax(ax,
                                                     solve.compute_residuals(obsol.query(query_str)),
                                                     )