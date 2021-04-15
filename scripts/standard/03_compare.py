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
dir_names.extend([r''])


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
exp = 0.4

# query_str = 'setting in [1,2,3,5,8] '
query_str = 'setting in [4,12,13,14,15] '

query_str += 'and success >= 96 '
query_str += 'and abs(frame-M/2)*(2/M) < 0.2 '
query_str += 'and std_frame - 0.8*M/4 > 0 '

query_str += 'and koff > 0.06*@exp '
# query_str += 'and konc*(1e-6/(@exp*vary*1e-12)) > 2.5'
query_str += 'and N < 2.8 '
query_str += 'and N > 0.5 '

# query_str += 'and N > 1.4 '
# query_str += 'and N < 1.4 '


data = obsol.query(query_str)

### Combine
data_combined = solve.combine_obsol(data)
print()
print('Combined solutions:')
solve.print_solutions(data_combined)


####################
'''
Plotting
'''
####################
################### Plot results

f = plt.figure(0,figsize = [5,9])
f.clear()
f.subplots_adjust(bottom=0.1,wspace=0.3)
#################### success
field = 'success'
bins = np.linspace(90,100,100)
ax = f.add_subplot(411)

ax.hist(data[field],
        bins=bins,histtype='step',ec='k')

#################### N
field = 'N'
bins = np.linspace(0,4,60)
ax = f.add_subplot(412)

ax.hist(data[field],
        bins=bins,histtype='step',ec='k')

#################### koff
field = 'koff'
bins = np.linspace(0,0.3,60)
ax = f.add_subplot(413)

ax.hist(data[field]/exp,
        bins=bins,histtype='step',ec='k')

#################### kon
field = 'konc'
bins = np.linspace(0,30,60)
ax = f.add_subplot(414)

ax.hist(data[field]*(1e-6/(exp*data.vary*1e-12)),
        bins=bins,histtype='step',ec='k')


#################### Plot residuals
eps_field = 'eps_direct'

f = plt.figure(1,figsize = [5,3])
f.clear()
ax = f.add_subplot(111)
visualize.residual_violinplot_toax(ax,
                                    solve.compute_residuals(data.query(query_str),eps_field=eps_field),
                                    )
ax.set_ylim(-5,5)
