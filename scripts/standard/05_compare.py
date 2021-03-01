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

### EGFR, aGFP 1nM for 1h, FOV1, eps_unkown = False
dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-02-25_EGFR-aGFP-controls/w6_03_561_FOV1_Pm2-8nt-c1250_p40uW_1/box5_mng600_pd12_use-eps'])
### EGFR,  aGFP 1nM for 1h, FOV2, eps_unkown = False
dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-02-25_EGFR-aGFP-controls/w6_05_561_FOV2_Pm2-8nt-c1250_p40uW_1/box5_mng600_pd12_use-eps'])
### EGFR,  aGFP 500pM for 30min, FOV4, eps_unkown = False
dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-02-25_EGFR-aGFP-controls/w7_10_561_FOV4_Pm2-8nt-c1250_p40uW_POCGX_1/box_mng600_pd12_use-eps'])

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
v = 1250
exp = 0.4

query_str = 'setting >=1 '
query_str += 'and success >= 96'
# query_str += 'and (eps-eps_direct)/eps_direct < 0.2'
query_str += 'and abs(frame-M/2)*(2/M) < 0.15'
query_str += 'and std_frame - 0.85*M/4 > 0'
query_str += 'and koff > 0.07*0.4 and koff < 0.22*0.4'
query_str += 'and konc*(1e-6/(@exp*vary*1e-12)) > 3'
# query_str += 'and M/(1/konc+1/koff) > 10'

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
####################
field = 'success'
bins = np.linspace(90,100,100)

f = plt.figure(0,figsize = [4,3])
f.clear()
ax = f.add_subplot(111)
ax.hist(data[field],
        bins=bins,histtype='step',ec='k')

####################
field = 'N'
bins = np.linspace(0,7,93)

f = plt.figure(1,figsize = [4,3])
f.clear()
ax = f.add_subplot(111)
ax.hist(data[field]*0.9,
        bins=bins,histtype='step',ec='k')

####################
field = 'koff'
bins = np.linspace(0,0.3,90)

f = plt.figure(2,figsize = [4,3])
f.clear()
ax = f.add_subplot(111)
ax.hist(data[field]/data.exp,
        bins=bins,histtype='step',ec='k')

####################
field = 'konc'
bins = np.linspace(0,50,90)

f = plt.figure(3,figsize = [4,3])
f.clear()
ax = f.add_subplot(111)
ax.hist(data[field]*(1e-6/(data.exp*data.vary*1e-12)),
        bins=bins,histtype='step',ec='k')


#################### Plot residuals
# eps_field = 'eps_direct'
# if params['eps_unknown']: eps_field = 'eps'

# f = plt.figure(1,figsize = [4,3])
# f.clear()
# ax = f.add_subplot(111)
# visualize.residual_violinplot_toax(ax,
#                                     solve.compute_residuals(obsol.query(query_str),eps_field=eps_field),
#                                     )
# ax.set_ylim(-3,3)
