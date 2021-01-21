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

### Old Pm2@200ms & T=23C
# dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/z.olddata/N1/20-01-18_JS'])
# dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/z.olddata/N4/20-01-18_JS'])
# dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/z.olddata/N12/20-01-18_JS'])

### Old Pm2@200ms & T=21C
# dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/z.olddata/N1_T21/20-01-19_FS'])

### New Pm2 @200ms & T=23C
# dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-09_N1_T23_ibidi_cseries/21-01-19_FS_id181'])

### 5xCTC@200ms&400ms & T=23C
dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-09_N1_T23_ibidi_cseries/21-01-19_FS_id180'])
dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-10_N1-5xCTC_cseries_varexp/21-01-19_FS_id180_exp200'])
dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-10_N1-5xCTC_cseries_varexp/21-01-19_FS_id180_exp400'])
dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-17_N1-2x5xCTC_cseries/20-12-17_FS_id180'])

### N != 1, 5xCTC@400ms & T=23C
# dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-18_N2-5xCTC_cseries/20-12-18_FS_id194'])

### S1_5xCTC adapter @200ms&400ms & T=23C
# dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-10_N1-5xCTC_cseries_varexp/21-01-19_FS_id163_exp200'])
# dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-10_N1-5xCTC_cseries_varexp/21-01-19_FS_id163_exp400'])


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
obsol_comb = solve.combine_obsol(obsol) 

###################### Preselect which measurements are taken into account
varies = np.unique(obsol.vary)

query_str = 'vary in @varies and N < 1.2 and N > 0.7'

sub = obsol.query(query_str)
sub_comb = solve.combine_obsol(sub)                                                  # Combined solutions
sub_ensemble = solve.get_obsol_ensemble(sub,weights = [np.nan])   # Old fitting approach
sub_old = solve.N_via_ensemble(sub,sub_ensemble)                            # Assign koff,kon&N based on old method

# solve.print_solutions(sub_comb)
# solve.print_solutions(sub_ensemble)
print(sub_comb[['vary','snr']])

##################### Plotting
varies = varies
field = 'N'
bins = np.linspace(0,3,100)
# bins = 'fd'

f = plt.figure(0,figsize = [4.5,3.5])
f.clear()
ax = f.add_subplot(111)
ax.hist(sub_old.query('vary in @varies')[field],bins=bins,histtype='step',ec='grey',lw=1.5)
ax.hist(sub.query('vary in @varies')[field],bins=bins,histtype='step',ec='darkblue',lw=1.5)
ax.set_xlabel('N')

### konc vs. snr
from scipy import optimize
x = sub_comb.konc * 1/sub_comb.exp; x = x.values
y = sub_comb.snr; y = y.values
x_inter = np.linspace(0,0.4,100)
def model(x,a,b,c): return a * np.exp(-x/b) + c
p0 = [13,0.15,2]
popt, pcov = optimize.curve_fit(model,x,y,p0=p0,method='trf')
print(popt)

f = plt.figure(1,figsize = [4.5,3.5])
f.clear()
ax = f.add_subplot(111)
ax.plot(x,y,'o')
ax.plot(x_inter,
        model(x_inter,*popt),
        '-',c='r')
ax.set_xlabel('1/konc [s]')
ax.set_ylabel('snr')
