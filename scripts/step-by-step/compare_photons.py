import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lbfcs
import lbfcs.simulate as simulate
import lbfcs.solveseries as solve
import picasso.io as io

plt.style.use('~/lbFCS/styles/paper.mplstyle')

#%%
##################### Parameters
savepath = '~/bla.hdf5'
reps = 1000
M = 1000
CycleTime = 0.4

N = 1
koff = 1.15e-1
kon = 17e6
c = 5e-9

box = 7
e_tot = 400
snr = lbfcs.snr_from_conc(c)
sigma = 0.9
use_weight = False

### Simulate hybridization reaction
locs, info, spots = simulate.generate_locs(savepath,reps,M,CycleTime,N,koff,kon,c,box, e_tot, snr, sigma, use_weight)

#%%
##################### Plot
bins = np.linspace(0,1400,100)
 
f=plt.figure(1,figsize=[4.5,3.5])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
f.clear()
ax = f.add_subplot(111)
ax.hist(locs['photons'],
        bins=bins,
        histtype='step',
        ec='k')
ax.set_yscale('log')


#%%
##################### Compare with real measurement
dir_name = '/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-10_N1-5xCTC_cseries_varexp/21-01-19_FS_id180_exp400'
file_name_locs = 'id180_Pm2-05nM_p40uW_exp400_1_MMStack_Pos0.ome_locs_render_picked.hdf5'
file_name_props = 'id180_Pm2-05nM_p40uW_exp400_1_MMStack_Pos0.ome_locs_render_picked_props.hdf5'


locs_exp_init = pd.DataFrame(io.load_locs(os.path.join(dir_name,file_name_locs))[0]) 
props_exp_init = pd.DataFrame(io.load_locs(os.path.join(dir_name,file_name_props))[0]) 

### Filter props and query locs for groups in picked
props_exp = solve.prefilter(props_exp_init)
groups = props_exp.group.values
locs_exp = locs_exp_init.query('group in @groups')

##################### Plot
bins = np.linspace(0,1400,100)

select = (np.abs(locs_exp.x - 350) < 50) & (np.abs(locs_exp.y - 350) < 50)

f=plt.figure(2,figsize=[4.5,3.5])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
f.clear()
ax = f.add_subplot(111)
ax.hist(locs_exp[select]['photons'],
        bins=bins,
        histtype='step',
        ec='k')
ax.set_yscale('log')
