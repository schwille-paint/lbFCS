import numpy as np
import matplotlib.pyplot as plt
import importlib

import lbfcs.simulate as simulate
import picasso_addon.localize as localize
importlib.reload(simulate)

plt.style.use('~/lbFCS/styles/paper.mplstyle')

##################### Parameters
savepath = '~/bla.hdf5'
reps = 100
M = 1000
CycleTime = 0.2

N = 10
koff = 0.28
kon = 6e6
c = 10e-9

e_tot = 300
snr = 3
sigma = 0.9
box = 9

### Simulate hybridization reaction
locs = simulate.generate_locs(savepath,reps,M,CycleTime,N,koff,kon,c,box)
### Generate spots
spots_noise, spots_readvar, spots, spots_shotnoise, spots_readnoise = simulate.generate_spots(locs, box, e_tot, snr, sigma)
### Fit spots
locs_fit = simulate.fit_spots(locs,box,spots_noise,spots_readvar,use_weight=True)

#%%
##################### Test net_gradient
ng = simulate.netgradient_spots(spots_noise)


f=plt.figure(1,figsize=[4,3])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
f.clear()
ax = f.add_subplot(111)
ax.hist(ng,
        bins=np.linspace(0,3000,100),
        histtype='step',
        density=True,
        ec='grey')