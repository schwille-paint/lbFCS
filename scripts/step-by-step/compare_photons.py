import numpy as np
import matplotlib.pyplot as plt
import importlib

import lbfcs.simulate as simulate
import picasso_addon.localize as localize
importlib.reload(simulate)

plt.style.use('~/lbFCS/styles/paper.mplstyle')

#%%
##################### Parameters
savepath = '~/bla.hdf5'
reps = 1000
M = 1000
CycleTime = 0.2

N = 1
koff = 0.15
kon = 15e6
c = 10e-9

box = 7
e_tot = 400
snr = 2
sigma = 0.9
use_weight = True

### Simulate hybridization reaction
locs, info, spots = simulate.generate_locs(savepath,reps,M,CycleTime,N,koff,kon,c,box, e_tot, snr, sigma, use_weight)

#%%
##################### Plot
bins = np.linspace(0,5,100)

# positives = locs_long['imagers']%1 > 0

# level = 0
# positives = (locs_long['imagers'] >level) & (locs_long['imagers'] <level+1)

positives = locs['net_gradient'] > 600
# f=plt.figure(1,figsize=[4,3])
# f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
# f.clear()
# ax = f.add_subplot(111)
# signal = ax.hist(locs['photons']/e_tot,
#         bins=bins,
#         histtype='step',
#         ec='g')[0]
# bg = ax.hist(locs[positives]['photons']/e_tot,
#         bins=bins,
#         histtype='step',
#         ec='r')[0]
# # ax.plot(bins[:-1] + (bins[1]-bins[0])/2,
# #         signal-bg,
# #         c='b')
# ax.set_yscale('log')
# ax.set_ylim(10,1e5)


f=plt.figure(2,figsize=[4,3])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
f.clear()
ax = f.add_subplot(111)
ax.scatter(locs[positives]['photons'],
            locs[positives]['net_gradient'],
            s=10,
            alpha=0.5,
            )
ax.set_xlim(0,5*e_tot)
for i in range(4): ax.axvline(i*e_tot,lw=2,c='r')