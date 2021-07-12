import numpy as np
import matplotlib.pyplot as plt
import importlib

import lbfcs.simulate as simulate
import picasso_addon.localize as localize
importlib.reload(simulate)

# plt.style.use('~/lbFCS/styles/paper.mplstyle')

##################### Parameters
savepath = '~/bla.hdf5'
reps = 10
M = 1000
CycleTime = 0.4

N = 1
koff = 0.11
kon = 17e6
c = 5e-9

box = 9
e_tot = 2000
snr = 10
sigma = 1.5
use_weight = False

### Simulate hybridization reaction
locs, info, spots = simulate.generate_locs(savepath,reps,M,CycleTime,N,koff,kon,c,box, e_tot, snr, sigma, use_weight)
spots_noise = spots[0]
spots_true = spots[2]
spots_shotnoise = spots[3]
spots_readnoise = spots[4]

#%%
##################### View spots
idx = 1 # Select
locs_select = locs.iloc[idx,:] 

##################### Print fitted parameters
print('Photons:        %i (%i)'%(locs_select['photons'],locs_select['imagers']*e_tot))
print('Background:  %i (%i)'%(locs_select['bg'], e_tot/(2*np.pi*sigma**2*snr)))
print('x:                   %.2f (%.2f)'%(locs_select['x'],locs_select['x_in']))
print('y:                   %.2f (%.2f)'%(locs_select['y'],locs_select['y_in']))
print('sx, sy:            %.2f, %.2f (%.2f)'%(locs_select['sx'],locs_select['sy'],sigma))

##################### Prepare fitted spot
i,j = np.meshgrid(np.arange(box), np.arange(box), indexing='ij')
p = np.zeros(6)
p[0] = locs_select['photons']
p[0] /= 2 * np.pi * locs_select['sx'] * locs_select['sy']
p[1] = int(box/2) + locs_select['y'] - np.round(locs_select['y_in'])
p[2] = int(box/2) + locs_select['x'] - np.round(locs_select['x_in'])
p[3], p[4], p[5] = locs_select['sy'], locs_select['sx'], locs_select['bg']
p.shape = (1,6)
spot_fit = simulate.gauss2D_on_mesh(p, i, j)

f=plt.figure(11,figsize=[7,4])
f.subplots_adjust(bottom=0.1,top=0.95,left=0.1,right=0.95)
f.clear()
##################### True spot
ax = f.add_subplot(231) 
mapp = ax.imshow(spots_true[idx,:,:],
                 vmin=np.min(spots_noise[idx,:,:].flatten()),
                 vmax=np.max(spots_noise[idx,:,:].flatten()),
                 cmap=plt.gray(),
                 interpolation='none',
                 origin='lower')
plt.colorbar(mapp,ax=ax,shrink=0.8)
ax.set_title(r'True spot [$e^-$]')
ax.set_xticks([])
ax.set_yticks([])
##################### Noisy spot
ax = f.add_subplot(232) 
mapp = ax.imshow(spots_noise[idx,:,:],
                 vmin=np.min(spots_noise[idx,:,:].flatten()),
                 vmax=np.max(spots_noise[idx,:,:].flatten()),
                 cmap=plt.gray(),
                 interpolation='none',
                 origin='lower')
plt.colorbar(mapp,ax=ax,shrink=0.8)
ax.set_title(r'Noisy spot [$e^-$]')
ax.set_xticks([])
ax.set_yticks([])
##################### Fitted spot
ax = f.add_subplot(233) 
mapp = ax.imshow(spot_fit[0,:,:],
                 vmin=np.min(spots_noise[idx,:,:].flatten()),
                 vmax=np.max(spots_noise[idx,:,:].flatten()),
                 cmap=plt.gray(),
                 interpolation='none',
                 origin='lower')
plt.colorbar(mapp,ax=ax,shrink=0.8)
ax.set_title(r'Fitted spot [$e^-$]')
ax.set_xticks([])
ax.set_yticks([])
##################### Shotnoise relative
ax = f.add_subplot(234) 
mapp = ax.imshow(np.abs(spots_shotnoise[idx,:,:]*(100/spots_true[idx,:,:])),
                 vmin= 0,
                 vmax=30,
                 cmap=plt.gray(),
                 interpolation='none',
                 origin='lower')
plt.colorbar(mapp,ax=ax,shrink=0.8)
ax.set_title(r'Shotnoise [% ]')
ax.set_xticks([])
ax.set_yticks([])
##################### Readnoise relative
ax = f.add_subplot(235) 
mapp = ax.imshow(np.abs(spots_readnoise[idx,:,:]*(100/spots_true[idx,:,:])),
                 vmin= 0,
                 vmax=30,
                 cmap=plt.gray(),
                 interpolation='none',
                 origin='lower')
plt.colorbar(mapp,ax=ax,shrink=0.8)
ax.set_title(r'Readnoise [% ]')
ax.set_xticks([])
ax.set_yticks([])
##################### Total noise relative
ax = f.add_subplot(236) 
mapp = ax.imshow(np.abs((spots_noise[idx,:,:] - spots_true[idx,:,:])*(100/spots_true[idx,:,:])),
                 vmin=0,
                 vmax=30,
                 cmap=plt.gray(),
                 interpolation='none',
                 origin='lower')
plt.colorbar(mapp,ax=ax,shrink=0.8)
ax.set_title(r'Total noise [% ]')
ax.set_xticks([])
ax.set_yticks([])