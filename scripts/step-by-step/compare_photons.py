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
M = 2000
CycleTime = 0.1

N = 10
koff = 0.56
kon = 6e6
c = 10e-9

e_tot = 150
snr = 2.5
sigma = 0.9
box = 9

### Simulate hybridization reaction
locs = simulate.generate_locs(savepath,reps,M,CycleTime,N,koff,kon,c,box,factor=10)
### Generate spots
spots_noise, spots_readvar, spots, spots_shotnoise, spots_readnoise = simulate.generate_spots(locs, box, e_tot, snr, sigma)
### Fit spots
locs_fit = simulate.fit_spots(locs,box,spots_noise,spots_readvar,use_weight=False)
spots_noise.shape = (len(locs),box,box)
#%%
##################### Plot
f=plt.figure(1,figsize=[4,3])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
f.clear()
ax = f.add_subplot(111)
ax.scatter(locs_fit['imagers'],
            locs_fit['photons'],
            s=10,
            alpha=0.5)
ax.set_xlim(0,N)
ax.set_ylim(0,N*e_tot)

f=plt.figure(3,figsize=[4,3])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
f.clear()
ax = f.add_subplot(111)
ax.hist(locs_fit['photons'],
        bins=np.linspace(0,N*e_tot,100),
        histtype='step',
        density=True,
        ec='grey')
ax.hist(locs_fit['imagers']*e_tot,
        bins=np.linspace(0,N*e_tot,100),
        histtype='step',
        density=True,
        ec='k')
ax.set_yscale('log')
#%%
i = 1
# outlier_crit = (locs_fit['photons']>3*e_tot) & (locs_fit['imagers']==1)
outlier_crit = (locs_fit['imagers']==1)
outliers = np.where(outlier_crit)[0]
i = outliers[i]

print('Photons:      %i'%locs_fit['photons'][i])
print('Background:%i'%locs_fit['bg'][i])
print('x_in, x:         %.2f, %.2f'%(locs['x'][i],locs_fit['x'][i]))
print('y_in, y:         %.2f, %.2f'%(locs['y'][i],locs_fit['y'][i]))
print('sx, sy:          %.2f, %.2f'%(locs_fit['sx'][i],locs_fit['sy'][i]))

f=plt.figure(2,figsize=[6,6])
f.subplots_adjust(bottom=0.1,top=0.95,left=0.1,right=0.95)
f.clear()

ax = f.add_subplot(221) 
mapp = ax.imshow(spots[i,:,:],
                 vmin=np.min(spots_noise[i,:,:].flatten()),
                 vmax=np.max(spots_noise[i,:,:].flatten()),
                 cmap=plt.magma(),
                 interpolation='none',
                 origin='lower')
plt.colorbar(mapp,ax=ax,shrink=0.8)
ax.set_title(r'True spot [$e^-$]')

ax = f.add_subplot(222) 
mapp = ax.imshow(np.abs(spots_readnoise[i,:,:]/spots_shotnoise[i,:,:])*100,
                 vmin=0,
                 vmax=200,
                 cmap=plt.magma(),
                 interpolation='none',
                 origin='lower')
plt.colorbar(mapp,ax=ax,shrink=0.8)
ax.set_title(r'Read/Shot noise [%]')

ax = f.add_subplot(223) 
mapp = ax.imshow(spots_noise[i,:,:],
                 vmin=np.min(spots_noise[i,:,:].flatten()),
                 vmax=np.max(spots_noise[i,:,:].flatten()),
                 cmap=plt.magma(),
                 interpolation='none',
                 origin='lower')
plt.colorbar(mapp,ax=ax,shrink=0.8)
ax.set_title(r'Noisy spot [$e^-$]')

ax = f.add_subplot(224) 
mapp = ax.imshow((spots_noise[i,:,:] - spots[i,:,:])*(100/spots[i,:,:]),
                 vmin=-20,
                 vmax=20,
                 cmap=plt.magma(),
                 interpolation='none',
                 origin='lower')
plt.colorbar(mapp,ax=ax,shrink=0.8)
ax.set_title(r'Total noise [%]')