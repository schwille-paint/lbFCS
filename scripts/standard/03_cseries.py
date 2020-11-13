#################################################### Load packages
import os #platform independent paths
import pandas as pd 
import numpy as np
import importlib
import sys
import matplotlib.pyplot as plt

# Load user defined functions
import picasso_addon.io as io
import lbfcs.cseries as cseries
importlib.reload(cseries)

############################################################## Parameters & labels
#### Aquistion cycle time (s)
CycleTime=0.2 

#### Define outliers
outliers=[]

#### Saving
save_results=False
savedir=''
savename=os.path.splitext(os.path.basename(sys.argv[0]))[0]

############################################################## Define data
dir_names=[]
file_names=[]

############## New Metabion imager @T=23, concentration adjusted!
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-10-30_SDS_T23_Pm2-Meta/id181_Pm2-5nM_p38uW_Plasma_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-10-30_SDS_T23_Pm2-Meta/id181_Pm2-10nM_p38uW_Plasma_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-10-30_SDS_T23_Pm2-Meta/id181_Pm2-20nM_p38uW_Plasma_1'])

file_names.extend(['id181_Pm2-5nM_p38uW_Plasma_1_MMStack_Pos0.ome_locs_render_picked_props.hdf5'])
file_names.extend(['id181_Pm2-10nM_p38uW_Plasma_1_MMStack_Pos0.ome_locs_render_picked_props.hdf5'])
file_names.extend(['id181_Pm2-20nM_p38uW_Plasma_1_MMStack_Pos0.ome_locs_render_picked_props.hdf5'])


############################################################## Read in data
#### Create list of paths
path=[os.path.join(dir_names[i],file_names[i]) for i in range(0,len(file_names))]
#### Read in locs of path
locs_props=pd.concat([io.load_locs(p)[0] for p in path],keys=range(len(file_names)),names=['expID'])
X=locs_props.copy()


############################################################## Filter
X=X.reset_index(level=['expID'])
X=X.groupby('expID').apply(cseries._filter)
X=X.drop(columns=['expID'])

############################################################## Statistics of observables
X_stats=cseries._stats(X,CycleTime)

############################################################## Concentration series fitting
X,X_stats,X_fit=cseries._fit(X.drop(outliers),X_stats.drop(outliers))

#%%
############################################################## Saving
if save_results:
    X_stats.to_hdf(os.path.join(savedir,savename+'_stats.h5'),key='stats')
    X_fit.to_hdf(os.path.join(savedir,savename+'_fit.h5'),key='fit')

############################################################## Plotting
#### lbFCS standard plots
cseries._plot(X_stats,X_fit)

#### Print results
print('lbFCS')
print('    koff = %.2e'%(X_fit.loc['lbfcs','popt0']))
print('    kon  = %.2e'%(X_fit.loc['lbfcs','popt1']))
print('    N  = %.2f'%(X.N.median()))
print('qPAINT')
print('    kon  = %.2e'%(X_fit.loc['qpaint','popt0']))

#### Plot certain distribtions
field='n_locs'
subset=0 # Certain measurement
# subset=X.vary>1000 # Boolean subset, e.g. imager concentration > 10nM

bins='fd'
bins=np.linspace(0,1,100)

f=plt.figure(12,figsize=[4,3])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
f.clear()

ax=f.add_subplot(111)
ax.hist(X.loc[subset,field].dropna(),bins=bins,edgecolor='k',color='gray');
#ax.set_xlim(0,5);
ax.set_xlabel(field);
ax.set_ylabel('Counts');

#%%
# importlib.reload(cseries)

# #### Plot certain distribtions
# field='n_locs'
# bins=np.linspace(0,1,100)

# f=plt.figure(13,figsize=[4,3])
# f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
# f.clear()
# ax=f.add_subplot(111)


# subset=0 
# ax.hist(X.loc[subset,field].dropna(),bins=bins,edgecolor='r',histtype='step',label='new',density=True);

# subset=2 
# ax.hist(X.loc[subset,field].dropna(),bins=bins,edgecolor='k',histtype='step',label='old',density=True);

# ax.legend()
# ax.set_xlabel(field);
# ax.set_ylabel('Counts');