#################################################### Load packages
import os #platform independent paths
import pandas as pd 
import importlib
import sys
# Load user defined functions
import lbfcs.io as io
import lbfcs.cseries as cseries
importlib.reload(cseries)

############################################################## Parameters & labels
#### Aquistion cycle time (s)
CycleTime=0.2 
#### Concentrations (nM) and repetitons
cs=[5,10,20]
rs=[1,1,1]
labels=['%4.1fnM-%i'%(c,r) for c,r in zip(cs,rs)]

#### Define outliers
outliers=[]

#### Saving
save_results=False
savedir='/fs/pool/pool-schwille-paint/Analysis/p04.lb-FCS/zz.Pm2-8nt/z.c-series/z.datalog'
savename=os.path.splitext(os.path.basename(sys.argv[0]))[0]

############################################################## Define data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-18_N=48/id114_5nM_p35uW_control_1/19-06-18_FS'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-18_N=48/id114_10nM_p35uW_control_1/19-06-18_FS'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-18_N=48/id114_20nM_p35uW_control_1/19-06-18_FS'])

file_names=[]
file_names.extend(['id114_5nM_p35uW_control_1_MMStack_Pos0.ome_locs_render_picked_props_ig1.hdf5']) # 11th stock (after N=48 on 19.06.18) 
file_names.extend(['id114_10nM_p35uW_control_1_MMStack_Pos0.ome_locs_render_picked_props_ig1.hdf5']) # 11th stock (after N=48 on 19.06.18) 
file_names.extend(['id114_20nM_p35uW_control_1_MMStack_Pos0.ome_locs_render_picked_props_ig1.hdf5']) # 11th stock (before N=48 on 19.06.18) 

############################################################## Read in data
#### Create list of paths
path=[os.path.join(dir_names[i],file_names[i]) for i in range(0,len(file_names))]
#### Read in locs of path
locs_props=pd.concat([io.load_locs(p)[0] for p in path],keys=labels,names=['expID'])
X=locs_props.copy()

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

#### Plot certain distributions
import matplotlib.pyplot as plt
import numpy as np

field='mono_A'
subset='20.0nM-1' # Certain measurement
subset=X.conc>=1 # Boolean subset, e.g. imager concentration > 10nM

bins='fd'
#bins=np.arange(0,5,0.1)

f=plt.figure(12,figsize=[4,3])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
f.clear()

ax=f.add_subplot(111)
ax.hist(X.loc[subset,field].dropna(),bins=bins,edgecolor='k',color='gray');
#ax.set_xlim(0,5);
ax.set_xlabel(field);
ax.set_ylabel('Counts');
