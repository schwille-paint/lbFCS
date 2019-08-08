#################################################### Load packages
import os #platform independent paths
import pandas as pd 
import importlib
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
savedir='/fs/pool/pool-schwille-paint/Analysis/p04.lb-FCS/zz.Pm2-8nt/z.c-series/z.datalog'
savename='N01_Gel_B_2B07_stock11'

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
X_stats.to_hdf(os.path.join(savedir,savename+'_stats.h5'),key='stats')
X_fit.to_hdf(os.path.join(savedir,savename+'_fit.h5'),key='fit')
############################################################## Plotting

#%%
#### Show histogram for expID
expID='20.0nM-1'
#f=plt.figure(num=10,figsize=[7,8])
#f.subplots_adjust(left=0.1,right=0.99,bottom=0.04,top=0.95)
#f.clear()
#ax=f.add_subplot(111)
#X.loc[expID,['std_frame','mean_frame',
#                      'mono_tau','mono_A',
#                      'N','tau_d']].hist(bins='fd',ax=ax)

#### Kinetics plot
kinetics._plot(X_stats,X_fit)

print('koff = %.2e'%(X_fit.loc['tau_conc','popt0']))
print('kon  = %.2e'%(X_fit.loc['tau_conc','popt1']))
print('N  = %.2f'%(X.N.median()))
