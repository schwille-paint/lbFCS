# Script to call lbfcs.pickprops
# Sets input parameters and defines path to data
#
############################################################# Set parameters
conc=[2.5,5,10,20] # Imager concentration [nM]
ignore=1 # Ignore_dark value for qPAINT analysis
savename_ext='_props_ig%i'%(ignore) # File extension for processed file

#### Advanced 
omit_dist=True # If True all lists will be excluded for saving (recommended) 
kin_filter=True # If True automatic filtering will be applied (recommended)
NoPartitions=30 # Number of partitions for dask parallel computing


############################################################## Define data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-07_N=48/id135_2-5nM_p35uW_1/19-06-07_JS/'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-07_N=48/id135_5nM_p35uW_1/19-06-07_JS/'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-07_N=48/id135_10nM_p35uW_1/19-06-07_JS/'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-06-07_N=48/id135_20nM_p35uW_1/19-06-07_JS/'])

file_names=[]
file_names.extend(['id135_2-5nM_p35uW_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
file_names.extend(['id135_5nM_p35uW_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
file_names.extend(['id135_10nM_p35uW_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])
file_names.extend(['id135_20nM_p35uW_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])


#################################################### Load packages
import os #platform independent paths
import importlib
import warnings
warnings.filterwarnings("ignore")

# Load user defined functions
import lbfcs.pickprops as props
import lbfcs.io as io
import lbfcs.pickprops_calls as props_call
# Reload modules
importlib.reload(props)
importlib.reload(props_call)

#%%
############################################################# Read locs, apply props & save locs
#### Create list of paths
path=[os.path.join(dir_names[i],file_names[i]) for i in range(0,len(file_names))]

#### Dictonary added content for info '.yaml' file
props_info={'Generated by':'pickprops.get_props',
            'ignore':ignore,
            'omit_dist':omit_dist,
            'kin_filter': kin_filter}

#### Read-Apply-Save loop
for i in range(0,len(path)):
    #### File read in
    print('File read in ...')
    locs,locs_info=io.load_locs(path[i])
    
    #### Get number of frames
    NoFrames=locs_info[0]['Frames']
    
    #### Apply props
    print('Calculating kinetics ...')
    if NoPartitions==1:
        print('... non-parallel')
        locs_props=props.apply_props(locs,conc[i],NoFrames,ignore)
    elif NoPartitions>1:
        print('... in parallel')
        locs_props=props.apply_props_dask(locs,conc[i],NoFrames,ignore,NoPartitions)
    
    #### Drop objects for saving if omit=True
    if omit_dist:
        print('Removing distribution-lists from output ...')
        locs_props=locs_props.drop(['trace','tau','g','tau_b_dist','tau_d_dist'],axis=1)
    
    if kin_filter:
        print('Applying kinetic filter ...')
        locs_props=props._kin_filter(locs_props)
    
    #### Add nearest neigbour pick and distance
    print('Calculating nearest neighbour ...')
    locs_props=props_call.props_add_nn(locs_props)
    
    #### Save .hdf5 and .yaml of locs_props
    if omit_dist:
        print('File saving ...')
        io.save_locs(path[i].replace('.hdf5',savename_ext+'.hdf5'),
                        locs_props,
                        [locs_info,props_info],
                        mode='picasso_compatible')

#%%
#import matplotlib.pyplot as plt
#import lbfcs.varfuncs as varfuncs
#import numpy as np
#plt.style.use('~/qPAINT/styles/paper.mplstyle')
#
#f=plt.figure(num=12,figsize=[4,3])
#f.subplots_adjust(bottom=0.1,top=0.99,left=0.2,right=0.99)
#f.clear()
#
##### Autocorrelation
#ax=f.add_subplot(311)
#for g in [49]:
#    print(locs_props.loc[g,'mono_tau'],locs_props.loc[g,'mono_tau_lin'])
#    ax.plot(locs_props.loc[g,'tau'],
#            varfuncs.ac_monoexp(locs_props.loc[g,'tau'],locs_props.loc[g,'mono_A'],locs_props.loc[g,'mono_tau']),
#            '-',lw=2,c='r')
#    ax.plot(locs_props.loc[g,'tau'],
#            varfuncs.ac_monoexp(locs_props.loc[g,'tau'],locs_props.loc[g,'mono_A_lin'],locs_props.loc[g,'mono_tau_lin']),
#            '-',lw=2,c='b')
#    ax.plot(locs_props.loc[g,'tau'],locs_props.loc[g,'g'],'*')
#    ax.axhline(1,ls='--',lw=2,color='k')
#ax.set_xscale('symlog')
##### Trace
#ax=f.add_subplot(312)
#ax.plot(locs_props.loc[g,'trace'])
#ax.set_ylim(0,1000)
#
##### tau_d_dist
#ax=f.add_subplot(313)
#x=varfuncs.get_ecdf(locs_props.loc[g,'tau_d_dist'])[0]
#y=varfuncs.get_ecdf(locs_props.loc[g,'tau_d_dist'])[1]
#x_fit=np.arange(0,np.max(x),0.1)
#ax.plot(x,y)
#ax.plot(x_fit,varfuncs.ecdf_exp(x_fit,locs_props.loc[g,'tau_d']))