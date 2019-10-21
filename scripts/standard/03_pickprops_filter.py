# Script to call lbfcs.pickprops
# Sets input parameters and defines path to data
#
############################################################# Set parameters
conc=[20] # Imager concentration [nM]
ignore=1 # Ignore_dark value for qPAINT analysis
savename_ext='_props_ig%i'%(ignore) # File extension for processed file

#### Advanced 
omit_dist=False # If True all lists will be excluded for saving (recommended) 
kin_filter=False # If True automatic filtering will be applied (recommended)
NoPartitions=30 # Number of partitions for dask parallel computing

############################################################## Define data
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p11.lbFCSnew/19-10-16_c-series_N1_T25/id154_Pm2-20nM_p38uW_control_1/19-10-16_JS'])

file_names=[]
file_names.extend(['id154_Pm2-20nM_p38uW_control_1_MMStack_Pos0.ome_locs_render_picked.hdf5'])

#################################################### Load packages
import os #platform independent paths
import importlib
import warnings
warnings.filterwarnings("ignore")

# Load user defined functions
import lbfcs.pickprops as props
import lbfcs.io
import lbfcs.pickprops_calls as props_call
# Reload modules
importlib.reload(props)
importlib.reload(props_call)

#%%
############################################################# Read locs, apply props & save locs
#### Create list of paths
path=[os.path.join(dir_names[i],file_names[i]) for i in range(0,len(file_names))]

#### Read-Apply-Save loop
for i in range(0,len(path)):
    #### File read in
    print('File read in ...')
    locs,locs_info=lbfcs.io.load_locs(path[i])
    
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
    
    
    #### Kinetic filtering
    groups_nofilter=len(locs_props) # Number of groups before filter
    if kin_filter:
        print('Applying kinetic filter ...')
        locs_props=props._kin_filter(locs_props)
    groups_filter=len(locs_props) # Number of groups after filter
    
    #### Add nearest neigbour pick and distance
    print('Calculating nearest neighbour ...')
    locs_props=props_call.props_add_nn(locs_props)
    
    #### Save .hdf5 and .yaml of locs_props
    props_info={'Generated by':'pickprops.get_props',
            'ignore':ignore,
            'kin_filter': kin_filter,
            '# of picks before filter':groups_nofilter,
            '# of picks after filter':groups_filter,
            }
    if omit_dist:
        print('File saving ...')
        lbfcs.io.save_locs(path[i].replace('.hdf5',savename_ext+'.hdf5'),
                           locs_props,
                           [locs_info,props_info],
                           mode='picasso_compatible')

#%%
import matplotlib.pyplot as plt
import lbfcs.varfuncs as varfuncs
import importlib
importlib.reload(varfuncs)
import numpy as np
plt.style.use('~/lbFCS/styles/paper.mplstyle')

############################################ Individual picks
f=plt.figure(num=12,figsize=[4,3])
f.subplots_adjust(bottom=0.1,top=0.99,left=0.2,right=0.99)
f.clear()

#### Autocorrelation
ax=f.add_subplot(311)
for g in [319]:
    print(locs_props.loc[g,'mono_tau'],locs_props.loc[g,'mono_tau_lin'])
    ax.plot(locs_props.loc[g,'tau'],
            varfuncs.ac_monoexp(locs_props.loc[g,'tau'],locs_props.loc[g,'mono_A'],locs_props.loc[g,'mono_tau']),
            '-',lw=2,c='r')
    ax.plot(locs_props.loc[g,'tau'],
            varfuncs.ac_monoexp(locs_props.loc[g,'tau'],locs_props.loc[g,'mono_A_lin'],locs_props.loc[g,'mono_tau_lin']),
            '-',lw=2,c='b')
    ax.plot(locs_props.loc[g,'tau'],locs_props.loc[g,'g'],'*')
    ax.axhline(1,ls='--',lw=2,color='k')
ax.set_xscale('symlog')
#### Trace
ax=f.add_subplot(312)
ax.plot(locs_props.loc[g,'trace'])
ax.set_ylim(0,2000)

#### tau_d_dist
ax=f.add_subplot(313)
x=varfuncs.get_ecdf(locs_props.loc[g,'tau_d_dist'])[0]
y=varfuncs.get_ecdf(locs_props.loc[g,'tau_d_dist'])[1]
x_fit=np.linspace(0,x[-1],100)
ax.plot(x,y)
ax.plot(x_fit,varfuncs.ecdf_exp(x_fit,locs_props.loc[g,'tau_d'],locs_props.loc[g,'tau_d_off'],locs_props.loc[g,'tau_d_a']))

ax.set_xlim(0,x.max()+1)
ax.set_ylim(0,y.max()+0.1)


########################################## Distributions
field='n_locs'
bins=np.arange(0,9000,100)

f=plt.figure(num=11,figsize=[4,3])
f.subplots_adjust(bottom=0.1,top=0.99,left=0.2,right=0.99)
f.clear()
ax=f.add_subplot(111)
#ax.hist(locs_props.loc[:,field].dropna(),bins=bins,color='gray',edgecolor='k',histtype='step',density=True);
ax.scatter(locs_props.n_locs,locs_props.mean_frame,s=50,alpha=0.01)

X=locs_props.copy()

istrue=np.abs(X.mono_tau_lin-X.mono_tau)/X.mono_tau<0.2
X=X.loc[istrue,:]
istrue=np.abs(X.mean_frame-NoFrames*0.5)/(NoFrames*0.5)<0.15
X=X.loc[istrue,:]
istrue=np.abs(X.std_frame-X.std_frame.mean())/(X.std_frame.mean())<0.15
X=X.loc[istrue,:]

#istrue=X.n_locs>2300
#X=X.loc[istrue,:]

ax.scatter(X.n_locs,X.mean_frame,s=50,alpha=0.01)
print(np.sum(istrue))

field='std_frame'
bins=np.arange(0,9000,100)

f=plt.figure(num=13,figsize=[4,3])
f.subplots_adjust(bottom=0.1,top=0.99,left=0.2,right=0.99)
f.clear()
ax=f.add_subplot(111)
ax.hist(X.loc[:,field].dropna(),bins=bins,color='gray',edgecolor='r',histtype='step',density=True);
        
        
        
        
        