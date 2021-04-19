import os 
import re
import sys
import glob
import os
from datetime import datetime
import getpass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import picasso.io as io


#%%
def load_all_pickedprops(dir_names,must_contain,filetype = 'props'):
    '''
    Load all _props.hdf5 files in list of dir_Names and return as combined pandas.DataFrame. Also returns comprehensive list of loaded files
    '''
    
    if filetype == 'props':
        filetype_pattern = '*_props*.hdf5'
    elif filetype == 'picked':
        filetype_pattern = '*_picked.hdf5'
    else:
        filetype_pattern = '*_props*.hdf5'
    
    ### Get sorted list of all paths to file-type in dir_names
    paths = []
    for dir_name in dir_names:
        path = sorted( glob.glob( os.path.join( dir_name,filetype_pattern) ) )
        paths.extend(path)

    paths = sorted(paths)
    
    ### Necessary string pattern to be found in files
    for pattern in must_contain:
        paths = [p for p in path if bool(re.search(pattern,p))]
        
    ### Load props
    ids = range(len(paths))
    try:
        props = pd.concat([pd.DataFrame(io.load_locs(p)[0]) for p in paths],keys=ids,names=['id'])
        props = props.reset_index(level=['id'])
    except ValueError:
            print('No files in directories!')
            print(dir_names)
            return 0,0
    
    ### Load infos and get aquisition dates
    infos = [io.load_info(p) for p in paths]
    try: # Data aquisition with Micro-Mananger
        dates = [info[0]['Micro-Manager Metadata']['Time'] for info in infos] # Get aquisition date
        dates = [int(date.split(' ')[0].replace('-','')) for date in dates]
    except: # Simulations
        dates = [info[0]['date'] for info in infos]
    
    ### Create comprehensive list of loaded files
    files = pd.DataFrame([],
                          index=range(len(paths)),
                          columns=['id','date','conc','file_name'])
    ### Assign id, date
    files.loc[:,'id'] = ids
    files.loc[:,'date'] = dates
    ### Assign file_names
    file_names = [os.path.split(path)[-1] for path in paths]
    files.loc[:,'file_name'] = file_names
    ### Assign concentration
    cs = props.groupby(['id']).apply(lambda df: df.conc.iloc[0])
    files.loc[:,'conc'] = cs.values
    
    return props, files






















# OBS_COLS = ['tau','A','occ','I','var_I','B','taud','events','eps_direct']
# OBS_COLS_LABELS = [r'$\tau$','A','occ','I',r'$I_{var}$','B',r'$\tau_d$','events',r'$\epsilon_{direct}$']

# #%%
# def residual_violinplot_toax(ax,obs_res,show_violin=False,show_cols=[0,1,2,3,8],csi=0):
#     ### Color_scheme
#     mfcs = ['w','lightgrey','r']
#     cs = ['k','grey','k']
    
#     ### Remove non-finite entries
#     arefinite = np.all(np.isfinite(obs_res[['tau','A','occ','I','eps_direct']]),axis=1)
#     obs_res = obs_res[arefinite]
    
#     ### Select columns to use
#     cols = [OBS_COLS[i] for i in show_cols]
#     cols_labels = [OBS_COLS_LABELS[i] for i in show_cols]
    
#     if show_violin:
#         ### Violin plot
#         parts = ax.violinplot(obs_res[cols],
#                       showmeans = False,
#                       showextrema = False,
#                       points = 50,
#                       widths = 0.7,
#                       )
#         for pc in parts['bodies']:
#             pc.set_facecolor('lightgrey')
#             pc.set_edgecolor('black')
#             pc.set_linewidth(1)
#             pc.set_alpha(1)
    
#     ### Centers and quantile bars as errorbar on top of violin
#     means = np.mean(obs_res[cols], axis=0)
#     quartile1, medians, quartile3 = np.percentile(obs_res[cols], [10, 50, 90], axis=0)
#     centers = medians.copy()                      # Center corresponds to median ...
#     centers[0] = means[0]                           # ... except for tau
#     quartile1 = np.abs(quartile1 - centers)
#     quartile3 = np.abs(quartile3 - centers)
    
#     line = ax.errorbar(range(1,len(cols)+1),
#                        centers,
#                        yerr=(quartile1,quartile3),
#                        fmt='o',mfc=mfcs[csi],mec='k',c=cs[csi],lw=1.5,
#                        )
    
#     ### Zero line as guide for the eye
#     ax.axhline(0,ls='--',c='k')
    
#     ax.set_xticks(range(1,len(cols)+1))
#     ax.set_xticklabels(cols_labels,rotation=70)
#     ax.set_ylabel(r'$\Delta_{rel}$ [%]')
#     ax.set_ylim([-50,50])
    
#     return line

