import os
import glob 
import re
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

import picasso.io as io
import lbfcs.pick_fcs as fcs
import lbfcs.pick_combine as combine


warnings.filterwarnings("ignore")

#%%
def load_all_pickedprops(dir_names,pattern = '',filetype = 'props'):
    '''
    Load all _props.hdf5 files in list of dir_Names and return as combined pandas.DataFrame.
    Also returns comprehensive list of loaded files
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
    
    ### Apply regex pattern to every path
    paths = [p for p in paths if bool( re.search( pattern,os.path.split(p)[-1] ) )]
        
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



#%%
def normalize_konc(df_in,
                   sets,
                   exp,
                   ref_q_str = 'occ < 0.4 and N < 2',
                   ):
    '''
    Normalize kon in DataFrame (df) according to datasets (sets).
    Datasets are given as nested list of ids,
    i.e.df must contain dataset id as column 'id'.
    
    kon will be normalized to first dataset (i.e. sets[0]).
    Reference will be selected in first dataset according to a reference query string.
    
    First output a DataFrame of same dimension as df (df_norm) with normalized koncs.
    Second output is a 1d numpy array of saem length as sets (flux). 
    First entry corresponds to reference kon of first dataset (set[0]), 
    following entries are given as fractions of reference kon.
    '''
    print()
    print('Normalization to:',sets[0])
    df_list =[]
    flux = []
    
    for i,s  in enumerate(sets):
        
        ### Get reference kon (median) in standardized units (10^6/Ms)
        ### Reference kon is selected in each dataset by ref_query str
        q_str = 'id in @s and '
        q_str += ref_q_str
        
        df_ref = df_in.query(q_str)
        kon_ref = df_ref.konc * (1e-6/(exp * df_ref.conc * 1e-12))
        
        ### Apply standardized filtering procedure to reference kon band
        for j in range(3):
            kon_ref = kon_ref[kon_ref > 0]
            kon_ref_med = np.nanmedian(kon_ref)
            positives = np.abs( (kon_ref-kon_ref_med) / kon_ref_med) < 0.6
            kon_ref = kon_ref[positives]
        
        ### Add median kon to flux
        flux.extend([np.nanmedian(kon_ref)])
        
        ### Normalize flux to first set
        if i>0: flux[i] = flux[i]/flux[0]
        
        ### Print some information
        if i == 0:
            print('   ',
                  s,
                  'contains %i reference groups'%(len(kon_ref)))
        else:
            print('   ',
                  s,
                  'contains %i reference groups'%(len(kon_ref)),
                  ' (%i'%(np.round(flux[i]*100)),
                  r'%)',
                  )
            
        ### Select datasets in 
        df = df_in.query('id in @s')
        
        ### Normalize kon in df according to flux
        if i>0: df.loc[:,'konc'] = df.loc[:,'konc'].values/flux[i]
        
        df_list.extend([df])
        
    df_norm = pd.concat(df_list)
    
    print('Normalization kon: %.2f (10^6/Ms)'%flux[0])
    print()
    
    return df_norm, flux


#%%
##########################################
'''
The following functions are used for summation of individual normalized picks 
to generate picks with higher N form experimental data.
'''
##########################################

def query_normalize_picked(picked_in,props,normalize=True):
    ### Query picked for groups in props
    groups = props.group.values
    picked = picked_in.query('group in @groups')
    
    ### Get normalization eps for each group and number of localizations
    eps = props.eps.values
    n_locs = picked.groupby('group').apply(lambda df: len(df)).values
    
    ### Define the normalization vector norm
    assign_eps = 200 # Assign arbitrary but homogeneous photon count after normalization
    norm = np.hstack([np.ones(n_locs[i])*e for i,e in enumerate(eps)]) / assign_eps
    
    ### Normalize
    picked_norm = picked.copy()
    if normalize:
        picked_norm.loc[:,'photons'] = picked_norm.photons.values/norm
        picked_norm.loc[:,'photons_ck'] = picked_norm.photons_ck.values/norm
    
    return picked_norm, groups


def sum_kpicks(df_in,groups,k,new_id):
    ### Number of frames in measurement
    M = int(df_in.iloc[0,:].M)
    
    ### Draw k random groups
    n = len(groups)
    ids = np.random.randint(0,n,k)
    g = groups[ids]
    
    ### Query df_in for k groups and combine to trace
    df = df_in.query('group in @g')
    t = fcs.trace_ac(df,M,field = 'photons',compute_ac=False)[0]
    t_ck = fcs.trace_ac(df,M,field = 'photons_ck',compute_ac=False)[0]
    
    ### Now get only non-zero summed photon values and frames 
    n_locs = np.sum(t>0)
    f = np.arange(0,M,1,dtype=int)[t>0]
    p = t[t>0]
    p_ck = t_ck[t>0]
    
    ### Initiate output of summed _picked
    picked = np.zeros((n_locs,len(combine.PICKED_COLS)), dtype = np.float32)
    
    ### Assign means to every column of output
    means = np.nanmean(df[combine.PICKED_COLS].values,axis=0)
    picked = np.tile(means,(n_locs,1))
    
    ### Assign group, frame and summed photon values to output
    picked[:,0] = new_id
    picked[:,1] = f
    picked[:,4] = p
    picked[:,12] = p_ck
    
    return picked


def nsum_kpicks(df,groups,n,k):
    
    picks = []
    for i in tqdm(range(n)):
        picks.extend([sum_kpicks(df,groups,k,i)])
    
    picked = np.vstack(picks)
    picked_df = pd.DataFrame(picked, columns = combine.PICKED_COLS)
    picked_df = picked_df.astype(combine.PICKED_DTYPE)
    
    return picked_df

def sum_normalized_picks(picked,props,n,k):
    
    ### 1) Query for picked for groups in props
    ### 2) Normalize photon values in picked according to eps
    picked_norm, groups = query_normalize_picked(picked,props)
    
    ### Sum k normalized picks. Do this n times
    picked_norm_sum = nsum_kpicks(picked_norm,groups,n,k)
    
    return picked_norm_sum

