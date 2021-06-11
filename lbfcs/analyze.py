import os
import glob 
import re
import numpy as np
import pandas as pd
import warnings

import picasso.io as io

warnings.filterwarnings("ignore")

#%%
def load_all_pickedprops(dir_names,must_contain = '',filetype = 'props'):
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

    # paths = sorted(paths)
    
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
            kon_ref_med = np.median(kon_ref)
            positives = np.abs( (kon_ref-kon_ref_med) / kon_ref_med) < 0.6
            kon_ref = kon_ref[positives]
        
        ### Add median kon to flux
        flux.extend([np.median(kon_ref)])
        
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




