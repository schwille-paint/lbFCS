# Import modules
import os
import time
from datetime import datetime
import getpass
import warnings

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.spatial
from tqdm import tqdm
import dask.dataframe as dd
import multiprocessing as mp

warnings.filterwarnings("ignore")

# Import own modules
import picasso_addon.io as addon_io

import lbfcs.pick_fcs as fcs
import lbfcs.pick_other as other
import lbfcs.pick_qpaint as qpaint
import lbfcs.pick_solve as solve

PROPS_DTYPE = {'conc':np.uint32,
               'group':np.uint16,
               #
               'tau':np.float32,
               'A':np.float32,
               'occ':np.float32,
               'I':np.float32,
               'eps':np.float32,
               #
               'koff':np.float32,
               'konc':np.float32,
               'N':np.float32,
               'success':np.float32,
               'success_epsnorm':np.float32,
               'p':np.float32,
               'N_p':np.float32,
               'N_start':np.float32,
               #
               'tau_b':np.float32,
               'tau_d':np.float32,
               'events':np.float32,
               'ignore':np.uint8,
               #
               'frame':np.float32,
               'std_frame':np.float32,
               'photons':np.float32,
               'bg':np.float32,
               'x':np.float32,
               'y':np.float32,
               'lpx':np.float32,
               'lpy':np.float32,
               'sx':np.float32,
               'sy':np.float32,
               #
               'nn_d':np.float32,
               'M':np.uint16,
               }

PROPS_ORDERCOLS = ['conc','group',
                   'tau','A','occ','I','eps',
                   'koff','konc','N','success','success_epsnorm','p','N_p','N_start',
                   'tau_b','tau_d','events','ignore',
                   'frame','std_frame','photons','bg','x','y','lpx','lpy','sx','sy',
                   'nn_d','M',
                   ]

PICKED_DTYPE = {'group':np.uint32,
                'frame':np.uint32,
                'x':np.float32,
                'y': np.float32,
                'photons':np.float32,
                'sx':np.float32,
                'sy':np.float32,
                'bg':np.float32,
                'lpx':np.float32,
                'lpy':np.float32,
                'ellipticity':np.float32,
                'net_gradient':np.float32,
                'photons_ck':np.float32,
                'conc':np.uint32,
                'M':np.uint16,
                }

PICKED_COLS = [field for field in PICKED_DTYPE]

#%%
def combine(df,NoFrames,ignore,weights):
    ''' 
    Combine pick properties (fcs,other,qpaint) and find solution for (koff,konc,N) based on properties.
    ''' 
    ### Compute individual pick based properties
    s_fcs = fcs.props_fcs(df,NoFrames)
    s_other = other.props_other(df,NoFrames)
    s_qpaint = qpaint.props_qpaint(df,ignore)
    
    ### Combine properties
    s_props = pd.concat([s_fcs,s_other,s_qpaint])
    
    ### Find solution based on perties
    s_sol = solve.solve(s_props,weights)
    
    ### Combine properties and solution
    s_out = pd.concat([s_props,s_sol])
    s_out = s_out[[col for col in PROPS_ORDERCOLS if col not in ['conc', 'group', 'nn_d', 'M']]]
    
    return s_out

#%%
def apply_combine(df,NoFrames,ignore,weights): 
    """
    Applies pick_props.get_props(df,NoFrames,ignore) to each group in non-parallelized manner. Progressbar is shown under calculation.
    """
    df=df.set_index('group')
    tqdm.pandas() # For progressbar under apply
    df_props=df.groupby('group').progress_apply(lambda df: combine(df,NoFrames,ignore,weights))

    return df_props

#%%
def apply_combine_dask(df,NoFrames,ignore,weights): 
    """
    Applies pick_props.get_props(df,NoFrames,ignore) to each group in parallelized manner using dask by splitting df into 
    various partitions.
    """
     
    ########### Define apply_props for dask which will be applied to different partitions of df
    def apply_combine_2part(df,NoFrames,ignore,mode): return df.groupby('group').apply(lambda df: combine(df,NoFrames,ignore,weights))

    ########## Partinioning and computing
    t0=time.time() # Timing
    
    ### Set up DataFrame for dask
    df=df.set_index('group') # Set group as index otherwise groups will be split during partition!!! 
    NoPartitions=max(1,int(0.8 * mp.cpu_count()))
    df=dd.from_pandas(df,npartitions=NoPartitions)                
        
    ### Compute using running dask cluster, if no cluster is running dask will start one with default settings (maybe slow since not optimized for computation!)
    df_props=df.map_partitions(apply_combine_2part,NoFrames,ignore,weights).compute()
    
    dt=time.time()-t0
    print('... Computation time %.1f s'%(dt)) 
    
    return df_props


#%%
def cluster_setup_howto():
    '''
    Print instruction howto start a DASK local cluster for efficient computation of apply_props_dask().
    Fixed ``scheduler_port=8787`` is used to easily reconnect to cluster once it was started.
    
    '''

    print('Please first start a DASK LocalCluster by running following command in directly in IPython shell:')
    print()
    print('Client(n_workers=max(1,int(0.8 * mp.cpu_count())),')
    print('       processes=True,')
    print('       threads_per_worker=1,')
    print('       scheduler_port=8787,')
    print('       dashboard_address=":1234")') 
    return


#%%
def main(locs,info,path,conc,**params):
    '''
    Get immobile properties for each group in _picked.hdf5 file (see `picasso.addon`_) and filter.
    
    
    Args:
        locs(pandas.DataFrame):    Grouped localization list, i.e. _picked.hdf5 as in `picasso.addon`_
        info(list):                Info _picked.yaml to _picked.hdf5 localizations as list of dictionaries.
        path(str):                 Path to _picked.hdf5 file.
        cond(float):               Experimental condition description
        
    Keyword Args:
        ignore(int=1):             Maximum interruption (frames) allowed to be regarded as one bright time.
        parallel(bool=False):      Apply parallel computing using DASK? Local cluster should be started before according to cluster_setup_howto()
    
    Returns:
        list:
            
        - [0](dict):             Dict of keyword arguments passed to function.
        - [1](pandas.DataFrame): Immobile properties of each group in ``locs`` as calulated by apply_props()
    '''
    
    ### Path of file that is processed and number of frames
    path=os.path.splitext(path)[0]
    NoFrames=info[0]['Frames']
    
    ### Define standard 
    standard_params={'ignore': 1,
                     'parallel': False,
                     'weights':[1,1,1,1],
                     }
    ### Set standard if not contained in params
    for key, value in standard_params.items():
        try:
            params[key]
            if params[key]==None: params[key]=standard_params[key]
        except:
            params[key]=standard_params[key]
    
    ### Remove keys in params that are not needed
    delete_key=[]
    for key, value in params.items():
        if key not in standard_params.keys():
            delete_key.extend([key])
    for key in delete_key:
        del params[key]
        
    ### Procsessing marks: extension&generatedby
    params['generatedby']='lbfcs.pickprops.main()'
    
    ##################################### Calculate kinetic properties
    print('Calculating kinetic information ...')
    if params['parallel']==True:
        print('... in parallel')
        locs_props=apply_combine_dask(locs,
                                      NoFrames,
                                      params['ignore'],
                                      params['weights'],
                                      )
    else:
        locs_props=apply_combine(locs,
                                 NoFrames,
                                 params['ignore'],
                                 params['weights'],
                                 )
    
    ##################################### Assign nearest neighbor distance
    data = locs_props[['x','y']].values.astype(np.float32)
    tree = scipy.spatial.cKDTree(data)              # Build up tree
    d,idx = tree.query(data,k=[2])                  # Query data for nearest neigbor (not self, i.e. k=2!)
    locs_props = locs_props.assign(nn_d = d)        # Assign to props 
                                   
    ##################################### Some final adjustments assigning conditions and downcasting dtypes wherever possible
    
    locs_props.reset_index(inplace=True) # Assign group to columns
    locs_props = locs_props.assign(conc = conc)
    locs_props = locs_props.assign(M = NoFrames)
    locs_props = locs_props[PROPS_ORDERCOLS]
    locs_props = locs_props.astype(PROPS_DTYPE)
    
    locs = locs.assign(conc = conc)
    locs = locs.assign(M = NoFrames)
    locs = locs.astype({'frame':np.uint16,
                        'group':np.uint16,
                        'conc':np.uint32,
                        'M':np.uint16,
                        })
    
    ### Add evaluation date and user name to params
    params['eval_date'] = datetime.now().strftime('%Y-%m-%d  %H:%M:%S') # Add evaluation date to yaml
    params['user_name'] = getpass.getuser()                             # Add user name to yaml
    
    ##################################### Saving
    print('Saving _props ...')
    info_props=info.copy()+[params]
    addon_io.save_locs(path+'_props.hdf5',
                       locs_props,
                       info_props,
                       mode='picasso_compatible')
    
    print('Saving _picked ...')
    addon_io.save_locs(path+'.hdf5',
                       locs,
                       info,
                       mode='picasso_compatible')
           
    return [params,locs_props]










