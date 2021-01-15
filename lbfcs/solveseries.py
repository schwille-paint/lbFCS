import sys
import glob
import os
import re
import numpy as np
import pandas as pd
import numba
import scipy.optimize as optimize
import warnings
from tqdm import tqdm
tqdm.pandas()

import picasso.io as io

warnings.filterwarnings("ignore")

### Global variables
PROPS_USEFIT_COLS = ['vary','tau_lin','A_lin','n_locs','tau_d','n_events'] + ['ignore','M']

OBSSOL_COLS = ['setting','vary','rep','group','tau','A','occ','taud','events'] + ['koff','konc','N','n_points','success'] + ['B','eps','snr','sx','sy'] + ['ignore','M','exp']
OBS_COLS = ['setting','vary','rep','group','tau_lin','A_lin','n_locs','tau_d','n_events','B','bg','sx','sy'] + ['ignore','M']
OBS_COLS_NEWNAME = ['setting','vary','rep','group','tau','A','occ','taud','events','B','snr','sx','sy'] +  ['ignore','M']

OBS_ENSEMBLE_USEFIT_COLS = ['vary','tau','A','occ','taud','events'] + ['ignore','M']

OBSOL_TYPE_DICT = {'setting':np.uint16,
                                   'vary':np.uint16,
                                   'rep':np.uint8,
                                   'group':np.uint16,
                                   'M':np.uint16,
                                   'ignore':np.uint8,
                                    'n_points':np.uint8,
                                    'success':np.uint8,
                                    'koff':np.float32,
                                    'konc':np.float32,
                                    'N':np.float32,
                                    'exp':np.float32,
                                    'eps':np.float32,
                                    'snr':np.float32,
                                    }

#%%
def load_props_in_dir(dir_name):
    '''
    Load all _props.hdf5 files indirectory and return as combined pandas.DataFrame. Also returns comprehensive list of loaded files
    '''
    ### Get sorted list of all paths to props in dir_name
    paths = sorted( glob.glob( os.path.join( dir_name,'*_props*.hdf5') ) )
    ### Load files
    props = pd.concat([pd.DataFrame(io.load_locs(p)[0]) for p in paths],keys=range(len(paths)),names=['rep'])
    props = props.reset_index(level=['rep'])
    ### Create comprehensive list of loaded files
    files = pd.DataFrame([],
                         index=range(len(paths)),
                         columns=['setting','vary','rep','file_name'])
    
    file_names = [os.path.split(path)[-1] for path in paths]
    files.loc[:,'file_name'] = file_names

    settings = props.groupby(['rep']).apply(lambda df: df.setting.iloc[0])
    files.loc[:,'setting'] = settings.values
    
    varies = props.groupby(['rep']).apply(lambda df: df.vary.iloc[0])
    files.loc[:,'vary'] = varies.values
    
    files.loc[:,'rep'] = range(len(paths))
    
    files = files.sort_values(by=['setting','vary','rep'])
    
    return props, files

#%%
def prefilter(df_in):
    '''
    Filter props.
    '''
    def it_medrange(df,field,ratio):
        for i in range(3):
            med    = np.median(df[field].values)
            if field == 'frame':
                med = np.unique(df.M)[0]/2
            istrue = (df[field] > ((1/ratio[0])*med)) & (df[field] < (ratio[1]*med))
            df     = df[istrue]
        return df
    
    df = df_in.copy()
    obs = ['tau_lin','A_lin','n_locs','tau_d','n_events'] 
    
    df = df[np.all(np.isfinite(df[obs]),axis=1)]                # Remove NaNs from all observables
    df = df[(np.abs(df.tau_lin-df.tau)/df.tau_lin) < 0.2]  # Deviation between two ac-fitting modes should be small
    
    df = it_medrange(df,'frame'  ,[1.3,1.3])
    df = it_medrange(df,'std_frame'  ,[1.25,100])
    
    df = it_medrange(df,'tau_lin'  ,[100,2])
    df = it_medrange(df,'tau_lin'  ,[2,2])
    
    df = it_medrange(df,'A_lin'  ,[100,5])
    df = it_medrange(df,'A_lin'    ,[2,5])
                     
    df = it_medrange(df,'n_locs'   ,[5,5])
    
    return df

#%%
def exclude_filter(props_init,exclude_rep):
    '''
    Exclude repetitions if needed and filter each measurement.
    '''
    ### Select subset of props
    props = props_init.query('rep not in @exclude_rep' )
    
    ### Filter props
    props = props.groupby(['rep']).apply(prefilter)
    props = props.droplevel( level = ['rep'])
    
    return props

#%%
def prep_data(df):
    '''
    Pepare necesary observables as numpy.array ``data`` for setting and solving equation set.
    '''
    data = df.loc[:,PROPS_USEFIT_COLS].values.astype(np.float32)
         
    return data

#%%
### Analytic expressions for observables

@numba.jit(nopython=True, nogil=True, cache=False)
def tau_func(koff,konc,tau_meas): return 1 / (koff+konc) - tau_meas

@numba.jit(nopython=True, nogil=True, cache=False)
def A_func(koff,konc,N,A_meas): return koff / (konc*N) - A_meas

@numba.jit(nopython=True, nogil=True, cache=False)
def occ_func(koff,konc,N,occ_meas):
    p   = ( 1/koff + 1 ) / ( 1/koff + 1/konc ) # Probability of bound imager
    occ = 1 - np.abs(1-p)**N
    occ = occ - occ_meas
    return occ

@numba.jit(nopython=True, nogil=True, cache=False)
def taud_func(konc,N,taud_meas): return 1/(N*konc) - taud_meas

@numba.jit(nopython=True, nogil=True, cache=False)
def events_func(frames,ignore,koff,konc,N,events_meas):
    p       = ( 1/koff + 1 ) / ( 1/koff + 1/konc )   # Probability of bound imager
    darktot = np.abs(1-p)**N * frames             # Expected sum of dark times
    taud    = taud_func(konc,N,0)                     # Mean dark time
    events  = darktot / (taud + ignore +.5)      # Expected number of events
    return events - events_meas

#%%
@numba.jit(nopython=True, nogil=True, cache=False)
def create_eqs(x,data, weights):
    '''
    Set-up equation system for varying koncs with given ``data`` as returned by ``prep_data()``. 
    '''
    ### Extract number of unknowns in system, by counting number of unique concentrations
    cs = np.unique(data[:,0])   # 1st row of data corresponds to concentration
    n = len(cs)                         # Unique concentrations equals number of koncs for system
    
    ### Unkown in system: x = [konc_1,konc_2,...,konc_n,koff,N]
    ###  -> x can only be positive
    x = np.abs(x) 
        
    ### Define weights for fitting, i.e. residual will be divided by (data_meas/(weight*100)).
    ### This means for weight = 1 a standard deviation of 1% is assumed in data_meas.
    w = weights * 100
    
    ### Initialize equation system consisting of 5 equations (i.e: tau,A,occ,tau_d,n_events) for every measurement
    system = np.zeros(len(data)*len(weights))
    eq_idx = 0
    
    for i in range(n): # i corresponds to konc_i
        cs_idx = np.where(data[:,0] == cs[i])[0]  # Get indices of datapoints belonging to konc_i
        
        for c_idx in cs_idx:
            
            system[eq_idx]      = ( w[0] / data[c_idx,1] ) * tau_func(x[n],x[i],data[c_idx,1])
            system[eq_idx+1] = ( w[1] / data[c_idx,2] ) * A_func(x[n],x[i],x[n+1],data[c_idx,2])
            system[eq_idx+2] = ( w[2] / data[c_idx,3] ) * occ_func(x[n],x[i],x[n+1],data[c_idx,3])
            system[eq_idx+3] = ( w[3] / data[c_idx,4] ) * taud_func(x[i],x[n+1],data[c_idx,4])
            system[eq_idx+4] = ( w[4] / data[c_idx,5] ) * events_func(data[c_idx,-1],data[c_idx,-2],x[n],x[i],x[n+1],data[c_idx,5])
            
            eq_idx += 5
            
    return system

#%%
@numba.jit(nopython=True, nogil=True, cache=False)
def estimate_unknowns(data):
    '''
    Get estimate x0 for unknowns x based on data to feed into fit as initial value x0.
    Return unique concentration points in data.
    '''
    cs = np.unique(data[:,0])                                # 1st row of data corresponds to concentration
    n = len(cs)                                                      # Unique concentrations equals number of koncs for system
    x0 = np.zeros(n+2,dtype=np.float32)           # Init estimate
    ### Estimate for koncs
    for i,c in enumerate(cs): x0[i] = 5e6 * (c * 1e-12) * 0.4
    ### Estimate for koff
    tau_max = np.max(data[data[:,0]==cs[0],1]) # Maximum tau @ lowest concentration
    x0[n] = 1 / tau_max
    ### Estimate for N
    x0[n+1] = 4 
    
    return x0,cs,n
    
#%%
def solve_eqs(data,weights):
    '''
    Solve equation system as set-up by ``create_eqs()`` for c-series.. 
    '''
    weights = np.array(weights,dtype=np.float32)
    
    ### Get estimate x0 and unique concentrations cs and number of unique concentrations n
    x0,cs,n = estimate_unknowns(data)
    
    ### Solve system of equations
    xopt = optimize.least_squares(lambda x: create_eqs(x,data,weights),
                                                     x0,
                                                     method ='trf',
                                                     ftol=1e-4,
                                                     )
    x = xopt.x
    success = xopt.success
    
    return x, success, cs, n

#%%
def eps_func(B,koff,tau):
    b0 = 0.9973
    b1 = -0.315
    eps = B / (koff * (b0*tau + b1) )
    return eps

#%%
def snr_func(eps,bg,sx,sy):
    snr = eps # Total photons
    snr /= 2*np.pi*sx*sy # Now snr corresponds to amplitude of 2D gaussian fit
    snr /= bg # (maximum) signal to noise ratio defined as amplitude/bg
    return snr

#%%
def obsol(df,weights):
    '''
    Create final output per group including observables and solution.
    '''
    ### Extract observables
    data = prep_data(df)
    
    ### Solve equation system
    x, success, cs, n = solve_eqs(data,weights)
    
    ### Prepare output
    groups = df.group.values.astype(np.int16) # Group used as index
    obs = pd.DataFrame([],
                       index = groups,
                       columns = OBSSOL_COLS,
                       dtype=np.float32)
    
    ### Assign observables
    obs[OBS_COLS_NEWNAME] = df[OBS_COLS].values
    ### Assign solution
    obs.koff = x[n]
    obs.N = x[n+1]
    for i,c in enumerate(cs): obs.loc[obs.vary==c,'konc'] = x[i]
    obs.n_points = len(df)
    obs.success = int(success)
    ### Assign photon related observables
    obs.eps = eps_func(obs.B, obs.koff, obs.tau)
    obs.snr = snr_func(obs.eps,obs.snr,obs.sx,obs.sy) # At this point bg is assigned to snr!
    
    return obs

#%%
def get_obsol(props, exp, weights, solve_mode):
    '''
    Groupby and apply obsol().
    
    If mode = single all groups of each measurment are solved individually
    
    If mode = series groups with same ids in each measurment are solved as series. 
    This option makes only sense if:
        - all measurements were taken from the same FOV
        - all measurements are aligned to each other
        - and it was taken care of the group identity, i.e. one origami has same group id in all measurements
    
    Exposure (aquisition cycle time) [s] will be assigned for unit conversion.
    '''
    if solve_mode == 'single':
        groupby_cols = ['vary','rep','group']
    elif solve_mode == 'series':
        groupby_cols = ['group']
    else:
        print('Please define solve_mode in get_obsol(). Execution aborted!')
        sys.exit()
    
    print()
    df = props.groupby(groupby_cols).progress_apply(lambda df: obsol(df,weights))
    df = df.droplevel( level = groupby_cols)
    print()
    
    ### Assign exposure
    df.exp =exp
    
    ### Some type conversions
    df = df.astype(OBSOL_TYPE_DICT)
    return df
    
#%%
def obs_ensemble(df):
    '''
    Prepare ensemble data from obsol of individual groups.
    '''
    ### Init
    s_out = pd.Series(index=OBSSOL_COLS, dtype=np.float32)
    
    ### Compute ensemble observables
    for c in OBSSOL_COLS:
        if c in ['tau','occ','n_points','success']:
            s_out[c] = np.nanmean(df.loc[:,c])
        elif c in ['setting','vary','rep','ignore','M','exp']:
            s_out[c] = df[c].iloc[0]
        else:
            s_out[c] = np.nanmedian(df.loc[:,c])
    
    s_out['group'] = len(df) # Number of groups
    
    return s_out
        
#%%
def combine_obsol(obsol_per_group):
    '''
    Group obsol by (setting,rep) and get ensemble means and medians.
    '''
    df = obsol_per_group.groupby(['setting','vary','rep']).apply(lambda df: obs_ensemble(df))
    df = df.droplevel( level = ['vary','rep'])
    df = df.drop(columns = ['setting'])
    df = df.reset_index()
    
    ### Some type conversions
    df = df.astype(OBSOL_TYPE_DICT)
    
    return df

#%%
def print_solutions(df):
    df = df.sort_values(by=['setting','vary','rep'])
    print()
    print('           ID            |   groups  |   koff [1/s]  |  kon [1/Ms] |     N')
    for i in range(len(df)):
        s = df.iloc[i]
        id_str = '(%s,%s,%s)   '%(str(int(s.setting)).zfill(3),str(int(s.vary)).zfill(5),str(int(s.rep)).zfill(2))
        groups_str = '   %s     '%str(int(s.group)).zfill(3)
        koff_str = '  %.2fe-01   '%(s.koff*(1/s.exp)*1e1)
        kon_str = ' %.2fe+07   '%(s.konc*(1/s.vary)*1e12*(1/s.exp)*1e-7)
        N_str = ' %.2f '%(s.N)
        
        print(id_str, groups_str,koff_str, kon_str,N_str)





#%%

##############################
'''
Ensemble solving
'''
##############################

#%%
def obsol_ensemble(obsol,weights):
    '''
    Create final output for ensemble including observables and solution.
    '''
    ### For every setting group according to (vary,rep)
    df = obsol.groupby(['vary','rep']).apply(obs_ensemble)
    
    ### Prepare data for fitting of ensemble 
    data = df.loc[:,OBS_ENSEMBLE_USEFIT_COLS].values.astype(np.float32)
    
    ### Solve equation system
    x, success, cs, n = solve_eqs(data,weights)
    
    ### Prepare output
    setting = df.setting.values
    obs = pd.DataFrame([],
                       index = setting,
                       columns = OBSSOL_COLS,
                       dtype=np.float32)
    
    ### Assign observables
    obs[OBSSOL_COLS] = df[OBSSOL_COLS].values
    obs = obs.rename({'group':'groups'})
    ### Assign solution
    obs.koff = x[n]
    obs.N = x[n+1]
    for i,c in enumerate(cs): obs.loc[obs.vary==c,'konc'] = x[i]
    obs.n_points = len(df)
    obs.success = int(success)
    
    return obs

#%%
def get_obsol_ensemble(obsol_per_group, weights):
    '''
    Groupby and apply obsol_ensemble(). For the ensemble solution always complete series for one setting is used.
    '''
    df = obsol_per_group.groupby(['setting']).apply(lambda df: obsol_ensemble(df,weights))
    df = df.droplevel( level = ['setting'])
    
    return df
