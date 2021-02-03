import sys
import glob
import os
import numpy as np
import pandas as pd
import numba
import scipy.optimize as optimize
import warnings
from tqdm import tqdm
tqdm.pandas()

import picasso.io as io
import picasso_addon.io as addon_io

warnings.filterwarnings("ignore")

### Columns names used for preparing fit data from props
PROPS_USEFIT_COLS = ['vary','tau_lin','A_lin','occ','I','var_I','tau_d','n_events','ignore','M']

### Columns of obsol
OBSSOL_COLS = (['setting','vary','rep','group']
               +['tau','A','occ','I','var_I','taud','events'] 
               + ['koff','konc','N','eps','n_points','success']
               + ['snr','sx','sy']
               + ['x','y','nn_d','frame','std_frame']
               + ['ignore','M','exp','date'])
### Columns transferred from props to obsol
OBS_COLS = (['setting','vary','rep','group']
            + ['tau_lin','A_lin','occ','I_ck','var_I_ck','tau_d','n_events']
            + ['bg','sx','sy']
            + ['x','y','nn_d','frame','std_frame']
            + ['ignore','M','date'])
### New names of columns transferred from props
OBS_COLS_NEWNAME = (['setting','vary','rep','group'] 
                    + ['tau','A','occ','I','var_I','taud','events']
                    + ['snr','sx','sy']
                    + ['x','y','nn_d','frame','std_frame']
                    + ['ignore','M','date'])
### Columns names used for preparing fit data from obs
OBS_USEFIT_COLS = ['vary','tau','A','occ','I','var_I','taud','events','ignore','M']

### Dict for obsol type conversion
OBSOL_TYPE_DICT = {'setting':np.uint16,
                                   'vary':np.uint16,
                                   'rep':np.uint8,
                                   'group':np.uint16,
                                   #
                                   'tau':np.float32,
                                   'A':np.float32,
                                   'occ':np.float32,
                                   'I':np.float32,
                                   'var_I':np.float32,
                                   'taud':np.float32,
                                   'events':np.float32,
                                   #
                                   'koff':np.float32,
                                   'konc':np.float32,
                                   'N':np.float32,
                                   'eps':np.float32,
                                   'n_points':np.uint8,
                                   'success':np.uint8,
                                   #
                                   'snr':np.float32,
                                   'sx':np.float32,
                                   'sy':np.float32,
                                   #
                                   'x':np.float32,
                                   'y':np.float32,
                                   'nn_d':np.float32,
                                   'frame':np.float32,
                                   'std_frame':np.float32,  
                                   #
                                   'ignore':np.uint8,
                                   'M':np.uint16,
                                   'exp':np.float32,
                                   'date':np.uint32,
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
                         columns=['setting','vary','rep','file_name'])
    ### Assign file_names
    file_names = [os.path.split(path)[-1] for path in paths]
    files.loc[:,'file_name'] = file_names
    ### Assign setting
    settings = props.groupby(['rep']).apply(lambda df: df.setting.iloc[0])
    files.loc[:,'setting'] = settings.values
    ### Assign vary
    varies = props.groupby(['rep']).apply(lambda df: df.vary.iloc[0])
    files.loc[:,'vary'] = varies.values
    ### Assign rep
    files.loc[:,'rep'] = range(len(paths))
    
    ### Sort file list
    files = files.sort_values(by=['setting','vary','rep'])
    ### Assign date to props
    props = props.assign(date = np.min(dates)) # Assign earliest aquisition date of measurement series
    
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
    obs = ['tau_lin','A_lin','occ','I','var_I','I_ck','var_I_ck'] 
    
    df = df[np.all(np.isfinite(df[obs]),axis=1)]                # Remove NaNs from all observables
    df = df[(np.abs(df.tau_lin-df.tau)/df.tau_lin) < 0.2]  # Deviation between two ac-fitting modes should be small
    
    df = it_medrange(df,'frame'  ,[1.3,1.3])
    df = it_medrange(df,'std_frame'  ,[1.25,100])
    
    df = it_medrange(df,'tau_lin'  ,[100,2])
    df = it_medrange(df,'tau_lin'  ,[2,2])
    
    df = it_medrange(df,'A_lin'  ,[100,5])
    df = it_medrange(df,'A_lin'    ,[2,5])
                     
    df = it_medrange(df,'occ'   ,[5,5])
    
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
def p_func(koff,konc,corr): 
    p = ( 1/koff + corr ) / ( 1/koff + 1/konc )
    return p

@numba.jit(nopython=True, nogil=True, cache=False)
def occ_func(koff,konc,N,occ_meas):
    p = p_func(koff,konc,1)
    occ = 1 - np.abs(1-p)**N
    occ = occ - occ_meas
    return occ

@numba.jit(nopython=True, nogil=True, cache=False)
def I_func(koff,konc,N,eps,I_meas):
    p = p_func(koff,konc,0)
    I = eps * N * p
    I = I - I_meas
    return I

@numba.jit(nopython=True, nogil=True, cache=False)
def var_I_func(koff,konc,N,eps,var_I_meas):
    p = p_func(koff,konc,0)
    var_I = eps**2 * N * p * (1-p)
    var_I = var_I - var_I_meas
    return var_I

@numba.jit(nopython=True, nogil=True, cache=False)
def taud_func(konc,N,taud_meas): return 1/(N*konc) - taud_meas

@numba.jit(nopython=True, nogil=True, cache=False)
def events_func(frames,ignore,koff,konc,N,events_meas):
    p = p_func(koff,konc,1)
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
    
    ### Unkown in system: x = [konc_1,konc_2,...,konc_n,koff,N,eps]
    ###  -> x can only be positive
    x = np.abs(x) 
        
    ### Define weights for fitting, i.e. residual will be divided by (data_meas/(weight*100)).
    ### This means for weight = 1 a standard deviation of 1% is assumed in data_meas.
    w = weights * 100
    
    ### Initialize equation system consisting of 5 equations (i.e: tau,A,occ,I,var_I) for every measurement
    system = np.zeros(len(data)*len(weights))
    eq_idx = 0
    
    for i in range(n): # i corresponds to konc_i
        cs_idx = np.where(data[:,0] == cs[i])[0]  # Get indices of datapoints belonging to konc_i
        
        for c_idx in cs_idx:
            
            system[eq_idx]      = ( w[0] / data[c_idx,1] ) * tau_func(x[n],x[i],data[c_idx,1])
            system[eq_idx+1] = ( w[1] / data[c_idx,2] ) * A_func(x[n],x[i],x[n+1],data[c_idx,2])
            system[eq_idx+2] = ( w[2] / data[c_idx,3] ) * occ_func(x[n],x[i],x[n+1],data[c_idx,3])
            
            system[eq_idx+3] = ( w[3] / data[c_idx,4] ) * I_func(x[n],x[i],x[n+1],x[n+2],data[c_idx,4])
            system[eq_idx+4] = ( w[4] / data[c_idx,5] ) * var_I_func(x[n],x[i],x[n+1],x[n+2],data[c_idx,5])
            
            system[eq_idx+5] = ( w[5] / data[c_idx,6] ) * taud_func(x[i],x[n+1],data[c_idx,6])
            system[eq_idx+6] = ( w[6] / data[c_idx,7] ) * events_func(data[c_idx,-1],data[c_idx,-2],x[n],x[i],x[n+1],data[c_idx,7])
            
            eq_idx += 7
    
    ### Remove NaNs, can only occur in taud and events!
    system[np.isnan(system)] = 0
    
    return system

#%%
@numba.jit(nopython=True, nogil=True, cache=False)
def estimate_unknowns(data,weights):
    '''
    Get estimate x0 for unknowns x based on data to feed into fit as initial value x0.
    Return unique concentration points in data.
    '''
    cs = np.unique(data[:,0])                                   # 1st row of data corresponds to concentration
    n = len(cs)                                                          # Unique concentrations equals number of koncs for system
    fit_eps = (weights[3] > 0) | (weights[4] > 0)     # Fit eps value if I or var_I have contributions in equation system
    
    x0 = np.zeros(n+3,dtype=np.float32)           # Init estimate
    
    ### Estimate for koncs
    for i,c in enumerate(cs): x0[i] = 5e6 * (c * 1e-12) * 0.4
    ### Estimate for koff
    tau_max = np.max(data[data[:,0]==cs[0],1]) # Maximum tau @ lowest concentration
    x0[n] = 1 / tau_max
    ### Estimate for N
    x0[n+1] = 4
    ### Estimate for eps
    if fit_eps:
        x0[n+2] = 400
    else:
        x0[n+2] = 0
    
    return x0,cs,n
    
#%%
def solve_eqs(data,weights):
    '''
    Solve equation system as set-up by ``create_eqs()`` for c-series.. 
    '''
    weights = np.array(weights,dtype=np.float32)
    
    ### Get estimate x0 and unique concentrations cs and number of unique concentrations n
    x0,cs,n = estimate_unknowns(data,weights)
    
    ### Solve system of equations
    xopt = optimize.least_squares(lambda x: create_eqs(x,data,weights),
                                                     x0,
                                                     method ='trf',
                                                     ftol=1e-4,
                                                     )
    x = np.abs(xopt.x)
    success = xopt.success
    
    return x, success, cs, n

#%%
def snr_func(eps,bg,sx,sy):
    snr = eps.copy()      # Total photons
    snr /= 2*np.pi*sx*sy  # Now snr corresponds to amplitude of 2D gaussian fit
    snr /= bg             # (maximum) signal to noise ratio defined as amplitude/bg
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
                       dtype = np.float64,
                       )
    
    ### Assign observables
    obs[OBS_COLS_NEWNAME] = df[OBS_COLS].values
    ### Assign solution
    obs.koff = x[n]
    obs.N = x[n+1]
    obs.eps = x[n+2]                                                   # If 0 weights were assigned to both I and var_I eps = 0 at this point -> see also estimate_unkowns()
    for i,c in enumerate(cs): obs.loc[obs.vary==c,'konc'] = x[i]
    obs.n_points = len(df)
    obs.success = int(success)
    
    ### Assign photon related observables
    fit_eps = (weights[3] > 0) | (weights[4] > 0)          # Was eps fitted by using either I or var_I?
    if not fit_eps:
        obs.eps = obs.I / (obs.N * p_func(obs.koff.values,obs.konc.values,0))
    obs.snr = snr_func(obs.eps,obs.snr,obs.sx,obs.sy) # Before this line value of snr corresponded to bg!
    
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
def combine(df_in,success_only):
    '''
    Prepare ensemble data from obsol of individual groups.
    '''
    ### Init
    s_out = pd.Series(index=OBSSOL_COLS)
    
    ### Reduce to groups which could be successfully fitted
    if success_only: df = df_in[df_in.success == 1]
    else: df = df_in.copy()
    
    ### Compute ensemble observables
    for c in OBSSOL_COLS:
        if c in ['tau','occ','n_points','success']:
            s_out[c] = np.nanmean(df.loc[:,c])
        elif c in ['setting','vary','rep','ignore','M','exp','date']:
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
    df = obsol_per_group.groupby(['date','setting','exp','vary','rep']).apply(lambda df: combine(df,success_only=True))
    df = df.droplevel( level = ['date','exp','vary','rep'])
    df = df.drop(columns = ['setting'])
    df = df.reset_index()
    
    ### Some type conversions
    df = df.astype(OBSOL_TYPE_DICT)
    
    return df

#%%
def print_solutions(df):
    df = df.sort_values(by=['date','setting','exp','vary','rep'])
    print()
    print('     date     |           id            |   groups  |   koff [1/s]  |  kon [1/Ms] |     N')
    for i in range(len(df)):
        s = df.iloc[i]
        date = '%s  '%str(int(s.date))
        id_str = '(%s,%s,%s)  '%(str(int(s.setting)).zfill(3),str(int(s.vary)).zfill(5),str(int(s.rep)).zfill(2))
        groups_str = '   %s     '%str(int(s.group)).zfill(4)
        koff_str = '  %.2fe-01   '%(s.koff*(1/s.exp)*1e1)
        kon_str = ' %.2fe+07   '%(s.konc*(1/s.vary)*1e12*(1/s.exp)*1e-7)
        N_str = ' %.2f '%(s.N)
        
        print(date,id_str, groups_str,koff_str, kon_str,N_str)

#%%
def save_series(files, obs, params):
    
    w_str = ''
    for w in params['weights']: w_str += str(w)
    
    savepath_obs = os.path.join(params['dir_name'],'_obsol-%s.hdf5'%w_str)
    savepath_files = os.path.join(params['dir_name'],'_files.csv')
    
    addon_io.save_locs(savepath_obs,
                                   obs,
                                   [params],
                                   mode='picasso_compatible')
    files.to_csv(savepath_files, index=False)
    
#%%

##############################
'''
Ensemble solving
'''
##############################
#%%
'''
In progress: Old method ensemble tau vs. c fitting
'''
def tau_conc_func(x,koff,kon): return 1/(koff+kon*x)

def estimate_koff_kon(data):
    cs = data[:,0]
    
    c_min = np.min(cs)
    c_max = np.max(cs)
    tau_max = np.max(data[data[:,0]==c_min,1]) # Maximum tau @ lowest concentration
    tau_min = np.max(data[data[:,0]==c_max,1]) # Minimum tau @ highest concentration
    
    koff = 1/tau_max
    kon = -koff**2 * ((tau_min-tau_max)/(c_min-c_max))
    
    return [kon, koff]
    
def solve_eqs_old(data):
    x = data[:,0]
    y = data[:,1]
    p0 = estimate_koff_kon(data)
    try:
        popt,pcov = optimize.curve_fit(tau_conc_func,x,y,p0=p0,method='trf')
        popt = np.abs(popt)
        success = 1
    except:
        popt = np.ones(2) * np.nan
        success = 0
        
    return popt, success
#%%
def obsol_ensemble_combine(obsol,weights):
    '''
    Combine group-wise observables to ensemble observables and solve.
    '''
    ### For every setting group according to (vary,rep) and combine observables
    df = obsol.groupby(['vary','rep']).apply(lambda df: combine(df,success_only=False))
    
    ### Prepare output
    setting = df.setting.values
    obs = pd.DataFrame([],
                       index = setting,
                       columns = OBSSOL_COLS,
                       dtype=np.float64)
    
    ### Assign observables
    obs[OBSSOL_COLS] = df[OBSSOL_COLS].values
    obs.n_points = len(df)
    
    ### Prepare data for fitting of ensemble 
    data = df.loc[:,OBS_USEFIT_COLS].values.astype(np.float32)
    
    ### Solve equation system
    if np.all(np.isfinite(weights)):                                                    # New fitting approach if wieghts are properly defined
        x, success, cs, n = solve_eqs(data,weights)  
        ### Assign solution
        obs.koff = x[n]
        obs.N = x[n+1]
        obs.eps = x[n+2]
        for i,c in enumerate(cs): obs.loc[obs.vary==c,'konc'] = x[i]
        obs.success = int(success)
    else:                                                                                         # Old fitting approach for ill defined weights
        popt,success = solve_eqs_old(data)
        obs.koff = popt[0]
        obs.konc = popt[1] * obs.vary
        obs.N = np.nan
        obs.eps = np.nan
        
    return obs

#%%
def N_from_rates(df,cols):
    
    koff = df.koff.values
    konc = df.konc.values
    eps = df.eps.values
    p0 = p_func(koff,konc,0)
    p1 = p_func(koff,konc,1)
    
    A = df.A.values
    occ = df.occ.values
    I = df.I.values
    
    Ns = np.zeros((len(df),3))
    Ns[:,0] = (1 / A) * (koff / konc)
    Ns[:,1] = np.log(1 - occ) / np.log(1 - p1)
    Ns[:,2] = I / (eps * p0)

    Ns[np.isinf(Ns)] = np.nan
    N = np.nanmean(Ns[:,cols], axis = 1)
    
    return N
    
#%%
def apply_ensemble(obs,obs_ensemble,Ncols):
    '''
    Apply rates koff&konc obtained from ensemble solution to obsol to compute group-wise N (using A) and eps/snr (using I).
    '''
    df = obs.copy()
    
    ### Search for correct measurement in ensemble solution (obs_ensemble) and get rates
    date = obs.date.iloc[0]
    setting = obs.setting.iloc[0]
    vary = obs.vary.iloc[0]
    rep = obs.rep.iloc[0]
    s = obs_ensemble.query('date==@date and setting==@setting and vary==@vary and rep==@rep')
    
    koff = float(s.koff)
    konc = float(s.konc)
    
    ### Assign ensemble rates koff&konc to groups
    df.koff = koff
    df.konc = konc
    ### Assign eps and snr based on rates
    ### First we have to convert snr back to bg 
    df.snr = (df.eps.values / (2*np.pi*df.sx.values*df.sy.values) ) * (1/df.snr.values)            # Now snr corresponds to bg again
    df.eps = df.var_I.values / (df.I.values * (1 - p_func(df.koff.values,df.konc.values,0)) )   # Assign new eps based on ensemble rates
    df.snr = snr_func(df.eps.values,df.snr.values,df.sx.values,df.sy.values)                                                   # Assign new snr based on new eps
    ### Assign N based on rates
    df.N = N_from_rates(df,Ncols)
    
    return df

#%%
def get_obsols_ensemble(obs, weights = [1,1,1,1,0,0,0], Ncols = [0,1,2]):
    '''
    Groupby and apply obsol_ensemble(). For the ensemble solution always complete series for one setting is used.
    '''
    ### Get ensemble solution
    obs_ensemble_combined = obs.groupby(['date','setting']).apply(lambda df: obsol_ensemble_combine(df,weights))
    obs_ensemble_combined = obs_ensemble_combined.droplevel( level = ['date','setting'])
    obs_ensemble_combined = obs_ensemble_combined.astype(OBSOL_TYPE_DICT)
    
    ### Assign ensemble rates to group-wise solution and calulate group-wise N, eps&snr using ensemble rates
    obs_ensemble = obs.groupby(['date','setting','vary','rep']).apply(lambda df: apply_ensemble(df,obs_ensemble_combined,Ncols))
    obs_ensemble = obs_ensemble.droplevel( level = ['date','setting','vary','rep'])
    obs_ensemble = obs_ensemble.astype(OBSOL_TYPE_DICT)
    
    ### Combine again to update N,eps&snr
    obs_ensemble_combined = obs_ensemble.groupby(['date','setting','exp','vary','rep']).apply(lambda df: combine(df,success_only=False))
    obs_ensemble_combined = obs_ensemble_combined.droplevel( level = ['date','exp','vary','rep'])
    obs_ensemble_combined = obs_ensemble_combined.drop(columns = ['setting'])
    obs_ensemble_combined = obs_ensemble_combined.reset_index()
    obs_ensemble_combined = obs_ensemble_combined.astype(OBSOL_TYPE_DICT)
    
    return obs_ensemble, obs_ensemble_combined

    