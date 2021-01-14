import numpy as np
import pandas as pd
import re
import numba
import time
import warnings

from tqdm import tqdm
from fast_histogram import histogram1d
from scipy.special import binom
import scipy.optimize as optimize

import lbfcs.visualizeseries as visualize

warnings.filterwarnings("ignore")
tqdm.pandas()

### Global variables
PROPS_USEFIT_COLS = ['vary','tau_lin','A_lin','n_locs','tau_d','n_events'] + ['ignore','M']

OBSSOL_COLS = ['setting','vary','rep','group','tau','A','occ','taud','events'] + ['koff','konc','N','n_points','success'] + ['B','eps','snr','sx','sy'] + ['ignore','M']
OBS_COLS = ['setting','vary','rep','group','tau_lin','A_lin','n_locs','tau_d','n_events','B','bg','sx','sy'] + ['ignore','M']
OBS_COLS_NEWNAME = ['setting','vary','rep','group','tau','A','occ','taud','events','B','snr','sx','sy'] +  ['ignore','M']

OBS_ENSEMBLE_USEFIT_COLS = ['vary','tau','A','occ','taud','events'] + ['ignore','M']
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
    Create final output for individual observables and solution.
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
def obs_ensemble(df):
    '''
    Prepare ensemble data from obsol of individual groups.
    '''
    ### Init
    s_out = pd.Series(index=OBSSOL_COLS, dtype=np.float32)
    
    ### Compute ensemble observables
    for c in OBSSOL_COLS:
        if c in ['tau','occ']:
            s_out[c] = np.nanmean(df.loc[:,c])
        elif c in ['setting','vary','rep','ignore','M']:
            s_out[c] = df[c].iloc[0]
        else:
            s_out[c] = np.nanmedian(df.loc[:,c])
    
    s_out['group'] = len(df) # Number of groups
    
    return s_out

#%%
def obsol_ensemble(obsol,weights):
    '''
    Create final output for ensemble observables and solution.
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

