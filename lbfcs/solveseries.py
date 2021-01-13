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

OBS_ALL_COLS = ['setting','vary','rep','group','tau','A','occ','taud','events'] + ['koff','konc','N','n_points','success'] + ['eps','snr'] + ['ignore','M']
PROPS_OBS_COLS = ['setting','vary','rep','group','tau_lin','A_lin','n_locs','tau_d','n_events'] + ['ignore','M']
PROPS_OBS_COLS_NEWNAME = ['setting','vary','rep','group','tau','A','occ','taud','events'] +  ['ignore','M']

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
def solve_eqs(df_in,weights):
    '''
    Solve equation system as set-up by ``create_eq()`` for c-series.. 
    '''
    df = df_in.copy()
    weights = np.array(weights,dtype=np.float32)
    
    ### Exctract observables
    data = prep_data(df)
    
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
    
    ### Prepare output
    obs = pd.DataFrame([],
                       index = range(len(df)),
                       columns = OBS_ALL_COLS)
    ### Assign observables
    obs[PROPS_OBS_COLS_NEWNAME] = df[PROPS_OBS_COLS].values
    ### Assign solution
    obs.koff = x[n]
    obs.N = x[n+1]
    for i,c in enumerate(cs): obs.loc[obs.vary==c,'konc'] = x[i]
    obs.n_points = len(data)
    obs.success = success
    
    return obs

#%%
'''
Work later on assignment of eps and snr since it is still needed to compare with simulations...
'''
# #%%
# def eps_func(B,koff,tau):
#     b0 = 0.9973
#     b1 = -0.315
#     eps = B / (koff * (b0*tau + b1) )
#     return eps

# #%%
# def assign_koff_eps_snr(props_in,sol,intensity='raw'):
    
#     ### Decide which brightness value to use
#     if intensity == 'raw':
#         B_field = 'B_ck'
#     elif intensity == 'chungkennedy':
#         B_field = 'B_ck'
#     elif intensity == 'netgradient':
#         B_field = 'B_ng'
#     else:
#         B_field = 'B_ck'
    
#     ### Get koff from solution @(setting)
#     setting = props_in.setting.iloc[0]
#     koff = float(sol.loc[sol.setting==setting,'koff'] )   
    
#     ### Get mean tau @(setting,vary,rep)
#     tau = np.nanmean(props_in.tau_lin) 
#     # tau = props_in.tau_lin.values                  # Testing individual taus!?! 
    
#     ### Assign koff to props
#     props = props_in.assign(koff = koff)
    
#     ### Assign eps to props using individual groups brightness but global koff & tau
#     eps  = eps_func(props[B_field].values,koff,tau)
#     props = props.assign(eps = eps)
    
#     ### Assign snr to props based on eps, bg, sx and sy
#     snr = eps # Total photons
#     snr /= 2*np.pi*props.sx.values*props.sy.values # Now snr corresponds to amplitude of 2D gaussian fit
#     snr /= props.bg.values # (maximum) signal to noise ratio defined as amplitude/bg
#     props = props.assign(snr = snr)
    
#     return props


#%%
'''
Work later on ensemble solution with bootstrapping ...
'''
# #%%
# def create_parts(props,N_parts,N_samples): 
#     '''
#     Create list of parts. Every part consists of 'N_samples' randomly drawn (repeats within part!) from groups of props. 

#     '''
    
#     df_out = pd.DataFrame([],                                         # Prepare output
#                                          index=range(N_parts),
#                                          columns=range(N_samples))
#     df_out.index.name = 'part'
#     groups = props.group.values                                     # Existing groups in props
    
#     for i in range(N_parts):
#         idx=np.random.randint(0,len(groups),N_samples)    # Draw random samples with possible repetitions
#         df_out.iloc[i,:] = groups[idx]
        
#     return df_out

# #%%
# def observables_from_part(part,props):
    
#     ### Get part in props
#     setting = part.setting.iloc[0]
#     vary = part.vary.iloc[0]
#     rep= part.rep.iloc[0]
#     groups = part.iloc[0,4:].values

#     subset = (props.setting == setting) & (props.vary==vary) & (props.rep==rep)
#     df = props[subset]
#     df = df.query('group in @groups')
    
#     ### Prepare observables
#     s_out = pd.Series(index=OBS_COLUMNS,dtype=np.float32)
    
#     ### Assign standard everything except bound imager probabilities
#     s_out.tau       = np.nanmean(   df.loc[:,'tau_lin'])
#     s_out.A         = np.nanmedian( df.loc[:,'A_lin'])
#     s_out.occ       = np.nanmean(   df.loc[:,'n_locs'])
#     s_out.taud      = np.nanmedian( df.loc[:,'tau_d'])
#     s_out.events    = np.nanmedian( df.loc[:,'n_events'])
    
#     ### Assign measurement duration and ignore value (only needed for taud and events!!)
#     s_out.ignore    = df.loc[:,'ignore'].unique()
#     s_out.M         = df.loc[:,'M'].unique()
    
#     return s_out