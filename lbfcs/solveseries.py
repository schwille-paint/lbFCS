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
PK_COLUMNS = ['p%i'%i for i in range(1,11)]

IMG_BIN_WIDTH = 0.05
IMG_BINS_RANGE= [IMG_BIN_WIDTH /2,
                                11 + IMG_BIN_WIDTH/2]
IMG_BINS = int( (IMG_BINS_RANGE[1] - IMG_BINS_RANGE[0]) / IMG_BIN_WIDTH  )
IMG_BIN_COLUMNS = np.arange(1,IMG_BINS+1,1).astype(np.uint8)

OBS_COLUMNS = ['tau','A','occ','taud','events'] + PK_COLUMNS + ['ignore','M'] + ['eps','snr']
SOL_OBS_COLUMNS = ['vary','tau','A','occ','taud','events'] + PK_COLUMNS + ['ignore','M'] 

#%%
def prefilter_partition(props_in,locs_in, field, select, parts, samples):
    '''
    Filter props and query locs for remaining groups. Create partitions by drawing random samples.

    '''
    ### Select subset of props
    props = props_in.query( field + ' in @select' )
    
    ### Filter props_in
    props = props.groupby(['setting','vary','rep']).apply(prefilter_props)
    props = props.droplevel(level=['setting','vary','rep']) # Remove index after groupby
    
    ### Query locs for groups in props @(setting, vary, rep)
    locs = locs_in.groupby(['setting','vary','rep']).apply(lambda df: prefilter_query_locs(df,props))
    locs = locs.droplevel(level=['setting','vary','rep']) # Remove index after groupby
    
    parts = props.groupby(['setting','vary','rep']).apply(lambda df: create_parts(df,parts,samples))
    parts = parts.reset_index(level=['setting','vary','rep','part'])
    
    return props, locs, parts

#%%
def draw_solve(props,parts,exp,weights=[1,1,1,0,0,1]):
    
    ### Draw: Compute ensemble observables for every part, i.e. @ (setting,vary,rep,part)
    print()
    print('Preparing observables ...')
    obs_parts = parts.groupby(['setting','vary','rep','part']).apply(lambda df: observables_from_part(df,props))
    obs_parts = obs_parts.reset_index(level=['setting','vary','rep','part'])
    
    ### Solve: Find part-wise solution over all vary and rep values, i.e. @(setting,part)
    print('Finding solution ...')
    sol_parts = obs_parts.groupby(['setting','part']).progress_apply(lambda df: solve_eqs(df,weights))
    sol_parts = sol_parts.reset_index(level=['setting','part'])
    
    ### Combine part-wise solutions, i.e. @(setting)
    sol = sol_parts.groupby(['setting']).apply(combine_solutions)
    sol = sol.reset_index(level=['setting'])
    
    ### Print solution
    print()
    print('Solution @ exp = %i ms:' %(int(1000*exp)))
    visualize.print_sol(sol,exp)

    return obs_parts, sol_parts, sol

#%%
def assign_sol_to_props(props_in, sol, locs, intensity='raw'):
    
    ### Assign eps to props using koff@(setting) & tau@(setting,vary,rep)
    props = props_in.groupby(['setting','vary','rep']).apply(lambda df: assign_koff_eps_snr(df,sol,intensity))
    props = props.droplevel(['setting','vary','rep'])
    
    ################ Assign solution to props
    print()
    print('Assigning bound imager histogram ...')
    props = locs.groupby(['setting','vary','rep','group']).progress_apply(lambda df: assign_imagerhist(df,props))
    props = props.drop(['setting','vary','rep','group'],axis=1)
    props = props.reset_index(level=['setting','vary','rep','group'])
    
    return props

#%%
def combine(obs_part):
    
    ### Combine observables for different parts @(setting,vary,rep)
    obs = obs_part.groupby(['setting','vary','rep']).apply(combine_observables)
    obs = obs.reset_index(level=['setting','vary','rep'])
    
    # obs_expect_list = []
    # for sol in sol_list:
    #      obs_expect = obs.groupby('setting').apply(lambda df: expected_observables(df,sol))
    #      obs_expect = obs_expect.droplevel(['setting'])
    #      obs_expect_list.extend([obs_expect])
         
    return obs

#%%
def prefilter_props(df_in):
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
    
    df = df[np.all(np.isfinite(df[obs]),axis=1)]           # Remove NaNs from all observables
    df = df[(np.abs(df.tau_lin-df.tau)/df.tau_lin) < 0.2]  # Deviation between two ac-fitting modes should be small
    
    df = it_medrange(df,'frame'  ,[2,2])
    df = it_medrange(df,'std_frame'  ,[2,100])
    
    df = it_medrange(df,'tau_lin'  ,[100,2])
    df = it_medrange(df,'tau_lin'  ,[2,2])
    
    df = it_medrange(df,'A_lin'  ,[100,5])
    df = it_medrange(df,'A_lin'    ,[2,5])
                     
    df = it_medrange(df,'n_locs'   ,[5,5])
    
    return df

#%%
def prefilter_query_locs(locs, props):
    '''
    Query locs for remaining groups in filtered props.

    '''
    ### Which data?
    setting = locs.setting.iloc[0]
    vary = locs.vary.iloc[0]
    rep = locs.rep.iloc[0]
    
    ### Query locs for groups in props @(setting, vary, rep)
    subset = (props.setting==setting) & (props.vary==vary) & (props.rep==rep)
    groups = props[subset].group.values
    locs = locs.query('group in @groups')
    
    ### Print filtering results
    print('Prefilter&query @(%i,%i,%i): %i '%(setting,vary,rep,len(groups)))
    return locs
    
#%%
def create_parts(props,N_parts,N_samples): 
    '''
    Create list of parts. Every part consists of 'N_samples' randomly drawn (no repeats within part!) from groups of props. 

    '''
    df_out = pd.DataFrame([],                                         # Prepare output
                                         index=range(N_parts),
                                         columns=range(N_samples))
    df_out.index.name = 'part'
    groups = props.group.values                                     # Existing groups in props
    
    for i in range(N_parts):
        # idx=np.random.randint(0,len(groups),N_samples)                                        # Draw random samples with possible repetitions
        try:
            idx = np.random.choice(range(len(groups)), N_samples, replace=False)    # Draw random samples without repetition
        except ValueError:
            print('Sample size larger than data!')
        
        df_out.iloc[i,:] = groups[idx]
        
    return df_out

#%%
def observables_from_part(part,props):
    
    ### Get part in props
    setting = part.setting.iloc[0]
    vary = part.vary.iloc[0]
    rep= part.rep.iloc[0]
    groups = part.iloc[0,4:].values

    subset = (props.setting == setting) & (props.vary==vary) & (props.rep==rep)
    df = props[subset]
    df = df.query('group in @groups')
    
    ### Prepare observables
    s_out = pd.Series(index=OBS_COLUMNS,dtype=np.float32)
    
    ### Assign standard everything except bound imager probabilities
    s_out.tau       = np.nanmean(   df.loc[:,'tau_lin'])
    s_out.A         = np.nanmedian( df.loc[:,'A_lin'])
    s_out.occ       = np.nanmean(   df.loc[:,'n_locs'])
    s_out.taud      = np.nanmedian( df.loc[:,'tau_d'])
    s_out.events    = np.nanmedian( df.loc[:,'n_events'])
    
    ### Assign  bound imager probabilities and eps & snr
    try:
        pks = assign_imagerprob(df)[-1]
        s_out[PK_COLUMNS] = pks
        s_out[['eps','snr']] = np.nanmedian( df.loc[:,['eps','snr']], axis = 0)
    except:
        s_out[PK_COLUMNS] = np.nan
        s_out[['eps','snr']] = np.nan
        
    ### Assign measurement duration and ignore value (only needed for taud and events!!)
    s_out.ignore    = df.loc[:,'ignore'].unique()
    s_out.M         = df.loc[:,'M'].unique()
    
    return s_out
    
#%%
def prep_data(df):
    '''
    Pepare necesary observables as numpy.array ``data`` for setting and solving equation set.

    '''
    data = df.loc[:,SOL_OBS_COLUMNS].values
         
    return data

#%%
### Analytic expressions for observables

def tau_func(koff,konc,tau_meas): return 1 / (koff+konc) - tau_meas

def A_func(koff,konc,N,A_meas): return koff / (konc*N) - A_meas

def occ_func(koff,konc,N,occ_meas):
    p   = ( 1/koff + 1 ) / ( 1/koff + 1/konc ) # Probability of bound imager
    occ = 1 - np.abs(1-p)**N
    occ = occ - occ_meas
    return occ

def taud_func(konc,N,taud_meas): return 1/(N*konc) - taud_meas

def events_func(frames,ignore,koff,konc,N,events_meas):
    p       = ( 1/koff + 1 ) / ( 1/koff + 1/konc )   # Probability of bound imager
    darktot = np.abs(1-p)**N * frames                # Expected sum of dark times
    taud    = taud_func(konc,N,0)                    # Mean dark time
    events  = darktot / (taud + ignore +.5)           # Expected number of events
    return events - events_meas

def binom_array(N,k):
    try:
        c = np.zeros(len(N))
        for i,n in enumerate(N):
            if np.ceil(n) >=k:
                c[i] = binom(n,k)
            else:
                c[i] = 1e-10
    except:
        if np.ceil(N) >=k:
            c = binom(N,k)
        else:
            c = 1e-10
    return c

def pk_func(k_out,koff,konc,N,pks_meas):
    
    ins = [koff,konc,N]
    try: n = max([len(item) for item in ins if isinstance(item,np.ndarray)])
    except: n = 1
    
    p = (1/koff+1)/(1/koff+1/konc) # Probability of bound imager
    pks = np.zeros((n,10))
    for i,k in enumerate(range(1,11)):
        pks[:,i] = binom_array(N,k) * (p)**k *(1-p)**(N-k)
    
    # Normalize
    norm = np.sum(pks,axis=1).reshape(n,1)
    norm = np.repeat(norm,10,axis=1)
    
    pks /= norm 
    
    return pks[:,k_out] - pks_meas

#%%
def create_eqs(x,data, weights):
    '''
    Set-up equation system for varying koncs with given ``data`` as returned by ``prep_data()``. 
    '''
    x = np.abs(x) # koff,konc and N can only be positive
        
    ### Estimate degrees of freedom = Number of unique koncs plus koff & N
    vals = np.unique(data[:,0])
    n = len(vals)          # Unique koncs
    
    ### Define weights
    w = np.array(weights) * 100 # Experimental data
    
    system = np.array([0])
    for i in range(n):
        vals_idx = np.where(data[:,0]==vals[i])[0]  # Get indices of datapoints belonging to one konc
        
        for j in vals_idx:          
            eq0 = ( w[0] / data[j,1] ) * tau_func(x[n],x[i],data[j,1])
            eq1 = ( w[1] / data[j,2] ) * A_func(x[n],x[i],x[n+1],data[j,2])
            eq2 = ( w[2] / data[j,3] ) * occ_func(x[n],x[i],x[n+1],data[j,3])
            eq3 = ( w[3] / data[j,4] ) * taud_func(x[i],x[n+1],data[j,4])
            eq4 = ( w[4] / data[j,5] ) * events_func(data[j,-1],data[j,-2],x[n],x[i],x[n+1],data[j,5])
            
            system = np.append(system,[eq0,eq1,eq2,eq3,eq4])
            
            for k in range(0,10):
                if np.isfinite(data[j,6+k]):
                    eq = ( w[5] / data[j,6+k])  * pk_func(k,x[n],x[i],x[n+1],data[j,6+k])
                    system = np.append(system,[eq])
    
    
    return system[1:]

#%%
def solve_eqs(obs_part,weights):
    '''
    Solve equation system as set-up by ``create_eq()`` for c-series.. 
    '''
    ### Exctract observables
    data = prep_data(obs_part)
    
    ### Get estimate for koffs, konc & N
    vals = np.unique(data[:,0])
    n = len(vals) # Unique concentration points (koncs)
    x0 = []
    for v in vals: x0.extend([1e6*v*1e-12*0.2]) # Estimate for koncs
    tau_max = np.max(data[data[:,0]==np.min(vals),1])
    x0.extend([1/tau_max]) # Estimate for koff
    x0.extend([4])    # Estimate for N   
    x0 = np.array(x0)
    
    ### Solve system of equations
    xopt = optimize.least_squares(lambda x: create_eqs(x,data,weights),
                                                     x0,
                                                     method ='trf',
                                                     ftol=1e-4,
                                                     )
    x = xopt.x
    success = xopt.success
    
    ### Prepare output
    out_idx = ['konc%i'%v for v in vals] + ['koff','N'] + ['success']
    out_vals = [np.abs(x[i]) for i in range(n)] + [np.abs(x[n]),np.abs(x[n+1]),success]
    s_opt = pd.Series(out_vals,index=out_idx)
    
    return s_opt

#%%
def eps_func(B,koff,tau):
    b0 = 0.9973
    b1 = -0.315
    eps = B / (koff * (b0*tau + b1) )
    return eps

#%%
def assign_koff_eps_snr(props_in,sol,intensity='raw'):
    
    ### Decide which brightness value to use
    if intensity == 'raw':
        B_field = 'B_ck'
    elif intensity == 'chungkennedy':
        B_field = 'B_ck'
    elif intensity == 'netgradient':
        B_field = 'B_ng'
    else:
        B_field = 'B_ck'
    
    ### Get koff from solution @(setting)
    setting = props_in.setting.iloc[0]
    koff = float(sol.loc[sol.setting==setting,'koff'] )   
    
    ### Get mean tau @(setting,vary,rep)
    tau = np.nanmean(props_in.tau_lin) 
    # tau = props_in.tau_lin.values                  # Testing individual taus!?! 
    
    ### Assign koff to props
    props = props_in.assign(koff = koff)
    
    ### Assign eps to props using individual groups brightness but global koff & tau
    eps  = eps_func(props[B_field].values,koff,tau)
    props = props.assign(eps = eps)
    
    ### Assign snr to props based on eps, bg, sx and sy
    snr = eps # Total photons
    snr /= 2*np.pi*props.sx.values*props.sy.values # Now snr corresponds to amplitude of 2D gaussian fit
    snr /= props.bg.values # (masimum) signal to noise ratio defined as amplitude/bg
    props = props.assign(snr = snr)
    
    return props

#%%
def assign_imagerhist(locs,props,intensity='raw'):
    ### Decide which photon values to use
    if intensity == 'raw':
        use_field = 'photons'
    elif intensity == 'chungkennedy':
        use_field = 'photons_ck'
    elif intensity == 'netgradient':
        use_field = 'net_gradient'
    else:
        use_field = 'photons'
        
    ### Which data?
    setting,vary,rep, groups = locs.iloc[0][['setting','vary','rep','group']]
    s_in = props.query('setting == @setting & vary == @vary & rep == @rep & group == @groups').iloc[0,:]
    
    ### Normalize photons
    data = locs[use_field].values.astype(np.float32) / float(s_in.eps)

    ### Compute histogram
    ydata = histogram1d(data,
                                       bins=IMG_BINS,
                                       range=(IMG_BINS_RANGE[0],IMG_BINS_RANGE[1]),
                                       )
    
    ydata = ydata * (100/np.sum(ydata).astype(np.float32))     # Normalize
    
    s_hist = pd.Series(ydata,index=IMG_BIN_COLUMNS)
    s_out=pd.concat([s_in,s_hist])
     
    return s_out

#%%
### Analytic expressions for gaussian fitting of imager levels

@numba.jit(nopython=True, nogil=True, cache=False)
def gauss_func(x,a,b,c):
    y = np.abs(a) * np.exp( -0.5 * ((x-b)/c)**2 )
    return y

@numba.jit(nopython=True, nogil=True, cache=False)
def gauss_comb(x,p):
    
    y = np.zeros(len(x),dtype=np.float32)
    
    for i,l in enumerate(range(1,11)):
        y += gauss_func(x,
                                    p[i],
                                    l * (1 + p[-1]),
                                    p[-2] * l**0.5,
                                    )

    return y

#%%
def assign_imagerprob(df):
    
    ### Get combined imager histogram for part
    ydata = df[list(IMG_BIN_COLUMNS)].values.astype(np.float32)
    ydata = np.nanmean(ydata,axis=0).astype(np.float32)                                     # Check if mean or median works better?!?
    xdata = IMG_BIN_COLUMNS * IMG_BIN_WIDTH
    xdata = xdata.astype(np.float32)
    
    # ### Prepare fit initials
    a = np.zeros(10) # Amplitudes of individual gaussians
    for i,l in enumerate(range(1,11)): a[i] = np.max(ydata[np.abs(xdata-l) < 0.4])
    width = np.ones(1) * 0.2 # Standard deviation
    shift = xdata[ydata==a[0]] - 1
    p0 = np.concatenate([a, width, shift]).astype(np.float32) # Combine fit initials
    
    ### Fit
    out = optimize.least_squares(lambda p: gauss_comb(xdata,p) - ydata,
                                                    p0,
                                                    method='trf',
                                                    ftol=1e-10,
                                                    )
    p = out.x
    
    ### Only positive amplitudes ands standard deviation allowed
    p[:10] = np.abs(p[:10]) # Only positive amplitudes allowed
    p[-2] = np.abs(p[-2])     # Only positive standard deviation allowed
    
    ### Remove gaussian peaks with amplitude lower than 1e-3
    try: first_idx = np.where(p[:10]<1e-3)[0][0]
    except: first_idx = 9
    first_idx = max(1,first_idx)
    p[first_idx:10] = 0
    
    ### Calculate normalized area under individual gaussians corresponding to bound imager probability pk 
    pks = np.sqrt(2*np.pi) * p[:10] * p[11] * np.arange(1,11)**0.5
    pks *= (1/np.sum(pks))
    
    ### Remove gaussian peaks with bound imager probability lower than 1e-2 in both p and pk
    p[:10][pks<1e-2] = 0
    pks[pks<1e-2] = np.nan
    
    return xdata, ydata, p, pks

#%%
def combine_solutions(df):
    
    df = df[df.success ==True]
    
    cols = df.columns.to_list()
    cols = [c for c in cols if bool(re.search('koff*|konc*|N$',c))]
    
    s_out = pd.Series(index = cols + [c+'_std' for c in cols])
    
    for c in cols:
        s_out[c]        = np.nanmedian(df[c])
        s_out[c+'_std'] = np.nanstd(df[c])

    return s_out

#%%
def combine_observables(obs_part):
    
    s_out = pd.Series(index = OBS_COLUMNS)
    s_out[OBS_COLUMNS]  = np.nanmean(obs_part[OBS_COLUMNS], axis = 0)

    return s_out

#%%
def expected_observables(obs,sol):
    df_out = obs.copy()
    
    setting = int(np.unique(obs.setting.values))
    select = sol.setting == setting
    
    M      = obs.M.values
    ignore = int(np.unique(obs.ignore.values))
    koff   = float(sol.loc[select,'koff'])
    concs  = obs.vary.values
    koncs  = sol.loc[select,['konc%i'%conc for conc in concs]].values.flatten()
    N      = float(sol.loc[select,'N'])
    
    df_out['tau']    = tau_func(koff,koncs,0)
    df_out['A']      = A_func(koff,koncs,N,0)
    df_out['occ']    = occ_func(koff,koncs,N,0)
    df_out['taud']   = taud_func(koncs,N,0)
    df_out['events'] = events_func(M,ignore,koff,koncs,N,0)
    
    for k in range(0,10):
        df_out['p'+('%i'%(k+1)).zfill(2)] = pk_func(k,koff,koncs,N,0)
    
    ### Assign NaNs to columns that are not to be expected
    cols = df_out.columns.to_list()
    cols = [c for c in cols if bool(re.search('_std|eps|snr',c))]
    df_out[cols] = np.nan
    
    return df_out

#%%
def assign_N(df,sol):
    
    props = df.copy()
    conc  = int(np.unique(props.vary))
    konc_field = 'konc%i'%conc        # Get correct konc in solution 
    
    koff = float(sol.koff)
    konc = float(sol[konc_field])
    
    A = props.A_lin.values
    N = (1/A) * (koff/konc)
    
    props = props.assign(N = N)
    
    return props