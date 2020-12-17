import numpy as np
import pandas as pd
import re
import numba as numba
import time
from scipy.special import binom
from lmfit import minimize
import scipy.optimize as optimize

import lbfcs.visualizeseries as visualize

#%%
def prefilter(props_in,locs_in, field, select):
    '''
    Filter props and query locs for remaining groups.

    '''
    ### Select subset of props
    props = props_in.query( field + ' in @select' )
    
    ### Filter props_in
    props = props.groupby(['setting','vary','rep']).apply(prefilter_props)
    props = props.droplevel(level=['setting','vary','rep']) # Remove index after groupby
    
    ### Query locs for groups in props @(setting, vary, rep)
    locs = locs_in.groupby(['setting','vary','rep']).apply(lambda df: prefilter_query_locs(df,props))
    locs = locs.droplevel(level=['setting','vary','rep']) # Remove index after groupby
    
    return props, locs

#%%
def bootstrap_firstsolve(props,exp,parts=20,samples=500,weights=[1,1,1,0,0,1]):
    
    ### Bootstrap props @(setting,vary,rep) -> part
    props_part = props.groupby(['setting','vary','rep']).apply(lambda df: bootstrap_props(df, parts ,samples))
    props_part = props_part.droplevel(['setting','vary','rep'])

    ### (Statistical) observables @(setting,vary,rep,part)
    obs_part = props_part.groupby(['setting','vary','rep','part']).apply(extract_observables)
    obs_part = obs_part.reset_index(level=['setting','vary','rep','part'])
    
    ### 1st iteration solution @(setting,part) using tau,A,occ
    sol_part = obs_part.groupby(['setting','part']).apply(lambda df: solve_eqs(df,weights))
    sol_part = sol_part.reset_index(level=['setting','part'])
    
    ### Combine 1st iteration solutions @(setting)
    sol = sol_part.groupby(['setting']).apply(combine_solutions)
    sol = sol.reset_index(level=['setting'])
    
    ### Print solution
    print()
    print('1st iteration solution (exp = %i ms):' %(int(1000*exp)))
    visualize.print_sol(sol,exp)

    return props_part, obs_part, sol_part, sol 

#%%
def levels_secondsolve(props_part_in,
                                         sol,
                                         locs,
                                         obs_part_in,
                                         exp,
                                         field='raw',
                                         weights=[1,1,1,0,0,1],
                                         ):
    print()
    ### Assign eps to props using koff@(setting) & tau@(setting,vary,rep)
    props_part = props_part_in.groupby(['setting','vary','rep']).apply(lambda df: assign_eps(df,sol,field))
    props_part = props_part.droplevel(['setting','vary','rep'])
    
    ### Get bound imager levels @(setting,vary,rep,part)
    print('Creating normalized photon histograms ...')
    levels_part = props_part.groupby(['setting','vary','rep','part']).apply(lambda df: get_levels(df,locs,sol,field))
    levels_part = levels_part.reset_index(level=['setting','vary','rep','part'])
    
    ### Fit levels and assign result to obs_part_in @(setting,vary,rep,part)
    print('Fitting bound imager probabilities...')
    obs_part = levels_part.groupby(['setting','vary','rep','part']).apply(lambda df: assign_levels(df,obs_part_in))
    obs_part = obs_part.droplevel(['setting','vary','rep','part'])
    
    ### 2nd iteration solution @(setting,part) using tau,A,occ and bound imager probabilities pk
    print('Finding new solution ...')
    sol_part_levels = obs_part.groupby(['setting','part']).apply(lambda df: solve_eqs(df,weights))
    sol_part_levels = sol_part_levels.reset_index(level=['setting','part'])
    
    ### Combine 2nd iteration solutions @(setting)
    sol_levels = sol_part_levels.groupby(['setting']).apply(combine_solutions)
    sol_levels = sol_levels.reset_index(level=['setting'])
    
    ### Print solution
    print()
    print('2nd iteration solution (exp = %i ms):' %(int(1000*exp)))
    visualize.print_sol(sol_levels,exp)
    
    return props_part, levels_part, obs_part, sol_part_levels, sol_levels

#%%
def combine(props_part, obs_part, levels_part,sol_list):
    
    ### Combine observables for different parts @(setting,vary,rep)
    obs = obs_part.groupby(['setting','vary','rep']).apply(lambda df: combine_observables(df,props_part))
    obs = obs.reset_index(level=['setting','vary','rep'])
    
    ### Combine levels for different parts @(setting,vary,rep)
    levels = levels_part.groupby(['setting','vary','rep']).apply(combine_levels)
    levels = levels.reset_index(level=['setting','vary','rep'])
    
    obs_expect_list = []
    for sol in sol_list:
         obs_expect = obs.groupby('setting').apply(lambda df: expected_observables(df,sol))
         obs_expect = obs_expect.droplevel(['setting'])
         obs_expect_list.extend([obs_expect])
         
    return obs, levels , obs_expect_list 
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
    
    df = it_medrange(df,'A_lin'  ,[100,2])
    df = it_medrange(df,'A_lin'    ,[2,2])
                     
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
    print('Prefilter @(%i,%i,%i): %i '%(setting,vary,rep,len(groups)))
    return locs
    
#%%
def bootstrap_props(df,parts,samples): 
    '''
    Bootstraping of props, i.e. result is DataFrame splitted into ``parts`` containing random samples ``samples``

    '''
    df_part_list=[]
    for i in range(parts):
        # idx=np.random.randint(0,len(df),samples)                                       # Draw random samples
        try:
            idx = np.random.choice(range(len(df)), samples, replace=False)    # Draw random samples without repetition
        except ValueError:
            print('Sample size larger than data!')
        df_part=df.iloc[idx,:].copy()
        
        df_part=df_part.assign(part=i)                                                # Assign partition ID
        df_part_list.extend([df_part])
    
    df_out=pd.concat(df_part_list)
    return df_out

#%%
def extract_observables(df):
    s_out = pd.Series(index=['tau','A','occ','taud','events','ignore','M'])
    
    s_out.tau       = np.nanmean(   df.loc[:,'tau_lin'])
    s_out.A         = np.nanmedian( df.loc[:,'A_lin'])
    s_out.occ       = np.nanmean(   df.loc[:,'n_locs'])
    s_out.taud      = np.nanmedian( df.loc[:,'tau_d'])
    s_out.events    = np.nanmedian( df.loc[:,'n_events'])
    
    s_out.ignore    = df.loc[:,'ignore'].unique()
    s_out.M         = df.loc[:,'M'].unique()
    
    return s_out
    
#%%
def prep_data(df):
    '''
    Pepare necesary observables as numpy.array ``data`` for setting and solving equation set.

    '''
    cols = df.columns.to_list()
    search_str = 'setting|rep|part'
    cols = [c for c in cols if not bool(re.search(search_str,c))]
    data = df.loc[:,cols].values
         
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
def create_eqs_konc(x,data, weights):
    '''
    Set-up equation system for varying koncs with given ``data`` as returned by ``prep_data()``. 
    '''
    x = np.abs(x) # koff,konc and N can only be positive
    
    ### Estimate degrees of freedom = Number of unique koncs plus koff & N
    vals = np.unique(data[:,0])
    n = len(vals)          # Unique koncs
    m = np.shape(data)[1]
    
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

            if m > 8: 
                for k in range(0,10):
                    if data[j,6+k] >= 1e-1* np.max(data[j,6:-2]):
                        eq = ( w[5] / data[j,6+k])  * pk_func(k,x[n],x[i],x[n+1],data[j,6+k])
                        system = np.append(system,[eq])
    
    
    return system[1:]

#%%
def solve_eqs_konc(data,weights):
    '''
    Solve equation system as set-up by ``create_eqs_konc()`` for c-series with given ``data`` as returned by ``prep_data()``. 
    '''
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
    xopt = optimize.least_squares(lambda x: create_eqs_konc(x,data,weights),
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

#%%'
def solve_eqs(df,weights):
    '''
    Combination of ``prep_data()`` and ``solve_eqs_konc()`` to set-up and find solution.
    '''
    data = prep_data(df)
    s_opt = solve_eqs_konc(data,weights)
    
    return s_opt

#%%
def eps_func(B,koff,tau):
    b0 = 0.9973
    b1 = -0.315
    eps = B / (koff * (b0*tau + b1) )
    return eps

#%%
def assign_eps(props_in,sol,field):
    
    ### Decide which brightness value to use
    if field == 'raw':
        B_field = 'B_ck'
    elif field == 'chungkennedy':
        B_field = 'B_ck'
    elif field == 'netgradient':
        B_field = 'B_ng'
    else:
        B_field = 'B_ck'
    
    ### Get koff from solution @(setting)
    setting = int (np.unique(props_in.setting))
    koff = float(sol.loc[sol.setting==setting,'koff'] )   
    
    ### Get mean tau @(setting,vary,rep)
    tau = np.nanmean(props_in.tau_lin)
    
    ### Assign eps to props using individual groups brightness but global koff & tau
    eps  = eps_func(props_in[B_field].values,koff,tau)
    props = props_in.assign(eps = eps)
    
    ### Assign snr to props bases on eps, bg, sx and sy
    snr = eps # Total photons
    snr /= 2*np.pi*props.sx.values*props.sy.values # Now snr corresponds to amplitude of 2D gaussian fit
    snr /= props.bg.values # (masimum) signal to noise ratio defined as amplitude/bg
    props = props.assign(snr = snr)
    
    return props

#%%
def get_levels(props,locs,sol,field):
    
    ### Decide which photon values to use
    if field == 'raw':
        photons_field = 'photons'
    elif field == 'chungkennedy':
        photons_field = 'photons_ck'
    elif field == 'netgradient':
        photons_field = 'net_gradient'
    else:
        photons_field = 'photons'
        
    ### Which data?
    setting = int (np.unique(props.setting))
    vary = int (np.unique(props.vary))
    rep = int (np.unique(props.rep))
    
    ### Group properties @(setting,vary,rep,part) in props
    groups  = props.group.values 
    eps     = props.eps.values            
    nlocs   = np.round( ( props.n_locs * props.M ).values ).astype(int)
    
    ### Query locs for groups in props @(setting,vary,rep)
    select_subset = (locs.setting == setting) & (locs.vary == vary) & (locs.rep == rep)
    locs_sub = locs.loc[select_subset,['group',photons_field]]  # Preselect to shorten query                 
    locs_sub = locs_sub.query('group in @groups')

    ### Transform photons to imager levels
    norm     = [eps[g] for g in range(len(groups)) for i in range(nlocs[g])]
    levels   = locs_sub[photons_field].values/norm

    ### Create normalized histogram of levels of fixed bin width
    bin_width = 0.05
    ydata,xdata = np.histogram(levels,
                                                  bins=np.arange(bin_width/2,11+bin_width,bin_width),
                                                  weights=100*np.ones(len(levels))/len(levels))
    xdata = xdata[:-1] + bin_width/2 # Adjust bins to center position
    
    ### Define output
    s_out = pd.Series(ydata,index=xdata)

    return s_out

#%%
### Analytic expressions for gaussian fitting of imager levels
def gauss_func(x,a,b,c):
    y = np.abs(a) * np.exp( -0.5 * ((x-b)/c)**2 )
    return y

def gauss_comb(x,p):
    
    y = np.zeros(len(x))
    
    for i,l in enumerate(range(1,11)):
        y += gauss_func(x,
                                    p[i],
                                    # l + p[-1] * ( 1 + int(l/2) ),
                                    l * (1 + p[-1]),
                                    p[-2] * l**0.5,
                                    )

    return y

#%%
def fit_levels(levels):
    
    ### Which columns to use
    cols = levels.columns.to_list()
    search_str = 'setting|vary|rep|part'
    cols = [c for c in cols if not bool(re.search(search_str,str(c)))]
    
    ### Prepare data
    xdata = np.array(list(levels[cols].columns))
    ydata = levels[cols].values.flatten()
    
    ### Prepare fit initials
    bin_width = xdata[1]-xdata[0]
    a = np.zeros(10)
    for i,l in enumerate(range(1,11)): a[i] = ydata[np.abs(xdata-l) < bin_width/2]
    c = np.ones(1) * 0.25
    
    p0 = np.concatenate([a,c,[-1e-2]])
    
    ### Fit
    out = optimize.least_squares(lambda p: gauss_comb(xdata,p) - ydata,
                                  p0,
                                  method='trf',
                                  )
    
    ### Clean fit results
    p = out.x
    p[:10] = np.abs(p[:10]) # Only positive amplitudes allowed
    p[-2] = np.abs(p[-2])   # Only positive standard deviation allowed
    
    ### Calculate area under individual gaussians
    area = np.sqrt(2*np.pi) * p[:10] * p[10:-1] * np.arange(1,11)**0.5
    area *= (1/np.sum(area))
    
    return xdata, ydata, p , area

#%%
def assign_levels(levels,obs):
    
    ### Which data?
    setting = int(levels.setting.iloc[0])
    vary  = int(levels.vary.iloc[0])
    rep  = int(levels.rep.iloc[0])
    part = int(levels.part.iloc[0])
    
    ### Fit levels to get pk @(setting,vary)
    pk = fit_levels(levels)[-1]
   
    ### Assign pk to obs for @(setting,vary,rep,part)
    select_subset = (obs.setting==setting) & (obs.vary==vary) & (obs.rep==rep) & (obs.part==part)
    obs_sub = obs.loc[select_subset,:].copy()
    keys    = ['p'+('%i'%k).zfill(2) for k in range(1,11)]
    dict_pk = dict(zip(keys,pk))
    obs_sub = obs_sub.assign(**dict_pk)
    
    ### Rearrange columns such that M & ignore are last
    cols = obs_sub.columns.to_list()
    cols = [c for c in cols if not bool(re.search(c,'M|ignore'))]
    cols = cols + ['ignore','M']
    obs_sub = obs_sub[cols]
    
    return obs_sub

#%%
def combine_observables(obs_part,props_part):
    
    setting = int(obs_part.setting.iloc[0])
    vary  = int(obs_part.vary.iloc[0])
    rep  = int(obs_part.rep.iloc[0])
    
    subset = (props_part.setting==setting) & (props_part.vary==vary) & (props_part.rep==rep)
    
    cols = obs_part.columns.to_list()
    cols = [c for c in cols if not bool(re.search('setting|vary|rep|part',c))]
    
    cols_add = ['eps','snr'] 
    s_out = pd.Series(index = cols + cols_add + [c+'_std' for c in cols] + [c+'_std' for c in cols_add])
    
    for c in cols:
        s_out[c]        = np.nanmean(obs_part[c])
        s_out[c+'_std'] = np.nanstd(obs_part[c])
    
    for c in cols_add:
        s_out[c] = np.nanmedian(props_part.loc[subset,c])
        s_out[c+'_std'] = np.nanstd(props_part.loc[subset,c]) 
    
    return s_out

#%%
def combine_levels(levels_part):
    
    cols = levels_part.columns.to_list()
    cols = [c for c in cols if not bool(re.search('setting|vary|rep|part',str(c)))]
    
    s_out = pd.Series(index = cols)
    
    for c in cols:
        s_out[c]        = np.nanmean(levels_part[c])

    return s_out

#%%
def combine_solutions(df):
    
    df = df[df.success ==True]
    
    cols = df.columns.to_list()
    cols = [c for c in cols if bool(re.search('koff*|konc*|N$',c))]
    
    s_out = pd.Series(index = cols + [c+'_std' for c in cols])
    
    for c in cols:
        s_out[c]        = np.nanmean(df[c])
        s_out[c+'_std'] = np.nanstd(df[c])

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