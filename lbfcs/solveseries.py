import numpy as np
import pandas as pd
import re
import numba as numba
import time
from scipy.special import binom
from lmfit import minimize
import scipy.optimize as optimize

#%%
def prefilter(df_in):
    
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
def bootstrap_props(df,parts=10,samples=500): 
    '''
    Bootstraping of props, i.e. result is DataFrame splitted into ``parts`` containing random samples ``samples``

    '''
    df_part_list=[]
    for i in range(parts):
        idx=np.random.randint(0,len(df),samples)                              # Draw random samples
        df_part=df.iloc[idx,:].copy()
        
        df_part=df_part.assign(part=i)                                        # Assign partition ID
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

###
def tau_func(koff,konc,tau_meas): return 1 / (koff+konc) - tau_meas

###
def A_func(koff,konc,N,A_meas): return koff / (konc*N) - A_meas

###
def occ_func(koff,konc,N,occ_meas):
    p   = ( 1/koff + 1 ) / ( 1/koff + 1/konc )
    occ = 1 - np.abs(1-p)**N
    occ = occ - occ_meas
    return occ

###
def taud_func(konc,N,taud_meas): return 1/(N*konc*1) - taud_meas

###
def events_func(frames,ignore,koff,konc,N,events_meas):
    p       = ( 1/koff + 1 ) / ( 1/koff + 1/konc )   # Probability of bound imager
    darktot = np.abs(1-p)**N * frames                # Expected sum of dark times
    taud    = taud_func(konc,N,0)                    # Mean dark time
    events  = darktot / (taud + ignore +.5)           # Expected number of events
    return events - events_meas

###
def binom_array(N,k):
    try:
        c = np.zeros(len(N))
        for i,n in enumerate(N):
            if np.ceil(n) >=k:
                c[i] = binom(n,k)
            else:
                c[i] = 0
                # c[i] = binom(n,k)
    except:
        if np.ceil(N) >=k:
            c = binom(N,k)
        else:
            c = 1e-10
            # c = binom(N,k)
    return c

###
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
def create_eqs_koff(x,data,model):
    '''
    Set-up equation system for varying koffs with given ``data`` as returned by ``prep_data()``. 
    '''
    x = np.abs(x) # koff,konc and N can only be positive
    
    ### Estimate degrees of freedom = Number of unique koffs plus konc & N
    vals = np.unique(data[:,0])
    n = len(vals)          # Unique koffs
    m = np.shape(data)[1]
    
    ### Define weights
    w = np.array([2, 1, 2, 1, 1, 1]) # Experimental data
    
    system = np.array([0])
    for i in range(n):
        vals_idx = np.where(data[:,0]==vals[i])[0]  # Get indices of datapoints belonging to one koff
        
        for j in vals_idx:
            eq0 = ( w[0] / (0.02*data[j,1]) ) * tau_func(x[i],x[n],data[j,1])
            eq1 = ( w[1] / (0.02*data[j,2]) ) * A_func(x[i],x[n],x[n+1],data[j,2])
            eq2 = ( w[2] / (0.02*data[j,3]) ) * occ_func(x[i],x[n],x[n+1],data[j,3])
            eq3 = ( w[3] / (0.02*data[j,4]) ) * taud_func(x[n],x[n+1],data[j,4])
            eq4 = ( w[4] / (0.02*data[j,5]) ) * events_func(data[j,-1],data[j,-2],x[i],x[n],x[n+1],data[j,5])
            
            if model=='full': system = np.append(system,[eq0,eq1,eq2,eq3,eq4])
            else: system = np.append(system,[eq0,eq1,eq2])
            
            if m > 8: 
                for k in range(0,10):
                    if data[j,6+k] >= 5e-2:
                        eq = ( w[5] / 2e-2 ) * pk_func(k,x[i],x[n],x[n+1],data[j,6+k])
                    else:
                        eq = 0
                    system = np.append(system,[eq])
                    
    return system[1:]

#%%
def create_eqs_konc(x,data,model):
    '''
    Set-up equation system for varying koncs with given ``data`` as returned by ``prep_data()``. 
    '''
    x = np.abs(x) # koff,konc and N can only be positive
    
    ### Estimate degrees of freedom = Number of unique koncs plus koff & N
    vals = np.unique(data[:,0])
    n = len(vals)          # Unique koncs
    m = np.shape(data)[1]
    
    ### Define weights
    w = np.array([1, 2, 2, 1, 1, 1]) # Experimental data
    
    system = np.array([0])
    for i in range(n):
        vals_idx = np.where(data[:,0]==vals[i])[0]  # Get indices of datapoints belonging to one konc
        
        for j in vals_idx:          
            eq0 = ( w[0] / data[j,1] ) * tau_func(x[n],x[i],data[j,1])
            eq1 = ( w[1] / data[j,2] ) * A_func(x[n],x[i],x[n+1],data[j,2])
            eq2 = ( w[2] / data[j,3] ) * occ_func(x[n],x[i],x[n+1],data[j,3])
            eq3 = ( w[3] / data[j,4] ) * taud_func(x[i],x[n+1],data[j,4])
            eq4 = ( w[4] / data[j,5] ) * events_func(data[j,-1],data[j,-2],x[n],x[i],x[n+1],data[j,5])
            
            if model=='full': system = np.append(system,[eq0,eq1,eq2,eq3,eq4])
            else: system = np.append(system,[eq0,eq1,eq2])
            
            if m > 8: 
                for k in range(0,10):
                    if data[j,6+k] >= 0.1 * np.max(data[j,6:-2]):
                        # eq = ( (data[j,6+k]/np.max(data[j,6:-2]))**0.33 / 2e-2 ) * pk_func(k,x[n],x[i],x[n+1],data[j,6+k])
                        eq = ( w[5] / data[j,6] )  * pk_func(k,x[n],x[i],x[n+1],data[j,6+k]) 
                        
                        system = np.append(system,[eq])
                        
    return system[1:]

#%%
def solve_eqs_koff(data,model):
    '''
    Solve equation system as set-up by ``create_eqs_koff()`` for T-series with given ``data`` as returned by ``prep_data()``. 
    '''
    ### Get estimate for koffs, konc & N
    vals = np.unique(data[:,0])
    n = len(vals) # Unique temperature points (koffs)
    x0 = []
    for i in range(n):
        vals_idx = np.where(data[:,0]==vals[i])  # Get indices of datapoints belonging to one koff
        x0.extend([1/np.mean(data[vals_idx,1])]) # Estimate for koff 
    x0.extend([1e-3]) # Estimate for konc
    x0.extend([4])    # Estimate for N

        
    x0 = np.array(x0)
    
    ### Solve system of equations
    xopt = optimize.least_squares(lambda x: create_eqs_koff(x,data,model),
                                  x0,
                                   jac='3-point',
                                  method ='trf',
                                  # loss = 'soft_l1',
                                   tr_solver='exact',
                                  )
    x = xopt.x
    success = xopt.success
    
    # xopt,xcov, info, msg, ier = optimize.leastsq(create_eqs_koff,
    #                                              x0,
    #                                              args=(data),
    #                                              full_output=1,
    #                                              )
    # x = xopt
    # success = ier>0
    
    ### Prepare output
    out_idx = ['koff%i'%i for i in range(n)] + ['konc','N'] + ['success']
    out_vals = [np.abs(x[i]) for i in range(n)] + [np.abs(x[n]),np.abs(x[n+1]),success]
    s_opt = pd.Series(out_vals,index=out_idx)
    
    return s_opt

#%%
def solve_eqs_konc(data,model):
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
    xopt = optimize.least_squares(lambda x: create_eqs_konc(x,data,model),
                                  x0,
                                   jac='3-point',
                                  method ='trf',
                                  tr_solver='exact',
                                  # loss = 'huber',
                                  # max_nfev=,10
                                  )
    x = xopt.x
    success = xopt.success
    
    # xopt,xcov, info, msg, ier = optimize.leastsq(create_eqs_konc,
    #                                               x0,
    #                                               args=(data,model),
    #                                               full_output=1,
    #                                               )
    # x = xopt
    # success = ier>0
    
    ### Prepare output
    out_idx = ['konc%i'%v for v in vals] + ['koff','N'] + ['success']
    out_vals = [np.abs(x[i]) for i in range(n)] + [np.abs(x[n]),np.abs(x[n+1]),success]
    s_opt = pd.Series(out_vals,index=out_idx)
    
    return s_opt

#%%'
def solve_eqs(df,series='konc',model='else'):
    '''
    Combination of ``prep_data()`` and ``solve_eqs_*()`` to set-up and find solution.
    '''
    data = prep_data(df)
    if series =='koff':
        s_opt = solve_eqs_koff(data,model)
    if series =='konc':
        s_opt = solve_eqs_konc(data,model)
    return s_opt

#%%
def combine_observables(df):
    
    cols = df.columns.to_list()
    cols = [c for c in cols if not bool(re.search('setting|vary|rep|part',c))]
    
    s_out = pd.Series(index = cols + [c+'_std' for c in cols])
    
    for c in cols:
        s_out[c]        = np.nanmean(df[c])
        s_out[c+'_std'] = np.nanstd(df[c])
    
    return s_out

#%%
def combine_solutions(df):
    
    cols = df.columns.to_list()
    cols = [c for c in cols if bool(re.search('koff*|konc*|N$',c))]
    
    s_out = pd.Series(index = cols + [c+'_std' for c in cols])
    
    for c in cols:
        s_out[c]        = np.nanmedian(df[c])
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
    
    return df_out

#%%
def eps_func(B,koff,tau):
    b0 = 0.9973
    b1 = -0.315
    eps = B / (koff * (b0*tau + b1) )
    return eps

#%%
def get_levels(props,locs,sol,filtered=True):
    
    ### Decide which photon and derived values to use
    photons_field = 'photons'
    B_field = 'B'
    if filtered:
        photons_field += '_ck'
        B_field += '_ck'
    
    ### Which data?
    setting = int (np.unique(props.setting))
    vary    = int (np.unique(props.vary))
    rep     = int (np.unique(props.rep))
    
    ### Get tau @(setting,vary,rep) in props and koff0 @(setting,:,:) in solution
    koff = float(sol.loc[sol.setting==setting,'koff'] )                      
    tau = np.nanmean(props.tau_lin)

    ### Group properties @(setting,vary,rep) in props
    groups  = props.group.values 
    eps     = eps_func(props[B_field].values,koff,tau)                     
    nlocs   = np.round( ( props.n_locs * props.M ).values ).astype(int)
    
    ### Query locs for groups in props @(setting,vary,rep)
    select_subset = (locs.setting == setting) & (locs.vary == vary) & (locs.rep == rep)
    locs_sub = locs.loc[select_subset,['group',photons_field]]                   
    locs_sub = locs_sub.query('group in @groups')


    ### Transform photons to imager levels
    norm     = [eps[g] for g in range(len(groups)) for i in range(nlocs[g])]
    levels   = locs_sub[photons_field].values/norm

    
    ### Create normalized histogram of levels of fixed bin width
    bin_width = 0.1 
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
        y += gauss_func(x,p[i], l*(1+p[-1]), p[-2] * l**0.5)

    return y

#%%
def fit_levels(levels):
    
    ### Which columns to use
    cols = levels.columns.to_list()
    search_str = 'setting|vary|rep'
    cols = [c for c in cols if not bool(re.search(search_str,str(c)))]
    
    ### Prepare data
    xdata = np.array(list(levels[cols].columns))
    ydata = levels[cols].values.flatten()
    
    ### Prepare fit initials
    bin_width = xdata[1]-xdata[0]
    a = np.zeros(10)
    for i,l in enumerate(range(1,11)): a[i] = ydata[np.abs(xdata-l) < bin_width/2]
    c = np.ones(1) * 0.25
    
    p0 = np.concatenate([a,c,[1e-2]])
    
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
    setting = float (levels.setting)
    vary    = float (levels.vary)
    rep     = float (levels.rep)
    
    ### Fit levels to get pk @(setting,vary)
    pk = fit_levels(levels)[-1]
   
    ### Assign pk to obs for @(setting,vary,:) i.e. to all partitions!
    select_subset = (obs.setting==setting) & (obs.vary==vary) & (obs.rep==rep)
    obs_sub = obs.loc[select_subset,:]
    
    keys    = ['p'+('%i'%k).zfill(2) for k in range(1,11)]
    dict_pk = dict(zip(keys,pk))
    df_pk   = pd.DataFrame(dict_pk, index = obs_sub.index)
    obs_sub = pd.concat([obs_sub.iloc[:,:-2],df_pk,obs_sub.iloc[:,-2:]],axis=1)
    
    return obs_sub

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