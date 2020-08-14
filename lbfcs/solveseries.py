import numpy as np
import pandas as pd
from scipy.optimize import least_squares


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
def prep_Tseries(df, series_field = 'koff'):
    '''
    Pepare necesary values for solving equation set for T-series as numpy.array ``data``.

    '''
    series  = df.loc[:,(series_field)].values
    tau     = df.loc[:,('tau_lin_one','mean')].values
    A       = df.loc[:,('A_lin_one','50%')].values
    occ     = df.loc[:,('n_locs','mean')].values
    taud    = df.loc[:,('tau_d','50%')].values
    events  = df.loc[:,('n_events','50%')].values
    frames  = df.loc[:,('M','mean')].values
    
    data = np.zeros((len(series),7))
    data[:,0] = series
    data[:,1] = tau
    data[:,2] = A
    data[:,3] = occ
    data[:,4] = taud
    data[:,5] = events
    
    data[:,6] = frames
         
    return data

#%%
def create_eqsTseries(x,data):
    '''
    Set-up equation system for T-series with given ``data`` as returned by ``prep_Teries()``. 
    '''
    ### Analytic expressions for observables
    def tau_func(koff,konc,tau_meas): return 1 / (koff+konc) - tau_meas
    
    def A_func(koff,konc,N,A_meas): return koff / (konc*N) - A_meas
    
    def occ_func(koff,konc,N,occ_meas):
        p   = (1/koff+1) / (1/koff+1/konc)
        occ = 1 - (1-p)**N
        occ = occ - occ_meas
        return occ
    
    def taud_func(konc,N,taud_meas): return 1 / (N*konc) - taud_meas
    
    def events_func(frames,koff,konc,N,events_meas):
        p       = (1/koff+1)/(1/koff+1/(konc))   # Probability of bound imager
        darktot = (1-p)**N * frames              # Expected sum of dark times
        taud    = 1/(N*konc)                     # Mean dark time
        events  = darktot / (taud+1.5)           # Expected number of events
        return events - events_meas

    ### Estimate degrees of freedom = Number of unique koffs plus konc & N
    vals = np.unique(data[:,0])
    n = len(vals) # Unique temperature points (koffs)
    
    system = []
    for i in range(n):
        vals_idx = np.where(data[:,0]==vals[i])  # Get indices of datapoints belonging to one koff
        
        for j in vals_idx:
            system.extend(tau_func(x[i],x[n],data[j,1]))
            system.extend(A_func(x[i],x[n],x[n+1],data[j,2]))
            system.extend(100 * occ_func(x[i],x[n],x[n+1],data[j,3]))
            system.extend(1/20 * taud_func(x[n],x[n+1],data[j,4]))
            system.extend(1/10 * events_func(data[j,-1],x[i],x[n],x[n+1],data[j,5]))
            
    return np.array(system)

#%%
def solve_eqsTseries(data):
    '''
    Solve equation system as set-up by ``create_eqsTseries_occ()`` for T-series with given ``data`` as returned by ``prep_Teries()``. 
    '''
    ### Get estimate for koffs, konc & N
    vals = np.unique(data[:,0])
    n = len(vals) # Unique temperature points (koffs)
    x0 = []
    for i in range(n):
        vals_idx = np.where(data[:,0]==vals[i])  # Get indices of datapoints belonging to one koff
        x0.extend([1/np.mean(data[vals_idx,1])]) # Estimate for koff 
    x0.extend([2e-3]) # Estimate for konc
    x0.extend([4])    # Estimate for N

        
    x0 = np.array(x0)
    
    ### Solve system of equations
    xopt = least_squares(lambda x: create_eqsTseries(x,data),
                         x0,
                         jac='3-point',
                         method='trf',
                         tr_solver='exact',
                         )
    ### Prepare output
    out_idx = ['koff%i'%i for i in range(n)] + ['konc','N','cost','success']
    out_vals = [xopt.x[i] for i in range(n)] + [xopt.x[n],xopt.x[n+1],xopt.cost, xopt.success]
    s_opt = pd.Series(out_vals,index=out_idx)
    
    return s_opt


#%%
def solve_Tseries(df,series_field = 'koff'):
    '''
    Combination of ``prep_Tseries()`` and ``solve_eqsTseries()`` to set-up and find solution to T-series using autocorrelation time and amplitude only.
    '''
    data = prep_Tseries(df,series_field = series_field)
    s_opt = solve_eqsTseries(data)
    
    return s_opt