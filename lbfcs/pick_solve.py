import numpy as np
import pandas as pd
import numba
import scipy.optimize as optimize
import warnings
from tqdm import tqdm
tqdm.pandas()

warnings.filterwarnings("ignore")

#%%
### Analytic expressions for observables

@numba.jit(nopython=True, nogil=True, cache=False)
def tau_func(koff,konc,tau_meas): return 1 / (koff+konc) - tau_meas

@numba.jit(nopython=True, nogil=True, cache=False)
def A_func(koff,konc,N,A_meas): return koff / (konc*N) - A_meas

@numba.jit(nopython=True, nogil=True, cache=False)
def p_func(koff,konc,corr=0): 
    p = ( 1/koff + corr ) / ( 1/koff + 1/konc )
    return p

@numba.jit(nopython=True, nogil=True, cache=False)
def occ_func(koff,konc,N,occ_meas):
    p = p_func(koff,konc,0.5)
    occ = 1 - np.abs(1-p)**N
            
    occ = occ - occ_meas
    return occ

@numba.jit(nopython=True, nogil=True, cache=False)
def I_func(koff,konc,N,eps,I_meas):
    p = p_func(koff,konc)
    I = eps * N * p
    I = I - I_meas
    return I

@numba.jit(nopython=True, nogil=True, cache=False)
def Inormed_func(koff,konc,N,I_meas):
    p = p_func(koff,konc)
    I = N * p
    I = I - I_meas
    return I

#%%
@numba.jit(nopython=True, nogil=True, cache=False)
def create_eqs(x,data,weights,normed):
    '''
    Set-up equation system.
    '''
    ### Unkowns in system: x = [konc,koff,N,eps]
    ###  -> x can only be positive
    x = np.abs(x) 
        
    ### Define weights for fitting, i.e. residual will be divided by (data_meas/(weight*100)).
    ### This means for weight = 1 a standard deviation of 1% is assumed in data_meas.
    w = weights / (data/100)
    
    ### Initialize equation system consisting of 4 equations (tau,A,occ,I)
    system = np.zeros(4)
    
    ### Add residuals to equation system
    system[0] = w[0] * tau_func(x[0],x[1],data[0])
    system[1] = w[1] * A_func(x[0],x[1],x[2],data[1])
    system[2] = w[2] * occ_func(x[0],x[1],x[2],data[2])
            
    if normed: # In this case no eps will be fitted
        system[3] = w[3] * Inormed_func(x[0],x[1],x[2],data[3])
        
    else:  # Fit eps value
        system[3] = w[3] * I_func(x[0],x[1],x[2],x[3],data[3])
        
    ### Remove NaNs
    system[np.isnan(system)] = 0
    
    return system

#%%
@numba.jit(nopython=True, nogil=True, cache=False)
def estimate_unknowns(data,normed):
    '''
    Get estimate x0 for unknowns x based on data to feed into fit as initial value x0.
    '''
    ### Init estimate
    x0 = np.zeros(4,dtype=np.float32)
    
    ### Make estimate for (koff,konc,N) based on analytical solution using (tau,A,I)
    tau = data[0]
    A = data[1]
    I = data[3]
    
    koff = A * (I/tau)
    konc = 1/tau - koff
    N = A * (koff/konc)

    x0[0] = koff 
    x0[1] = konc
    x0[2] = N
    x0[3] = 1 # Estimate for eps after normalization!
    
    ### Remove eps estimate from x0 if normalized I is used
    if normed: x0 = x0[:3]

    return x0
    
#%%
def solve_eqs(data,weights,normed):
    '''
    Solve equation system as set-up by ``create_eqs()``.
    '''
    weights = np.array(weights,dtype=np.float32)
    
    ### Get estimate x0
    x0 = estimate_unknowns(data,normed)
    
    ### Solve system of equations
    try:
        xopt = optimize.least_squares(lambda x: create_eqs(x,data,weights,normed),
                                                         x0,
                                                         method ='lm',
                                                         )
        x = np.abs(xopt.x)
        
        ### Compute mean residual relative to data in percent
        res = np.abs(create_eqs(x,data,np.ones(4,dtype=np.float32),normed))
        res = np.mean(res)
        
        ### Assign 100 - res to success and if success < 0 assign 0 (i.e. solution deviates bymore than 100%!)
        ### So naturally the closer success to 100% the better the solution was found
        success = 100 - res
        if success < 0: success = 0
        
    except:
        x = x0.copy()
        x[:] = np.nan
        success = 0
        
    return x, success

#%%
def solve(s,weights):
    
    ### Prepare data for fitting
    data = s[['tau','A','occ','I']].values
    eps = float(s['eps'])  # Get normalization factor for I
    data[3] = data[3]/eps  # Normalize I
    data = data.astype(np.float32)
    
    ### Solve eq.-set with normalized and non-normalized I
    x_norm, success_norm = solve_eqs(data,weights,normed=True)
    x, success = solve_eqs(data,weights,normed=False)
    
    s_out=pd.Series({'koff': x_norm[0],
                     'konc': x_norm[1],
                     'N': x_norm[2],
                     'success': success_norm,
                     'eps_normstat': x[3],
                     })
    
    return s_out
