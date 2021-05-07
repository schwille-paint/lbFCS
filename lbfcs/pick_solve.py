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
def create_eqs(x,data,weights,known_eps):
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
            
    if known_eps: # In this case no eps will be fitted
        system[3] = w[3] * Inormed_func(x[0],x[1],x[2],data[3])
        
    else:  # Fit eps value
        system[3] = w[3] * I_func(x[0],x[1],x[2],x[3],data[3])
        
    ### Remove NaNs
    system[np.isnan(system)] = 0
    
    return system

#%%
@numba.jit(nopython=True, nogil=True, cache=False)
def estimate_unknowns(data,known_eps):
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
    N = (1/A) * (koff/konc)

    x0[0] = koff 
    x0[1] = konc
    x0[2] = N
    x0[3] = 1 # Estimate for eps after normalization!
    
    ### Remove eps estimate from x0 if eps is not assumed to be unknown
    if known_eps: x0 = x0[:3]

    return x0
    
#%%
def solve_eqs(data,weights,known_eps):
    '''
    Solve equation system as set-up by ``create_eqs()``.
    '''
    weights = np.array(weights,dtype=np.float32)
    
    ### Get estimate x0
    x0 = estimate_unknowns(data,known_eps)
    
    ### Solve system of equations
    try:
        xopt = optimize.least_squares(lambda x: create_eqs(x,data,weights,known_eps),
                                                         x0,
                                                         method ='lm',
                                                         )
        x = np.abs(xopt.x)
        
        ### Compute mean residual relative to data in percent
        res = np.abs(create_eqs(x,data,np.ones(4,dtype=np.float32),known_eps))
        res = np.mean(res)
        
        ### Assign 100 - res to success and if success < 0 assign 0 (i.e. solution deviates bymore than 100%!)
        ### So naturally the closer success to 100% the better the solution was found
        success = 100 - res
        if success < 0: success = 0
        
    except:
        x = x0.copy()
        x[:] = np.nan
        success = 0
        
    return x, success, x0

#%%
def solve(s,weights):
    
    ### Prepare data for fitting
    data = s[['tau','A','occ','I']].values
    eps = float(s['eps'])  # Get normalization factor for I
    data[3] = data[3]/eps  # Normalize I in any case, i.e. does not matter if eps is assumed to be known or unkown!
    data = data.astype(np.float32)
    
    ### Solve eq.-set with normalized and non-normalized I
    x_knowneps, success_knowneps, xstart_knowneps = solve_eqs(data, weights, known_eps = True)
    x, success, xstart = solve_eqs(data, [1,1,1,1], known_eps = False)
    
    ### Compute relative mean deviation between solutions to indicate status of normalization
    x0 = np.append(x_knowneps,1) # Normalized to solution assuming known eps
    success_epsnorm = 100 - np.mean(np.abs(x - x0) / x0) * 100
    
    ### Solve eq. set for occ,Inorm after p,N
    p,N_p = solve_pN(data[3],data[2])
    
    ### Combine to pandas.Series output
    s_out=pd.Series({'koff': x_knowneps[0],
                     'konc': x_knowneps[1],
                     'N': x_knowneps[2],
                     'success': success_knowneps,
                     'success_epsnorm': success_epsnorm,
                     'p':p,
                     'N_p':N_p,
                     'N_start': xstart_knowneps[2],
                     })
    
    return s_out

#%%

def solve_pN(Inorm,occ):
    '''
    Return solution (p,N) to eq. set defined by observables (Inorm,occ) with unknows (p,N).
    Defining set of eq.:
        1) Inorm = N*p
        2) occ   = 1 - (1-p)^N
    
    Defining eq. for exact solution for p:
    ln(1-p)/p - ln(1-occ)/Inorm = 0
    
    With polynomial approximate defining eq. up to order 7 for solution for p:
    (-1 - p/2 - p^2/3 - p^3/4 - p^4/5 -p^5/6 - p^6/7 - p^7/8) +.... - (ln(1-occ)/Inorm) = 0
    or:
    (-1 - ln(1-occ)/Inorm) - p/2 - p^2/3 - p^3/4 - p^4/5 -p^5/6 - p^6/7 - p^7/8 = 0
    
    Approximation for p will be solved using numpy.roots.Reasonable solution for p will be selected by the rules:
        1) p must be real
        2) p must lie in interval 0 < p < p
    
    Then N is computed using eq.: N = Inorm/p
    '''
    ### Define polynomial coeffs. for approx. sol. to p up to order 7
    ### First element correspond to highest order (7) last element to lowest order (0) as needed by numpy.roots
    cs = -1/np.arange(8,0,-1)
    cs[-1] -=  np.log(1-occ)/Inorm
    
    try:
        ### Solve polynom
        r = np.roots(cs)
        p = r[(r.imag == 0) & (r.real > 0) & (r.real < 1)].real[0]
        
        ### Select reasonable solution, if not possible assign NaNs
        N = Inorm/p
    except:
        p = np.nan
        N = np.nan
    
    return p,N
