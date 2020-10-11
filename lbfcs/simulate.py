import numpy as np 
import pandas as pd
import numba
import time

import picasso_addon.io as addon_io

LOCS_DTYPE = [
    ('frame', 'u4'),
    ('x', 'f4'),
    ('y', 'f4'),
    ('photons', 'f4'),
    ('photons_ck', 'f4'),
    ('sx', 'f4'),
    ('sy', 'f4'),
    ('bg', 'f4'),
    ('lpx', 'f4'),
    ('lpy', 'f4'),
    ('group', 'u4'),
    ('M','u4'),
    ('exp','f4'),
    ('koff', 'f4'),
    ('kon', 'f4'),
    ('conc', 'f4'),
    ('N', 'u4'),
#    ('net_gradient', 'f4'),
#    ('likelihood', 'f4'),
#    ('iterations', 'i4'),
]

#%%
def generate_times(M,CycleTime,koff,kon,c,factor):
    
    ### Checks
    assert isinstance(M,int)
  
    ### Unit conversion from [s] to [frames]
    tb=(1/koff)/CycleTime         # Bright time
    td=(1/(kon*c))/CycleTime      # Dark time
    tcycle=tb+td                  # Duty cycle time
    Mmin=factor*int(np.ceil(M/tcycle)) # Estimate of needed cycles to create trace of length ~ factor x M
    
    ### Generate bright and dark times random dsitributions 
    bright=np.random.exponential(tb,Mmin)
    dark=np.random.exponential(td,Mmin)
    
    ### Duty cycle = dark + bright
    ### ts = start of bright part of duty cycle up to ...
    ### te = end of duty cycle 
    te=np.cumsum(dark+bright)
    ts=te-bright
    
    return ts,te
    
#%%
@numba.jit(nopython=True,nogil=True,cache=True)
def times_to_trace(ts,te,M):
    
  
    Msim=len(te) # Iterator: len(te)=len(ts) by definition
    trace=np.zeros(int(np.ceil(te[-1])),dtype=np.float64) # Init trace

    for i in range(0,Msim): # Assign photons
        its=int(ts[i])
        ite=int(te[i])
        ### Bright event within one frame
        if its==ite:
            trace[its]+=te[i]-ts[i] 
    	### Partial starts and ends of bright events
        if ite-its>=1:
            trace[its]+=1-ts[i]%1
            trace[ite]+=te[i]%1
    	### Full photons for bright events spanning full frames
        if ite-its>1:
            trace[its+1:ite]+=1
    
    ### Cut trace to needed length
    start=int(0.2*len(trace))
    trace=trace[start:start+M]
    
    return trace
   
#%%
def generate_trace(M,CycleTime,N,koff,kon,c,factor=10):
    
    ### Checks
    assert isinstance(M,int)
    assert isinstance(N,int)
    
    traceN=np.zeros(M)
    for i in range(N):
        ts,te = generate_times(M,CycleTime,koff,kon,c,factor)
        trace = times_to_trace(ts,te,M)
        traceN+=trace
        
    return traceN

#%%
def generate_locs(savepath,reps,M,CycleTime,N,koff,kon,c,factor=10):
    
    ### Print statement
    print('%ix simulations of:'%reps)
    print('M:    %i'%M)
    print('exp:  %.2f [s]'%CycleTime)
    print('N:    %i'%N)
    print('koff: %.1e [1/s]'%koff)
    print('kon:  %.1e [1/Ms]'%kon)
    print('conc: %.1e [M]'%c)
    
    t0=time.time()
    ### Checks
    assert isinstance(M,int)
    assert isinstance(N,int)
    assert isinstance(reps,int)
    
    ### Init locs
    locs=np.zeros(reps*M,dtype=LOCS_DTYPE)
    frames=np.arange(0,M,1).astype(int) # Timestamp
    
    ### Fill locs
    for r in range(reps):
        ### Generate trace
        trace = generate_trace(M,CycleTime,N,koff,kon,c,factor)
        
        ### Assign
        start=r*M
        end=(r+1)*M
        locs['frame'][start:end]=frames
        locs['photons'][start:end]=trace  
        locs['group'][start:end]=r 
    
    ### Assign input parameters
    locs['M']=M
    locs['exp']=CycleTime
    locs['N']=N
    locs['koff']=koff
    locs['kon']=kon
    locs['conc']=c
    
    ### Remove zeros in photons
    locs=locs[locs['photons']>0]
    
    ### Add photon noise to photons_ck
    locs['photons_ck']=locs['photons'] # Copy
    
    ### Convert 
    locs=pd.DataFrame(locs)
    
    print('Total time: %.2f [s]'%(time.time()-t0))
    print()
    
    ### Saving
    print('Saving ...')
    info=[{'reps':reps,
           'Frames':M,
           'exp':CycleTime,
           'N':N,
           'koff':koff,
           'kon':kon,
           'conc':c,
           }]
    addon_io.save_locs(savepath,
                       locs,
                       info,
                       mode='picasso_compatible')
    return locs

