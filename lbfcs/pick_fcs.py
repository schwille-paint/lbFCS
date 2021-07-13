import numpy as np
import pandas as pd
import scipy.optimize

# Import own modules
import lbfcs.varfuncs as varfuncs
import lbfcs.multitau as multitau

#%%
def trace_ac(df,NoFrames,field = 'photons',compute_ac=True):
    '''
    Get fluorescence trace for single pick and normalized multitau autocorrelation function (AC) employing multitau.autocorrelate().

    Args:
        df (pandas.DataFrame): Single group picked localizations. See picasso.render and picasso_addon.autopick.
        NoFrames (int):        No. of frames in measurement, i.e. duration in frames.

    Returns:
        list:
        - [0] (numpy.array): Fluorescence trace of ``len=NoFrames``
        - [1] (numpy.array): First column corresponds to lagtimes, second to autocorrelation value.

    '''
    
    ############################# Prepare trace
    df[field] = df[field].abs() # Sometimes nagative values??
    df_sum = df[['frame',field]].groupby('frame').sum() # Sum multiple localizations in single frame

    trace = np.zeros(NoFrames)
    trace[df_sum.index.values] = df_sum[field].values # Add (summed) photons to trace for each frame
    
    ############################# Autocorrelate trace
    if compute_ac:
        ac = multitau.autocorrelate(trace,
                                    m=32,
                                    deltat=1,
                                    normalize=True,
                                    copy=False,
                                    dtype=np.float64(),
                                    )
    else:
        ac = 0
    
    return [trace,ac]

#%%
def fit_ac_lin(ac,max_it=10):
    ''' 
    Linearized iterative version of AC fit. 

    '''
    ###################################################### Define start parameters
    popt=np.empty([2]) # Init
    
    popt[0]=ac[1,1]-1                                               # Amplitude
    
    l_max=8                                                         # Maximum lagtime   
    try: l_max_nonan=np.where(np.isnan(-np.log(ac[1:,1]-1)))[0][0]  # First lagtime with NaN occurence
    except: l_max_nonan=len(ac)-1
    l_max=min(l_max,l_max_nonan)                                    # Finite value check
    
    popt[1]=(-np.log(ac[l_max,1]-1)+np.log(ac[1,1]-1))              # Correlation time tau corresponds to inverse of slope                          
    popt[1]/=(l_max-1)
    popt[1]=1/popt[1]
    
    ###################################################### Fit boundaries
    lowbounds=np.array([0,0])
    upbounds=np.array([np.inf,np.inf])
    
    ###################################################### Apply iterative fit
    if max_it==0: return popt[0],popt[1],l_max,0,np.nan
    
    else:
        popts=np.zeros((max_it,2))
        for i in range(max_it):
            l_max_return=l_max # Returned l_max corresponding to popt return
            try:
                ### Fit
                popts[i,:],pcov=scipy.optimize.curve_fit(varfuncs.ac_monoexp_lin,
                                                         ac[1:l_max,0],
                                                         -np.log(ac[1:l_max,1]-1),
                                                         popt,
                                                         bounds=(lowbounds,upbounds),
                                                         method='trf')
                
                ### Compare to previous fit result
                delta=np.max((popts[i,:]-popt)/popt)*100
                if delta<0.25: break
            
                ### Update for next iteration
                popt=popts[i,:]
                l_max=int(np.round(popt[1]*0.8))       # Optimum lagtime
                l_max=np.argmin(np.abs(ac[:,0]-l_max)) # Get lagtime closest to optimum (multitau!)
                l_max=max(3,l_max)                     # Make sure there are enough data points to fit
                l_max=min(l_max,l_max_nonan)           # Make sure there are no NaNs before maximum lagtime
                
            except:
                popt=np.ones(2)*np.nan
                delta=np.nan
                break
        
        return popt[0],popt[1],popts[0,0],popts[0,1],ac[l_max_return,0],i+1,delta

#%%
def props_fcs(df,NoFrames,max_it=1):
    '''
    1) Compute linearized autocorrelation and apply fit (tau,A)
    2) Compute I.
    '''
    
    #### Get trace and ac
    trace,ac = trace_ac(df,NoFrames)
    trace_ck,ac_ck = trace_ac(df,NoFrames,field = 'photons_ck') # Chung-Kennedy filtered trace
    
    ### Get autocorrelation fit results
    A,tau = fit_ac_lin(ac,max_it)[:2]  # Linearized fit
    
    ### Calculate I
    I = np.mean(trace_ck) # Chung-Kennedy filtered trace
    
    ### Assignment to series 
    s_out=pd.Series({'A':A,'tau':tau,'I':I,}) 
    
    return s_out