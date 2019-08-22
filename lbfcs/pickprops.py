# Import modules
import numpy as np
import scipy.optimize
import importlib
import pandas as pd
from tqdm import tqdm
# Import own modules
import lbfcs.varfuncs as varfuncs
importlib.reload(varfuncs)
import lbfcs.multitau as multitau
importlib.reload(multitau)

#%%
def get_ac(df,NoFrames):
    """ 
    1) Convert localizations of single pick to trace
        - trace(frame)=0 if no localization in frame
        - else trace(frame)=photons
        - length of trace will be NoFrames
    2) Compute normalized multi-tau autocorrelation function of trace -> help(multitau.autocorrelate)
    3) Least square fit of function f(tau)=mono_A*exp(-tau/mono_tau)+1 to normalized autocorrelation function -> help(pickprops.fit_ac_single)
    
    Parameters
    ---------
    df : pandas.DataFrame
        Picked localizations for single group. Required fields are 'frame' and 'photons'
    NoFrames : int
        Number of frames of localized image stack
    
    Returns
    -------
    s_out : pandas.Series
        Length: 6
        Column:
            'group' : int
        Index: 
            'trace' : numpy.ndarray
                trace
            'tau' : numpy.ndarray
                tau of autocorr.
            'g' : numpy.ndarray
                g(tau) autocorrelation value
            'mono_A' : float64
                Amplitude of monoexponential fit function
            'mono_tau' : float64
                Correlation time of monoeponential fit function
            'mono_chi' : float64
                Square root of chisquare value of fit with sigma=1 for all data points
    """
    ###################################################### Prepare trace
    # Get absolute values of photons, since sometimes negative values can be found
    df['photons']=df['photons'].abs() 
    # Sum multiple localizations in single frame
    df_sum=df[['frame','photons']].groupby('frame').sum()
    # Define trace of length=NoFrames with zero entries
    trace=np.zeros(NoFrames)
    # Add (summed) photons to trace for each frame
    trace[df_sum.index.values]=df_sum['photons'].values
    ###################################################### Generate autocorrelation
    ac=multitau.autocorrelate(trace,m=16, deltat=1,
                                 normalize=True,copy=False, dtype=np.float64())
    
    ###################################################### Fit mono exponential decay to autocorrelation
    mono_A,mono_tau,mono_chi=fit_ac_mono(ac) # Get fit
    mono_A_lin,mono_tau_lin=fit_ac_mono_lin(ac) # Get fit
    ###################################################### Calculate tau_b, tau_d from mono, just valid for N=1
    mono_taub=(mono_A+1)*(mono_tau/mono_A)
    mono_taud=(mono_A+1)*(mono_tau)
    ###################################################### Assignment to series 
    s_out=pd.Series({'trace':trace,
                     'tau':ac[1:-15,0],'g':ac[1:-15,1], # Autocorrelation function
                     'mono_A':mono_A,'mono_tau':mono_tau,'mono_chi':mono_chi, # mono exponential fit results
                     'mono_taub':mono_taub,'mono_taud':mono_taud, # tau_b, tau_d for mono just valid for N=1
                     'mono_A_lin':mono_A_lin,'mono_tau_lin':mono_tau_lin}) # mono exponential fit results for linearized fit
    
    return s_out

#%%
def fit_ac_mono(ac):
    """ 
    Least square fit of function f(tau)=mono_A*exp(-tau/mono_tau)+1 to normalized autocorrelation function.
    
    Parameters
    ---------
    ac : numpy.ndarray
        1st column should correspond to delay time tau of autocorrelation function.
        2nd column should correspond to value g(tau) of autocorrelation function
    
    Returns
    -------
    mono_A : float64
        Amplitude of monoexponential fit function
    mono_tau : float64
        Correlation time of monoeponential fit function
    mono_chi : float64
        Chisquare value of fit with sigma=1 for all data points   
    """
    ###################################################### Define start parameters
    p0=np.empty([2])
    p0[0]=ac[1,1] # Amplitude
    halfvalue=1.+(p0[0]-1.)/2 # Value of half decay of ac
    p0[1]=np.argmin(np.abs(ac[:,1]-halfvalue)) # tau
    ###################################################### Fit boundaries
    lowbounds=np.array([0,0])
    upbounds=np.array([np.inf,np.inf])
    ###################################################### Fit data
    try:
        popt,pcov=scipy.optimize.curve_fit(varfuncs.ac_monoexp,ac[1:-15,0],ac[1:-15,1],p0,bounds=(lowbounds,upbounds),method='trf')
    except RuntimeError:
        popt=p0
    except ValueError:
        popt=p0
    except TypeError:
        popt=p0
    
    ###################################################### Calculate chisquare    
    chisquare=np.sum(np.square(varfuncs.ac_monoexp(ac[:,0],*popt)-ac[:,1]))/((len(ac)-2))
    
    return popt[0],popt[1],np.sqrt(chisquare)

#%%
def fit_ac_mono_lin(ac):
    """ 
    Least square fit of function f_lin(tau)=-log(mono_A)-tau/mono_tau to linearized autocorrelation function -log(g(tau)-1).
    Only first 8 points of autocorrelation are used for fitting 
    
    Parameters
    ---------
    ac : numpy.ndarray
        1st column should correspond to delay time tau of autocorrelation function.
        2nd column should correspond to value g(tau) of autocorrelation function
    
    Returns
    -------
    mono_A_lin : float64
        Amplitude of monoexponential fit function
    mono_tau_lin : float64
        Correlation time of monoeponential fit function   
    """
    ###################################################### Fit function definition
    def ac_monoexp_lin(t,A,tau):
        g=t/tau-np.log(A)
        return g
    ###################################################### Define start parameters
    p0=np.empty([2])
    p0[0]=ac[1,1] # Amplitude
    halfvalue=1.+(p0[0]-1.)/2 # Value of half decay of ac
    p0[1]=np.argmin(np.abs(ac[:,1]-halfvalue)) # tau
    ###################################################### Fit boundaries
    lowbounds=np.array([0,0])
    upbounds=np.array([np.inf,np.inf])
    ###################################################### Fit data
    try:
        popt,pcov=scipy.optimize.curve_fit(ac_monoexp_lin,ac[1:10,0],-np.log(ac[1:10,1]-1),p0,bounds=(lowbounds,upbounds),method='trf')
    except RuntimeError:
        popt=p0
    except ValueError:
        popt=p0
    except TypeError:
        popt=p0
    
    ###################################################### Calculate chisquare    
    chisquare=np.sum(np.square(ac_monoexp_lin(ac[1:10,0],*popt)-np.log(ac[1:10,1]-1)))/(len(ac)-2)
    
    return popt[0],popt[1]
#%%
def get_tau(df,ignore=1,mode='ralf'):
    """ 
    1) Computes bright, dark-time distributions and number of events a la Picasso
    2) Least square fit of function f(tau)=1-exp(-t/tau) to experimental continuous distribution function (ECDF) 
        -> help(pickprops.fit_tau)
    
    Parameters
    ---------
    df : pandas.DataFrame
        Picked localizations for single group. Required columns are 'frame'.
    ignore : int
        Disrupted binding events by duration ignore will be treated as single event with bright time of 
        total duration of bridged events. Defaults to ignore=1.
        
    
    Returns
    -------
    s_out : pandas.Series
        Length: 9
        Column:0
            'group' : int
        Index: 
            'tau_b_dist' : numpy.ndarray
                Distribution of bright times with ignore value taken into account
            'tau_b' : float64
                Fit result of exponential function to bright time ECDF
            'tau_b_lin' : float64
                Fit result of line with offset=0 to linearized bright time ECDF given by -ln(1-ECDF)
            'tau_d_dist' : numpy.ndarray
                Distribution of dark times with ignore value taken into account
            'tau_b' : float64
                Fit result of exponential function to dark time ECDF
            'tau_b_lin' : float64
                Fit result of line with offset=0 to linearized dark time ECDF given by -ln(1-ECDF)
            'n_events' : float64
                Number of binding events
            'mean_event_times' : float64
                Mean of event times
            'std_event_times' : float64
                Standard deviation of event times
    """
    
    frames=df['frame'].values # Get sorted frames as numpy.ndarray
    frames.sort()
    ################################################################ Get tau_d distribution
    dframes=frames[1:]-frames[0:-1] # Get frame distances i.e. dark times
    dframes=dframes.astype(float) # Convert to float values for later multiplications
    
    tau_d_dist=dframes-1 # 1) We have to substract -1 to get real dark frames, e.g. suppose it was bright at frame=2 and frame=4
                         #    4-2=2 but there was actually only one dark frame.
                         #
                         # 2) Be aware that counting of gaps starts with first bright localization and ends with last
                         #    since values before or after are anyway artificially shortened, i.e. one bright event means no dark time!
    
    tau_d_dist=tau_d_dist[tau_d_dist>(ignore)] # Remove all dark times>ignore
    tau_d_dist=np.sort(tau_d_dist) # Sorted tau_d distribution
 
    ################################################################ Get tau_b_distribution
    dframes[dframes<=(ignore+1)]=0 # Set (bright) frames to 0 that have nnext neighbor distance <= ignore+1
    dframes[dframes>1]=1 # Set dark frames to 1
    dframes[dframes<1]=np.nan # Set bright frames to NaN
    
    mask_end=np.concatenate([dframes,[1]],axis=0) # Mask for end of events, add 1 at end
    frames_end=frames*mask_end # Apply mask to frames to get end frames of events
    frames_end=frames_end[~np.isnan(frames_end)] # get only non-NaN values, removal of bright frames
    
    mask_start=np.concatenate([[1],dframes],axis=0) # Mask for start of events, add one at start
    frames_start=frames*mask_start # Apply mask to frames to get start frames events
    frames_start=frames_start[~np.isnan(frames_start)] # get only non-NaN values, removal of bright frames
    
    tau_b_dist=frames_end-frames_start+1 # get tau_b distribution
    tau_b_dist=np.sort(tau_b_dist) # sort tau_b distribution
    
    ################################################################# Number of events and their timing
    n_events=float(np.size(tau_b_dist)) # Number of binding events

    ################ Extract tau's
    # Bright time
    tau_b,tau_b_off,tau_b_a=fit_tau(tau_b_dist,mode)
    # Dark time
    tau_d,tau_d_off,tau_d_a=fit_tau(tau_d_dist,mode)

    ###################################################### Assignment to series 
    s_out=pd.Series({'tau_b_dist':tau_b_dist,'tau_b':tau_b,'tau_b_off':tau_b_off,'tau_b_a':tau_b_a,'tau_b_mean':np.mean(tau_b_dist), # Bright times
                     'tau_d_dist':tau_d_dist,'tau_d':tau_d,'tau_d_off':tau_d_off,'tau_d_a':tau_d_a,'tau_d_mean':np.mean(tau_d_dist), # Dark times
                     'n_events':n_events}) # Events
    return s_out    
#%%     
def fit_tau(tau_dist,mode='ralf'):
    """ 
    Least square fit of function f(t)=a*(1-exp(-t/tau))+off to experimental continuous distribution function (ECDF) of tau_dist
    -> help(varfuncs.get_ecdf) equivalent to:
        matplotlib.pyplot.hist(tau_dist,bins=numpy.unique(tau_dist),normed=True,cumulative=True)
    
    Parameters
    ---------
    tau_dist : numpy.ndarray
        1 dimensional array of bright or dark times t 
    mode: str
        If mode is 'ralf' amplitude and offset will be floating freely, else fit will be performed with fixed parameters off=0 and a=1 as it was published
    Returns
    -------
    tau : float64
        tau as obtained by fitting f(t) to ECDF(t) with least squares.
    off : float64
        off as obtained by fitting f(t) to ECDF(t) with least squares. Set to 0 for non 'ralf' mode.
    a : float64
        a as obtained by fitting f(t) to ECDF(t) with least squares. Set to 1 for non 'ralf' mode.
    """

    try:
        #### Get ECDF
        tau_bins,tau_ecdf=varfuncs.get_ecdf(tau_dist)
    
        ##### Define start parameter
        p0=np.zeros(3)   
        p0[0]=np.mean(tau_bins) # tau
        p0[1]=np.min(tau_ecdf) # offset
        p0[2]=np.max(tau_ecdf)-p0[1] # amplitude
        
        #### Fit
        if mode=='ralf':
            popt,pcov=scipy.optimize.curve_fit(varfuncs.ecdf_exp,tau_bins,tau_ecdf,p0) 
            tau=popt[0]
            off=popt[1]
            a=popt[2]
        else:
            popt,pcov=scipy.optimize.curve_fit(varfuncs.ecdf_exp,tau_bins,tau_ecdf,p0[0]) 
            tau=popt[0]
            off=0
            a=1
    
    except IndexError:
        tau,off,a=np.nan,np.nan,np.nan        
    except RuntimeError:
        tau,off,a=np.nan,np.nan,np.nan
    except ValueError:
        tau,off,a=np.nan,np.nan,np.nan
    except TypeError:
        tau,off,a=np.nan,np.nan,np.nan
        
    return tau,off,a

#%%
def get_other(df):
    """ 
    Get mean and std values for a single group.
    
    Parameters
    ---------
    df : pandas.DataFrame
        Picked localizations for single group. Required columns are 'frame'.

    Returns
    -------
    s_out : pandas.Series
        Length: 6
        Column:
            'group' : int
        Index: 
            'mean_frame' : float64
                Mean of frames for all localizations in group
            'mean_x' : float64
                Mean x position
            'mean_y' : float64
                Mean y position
            'mean_photons' : float64
                Mean of photons for all localizations in group
            'mean_bg' : float64
                Mean background
            'std_frame' : flaot64
                Standard deviation of frames for all localizations in group
            'std_x' : float64
                Standard deviation of x position
            'std_y' : float64
                Standard deviation of y position
            'std_photons' : float64
                Standard deviation of photons for all localizations in group
            'n_locs' : int
                Number of localizations in group
    """
    # Get mean values
    s_mean=df[['frame','x','y','photons','bg']].mean()
    # Get number of localizations
    s_mean['n_locs']=len(df)
    mean_idx={'frame':'mean_frame','x':'mean_x','y':'mean_y','photons':'mean_photons','mean_bg':'bg'}
    # Get std values
    s_std=df[['frame','x','y','photons']].std()
    std_idx={'frame':'std_frame','x':'std_x','y':'std_y','photons':'std_photons'}
    # Combine output
    s_out=pd.concat([s_mean.rename(mean_idx),s_std.rename(std_idx)])
    
    return s_out

#%%
def get_props(df,NoFrames,ignore,mode='ralf'):
    """ 
    Wrapper function to combine:
        - pickprops.get_ac(df,NoFrames)
        - pickprops.get_tau(df,ignore)
        - pickprops.get_other(df)
    
    Parameters
    ---------
    df : pandas.DataFrame
        'locs' of locs_picked.hdf5 as given by Picasso
    NoFrames: int
        Number of frames of image stack corresponding to locs_picked.hdf5
    ignore: int
        Ignore as defined in props.get_tau
    Returns
    -------
    s : pandas.DataFrame
        Columns as defined in individual functions get_ac, get_tau. Index corresponds to 'group'.
    """
    
    # Call individual functions   
    s_ac=get_ac(df,NoFrames)
    s_tau=get_tau(df,ignore,mode)
    s_other=get_other(df)
    # Combine output
    s_out=pd.concat([s_ac,s_tau,s_other])
    
    return s_out

#%%
def apply_props(df,conc,NoFrames,ignore,mode='ralf'): 
    """
    Applies pick_props.get_props(df,NoFrames,ignore) to each group in non-parallelized manner. Progressbar is shown under calculation.
    """
    tqdm.pandas() # For progressbar under apply
    df_props=df.groupby('group').progress_apply(lambda df: get_props(df,NoFrames,ignore,mode))
    df_props['conc']=conc
    
    return df_props

#%%
def apply_props_dask(df,conc,NoFrames,ignore,NoPartitions,mode='ralf'): 
    """
    Applies pick_props.get_props(df,NoFrames,ignore) to each group in parallelized manner using dask by splitting df into 
    various partitions.
    """
    ########### Load packages
    import dask
    import dask.multiprocessing
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar
    ########### Globally set dask scheduler to processes
    dask.set_options(get=dask.multiprocessing.get)
    ########### Partionate df using dask for parallelized computation
    df=df.set_index('group') # Set group as index otherwise groups will be split during partition!!!
    df=dd.from_pandas(df,npartitions=NoPartitions) 
    ########### Define apply_props for dask which will be applied to different partitions of df
    def apply_props_2part(df,NoFrames,ignore,mode): return df.groupby('group').apply(lambda df: get_props(df,NoFrames,ignore,mode))
    ########### Map apply_props_2part to every partition of df for parallelized computing    
    with ProgressBar():
        df_props=df.map_partitions(apply_props_2part,NoFrames,ignore,mode).compute()
    df_props['conc']=conc
    return df_props

#%%
def _kin_filter(df):
    """ 
    Kinetics filter for '_props' DataFrame as generated by pickprops.py. 
          
    """
    #### Function definitions percentiles
    from numpy import percentile as perc

    #### Copy df
    df_drop=df.copy()
             
    ####Remove groups with ...
    #### ... deviation between linear fit and exponential fit higher than 20%
    istrue=(np.abs(df_drop.mono_tau_lin-df_drop.mono_tau)/df_drop.mono_tau_lin)>0.2
    df_drop.drop(df_drop.loc[istrue].index,inplace=True)
    
    #### ... mean_frame higher than 1.15 x 50-percentile
    #### ............or lower than 0.85 x 50-percentile
    istrue=df_drop.mean_frame>(1.15*perc(df_drop.mean_frame,50))
    istrue=istrue | (df_drop.mean_frame<(0.85*perc(df_drop.mean_frame,50)))
    df_drop.drop(df_drop.loc[istrue].index,inplace=True)
    
    #### ... or std_frame lower than 0.85 x 50-percentile
    istrue=df_drop.std_frame<(0.85*perc(df_drop.std_frame,50))
    df_drop.drop(df_drop.loc[istrue].index,inplace=True)
    
    #### Boolean: mono_tau higher than 2 x 50-percentile
    #### ..............or lower than 0.5 x 50-percentile
    istrue=df_drop.mono_tau>(2*perc(df_drop.mono_tau,50))
    istrue=istrue | (df_drop.mono_tau<(0.5*perc(df_drop.mono_tau,50)))
    df_drop.drop(df_drop.loc[istrue].index,inplace=True)
    
    #### Boolean: mono_A higher than 4 x 50-percentile
    istrue=df_drop.mono_A>(4*perc(df_drop.mono_A,50))
    df_drop.drop(df_drop.loc[istrue].index,inplace=True)
  
    return df_drop












