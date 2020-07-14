# Import modules
import os
import numpy as np
import scipy.optimize
import importlib
import pandas as pd
from tqdm import tqdm
import dask.dataframe as dd
import multiprocessing as mp
import time
import warnings
warnings.filterwarnings("ignore")

# Import own modules
import lbfcs.varfuncs as varfuncs
import lbfcs.multitau as multitau
import picasso_addon.io as addon_io

importlib.reload(varfuncs)
importlib.reload(multitau)

#%%
def trace_and_ac(df,NoFrames):
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
    df['photons']=df['photons'].abs() # Sometimes nagative values??
    df_sum=df[['frame','photons']].groupby('frame').sum() # Sum multiple localizations in single frame

    trace=np.zeros(NoFrames)
    trace[df_sum.index.values]=df_sum['photons'].values # Add (summed) photons to trace for each frame
    
    ############################# Autocorrelate trace
    ac=multitau.autocorrelate(trace,
                              m=32,
                              deltat=1,
                              normalize=True,
                              copy=False,
                              dtype=np.float64(),
                              )
    
    return [trace,ac]
    
#%%
def fit_ac(ac,max_it=10):
    ''' 
    Exponential iterative version of AC fit. 

    '''
    ###################################################### Define start parameters
    popt=np.empty([2]) # Init
    
    popt[0]=ac[1,1]-1                                               # Amplitude
    
    l_max=8                                                         # Maximum lagtime for correlation time estimate  
    try: l_max_nonan=np.where(np.isnan(-np.log(ac[1:,1]-1)))[0][0]  # First lagtime with NaN occurence
    except: l_max_nonan=len(ac)-1
    l_max=min(l_max,l_max_nonan)                                    # Finite value check
    
    popt[1]=(-np.log(ac[l_max,1]-1)+np.log(ac[1,1]-1))              # Correlation time tau corresponds to inverse of slope                          
    popt[1]/=(l_max-1)
    popt[1]=1/popt[1]
    
    l_max=-15                                                       # Maximum lagtime used for first fit
    
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
                popts[i,:],pcov=scipy.optimize.curve_fit(varfuncs.ac_monoexp,
                                                         ac[1:l_max,0],
                                                         ac[1:l_max,1],
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
                
            except:
                popt=np.ones(2)*np.nan
                delta=np.nan
                break
        
        return popt[0],popt[1],ac[l_max_return,0],i+1,delta

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
        
        return popt[0],popt[1],ac[l_max_return,0],i+1,delta

#%%
def props_fcs(df,NoFrames,max_it=10):
    """ 
    Apply least square fit to normalized autocorrelation function (AC) using ``AC(l) = A*exp(-l/tau) + 1`` by two methods:
        
        - Exponential least square fit using full AC (``A``,``tau``). See fit_ac().
        - Linearized fitting (by applying logarithmic transformation) using only first 8 lagtimes of AC (``A_lin``,``tau_lin``). See fit_ac_lin().
    
    Calculate brightness value B, i.e. variance/mean of fluorescence trace (``B``).
    
    Args:
        df (pandas.DataFrame): Single group picked localizations. See picasso.render and picasso_addon.autopick.
        NoFrames (int):        No. of frames in measurement, i.e. duration in frames.
    
    Returns:
        pandas.Series:
        - ``A`` Amplitude resulting from exponential fit to AC.
        - ``tau`` Autocorrelation time resulting from exponential fit to AC.
        - ``A_lin`` Amplitude resulting from linearized fit to AC.
        - ``tau_lin`` Autocorrelation time resulting from resulting fit to AC.
        - ``B`` Brighntess fo fluorescence trace, i.e. its variance/mean.
    """
    
    ############################# Get trace and ac
    trace,ac=trace_and_ac(df,NoFrames)
    ac_zeros=np.sum(ac[1:,1]<0.1) # Number of lagtimes with almost zero ac values, can be used for filtering
    
    ############################# Get autocorrelation fit results
    A,tau,lmax,it,delta                     = fit_ac(ac,1)
    A_lin,tau_lin,lmax_lin,it_lin,delta_lin = fit_ac_lin(ac,max_it)

    ############################# Calculate brightness value B using trace
    B=np.var(trace)/np.mean(trace)

    ############################# Assignment to series 
    s_out=pd.Series({'ac_zeros':ac_zeros,                                                                      # Number of almost zeros entries in AC
                     'A':A,'tau':tau,'lmax':lmax,'it':it,'delta':delta,                                        # AC fit results
                     'A_lin':A_lin,'tau_lin':tau_lin,'lmax_lin':lmax_lin,'it_lin':it_lin,'delta_lin':delta_lin,# AC linearized iterative fit results
                     'B':B,                                                                                    # Brightness
                     }) 
    
    return s_out

#%%
def darkbright_times(df,ignore):
    ''' 
    Compute bright, dark-time distributions and number of events with allowed ``ignore`` value a la picasso.render.
    
    Args:
        df(pandas.DataFrame): Single group picked localizations. See picasso.render and picasso_addon.autopick.
        ignore(int=1):         Disrupted binding events by duration ignore will be treated as single event with bright time of total duration of bridged events.
        
    Returns:
        list:
        - [0](numpy.array): Distribution of dark times with ``ignore`` value taken into account.
        - [1](numpy.array): Distribution of bright times with ``ignore`` value taken into account.
        - [2](int):         Number of binding events with ``ignore`` value taken into account.
    '''
    
    frames=df['frame'].values # Get sorted frames as numpy.ndarray
    frames.sort()
    
    ############################# Dark time distribution
    dframes=frames[1:]-frames[0:-1] # Get frame distances i.e. dark times
    dframes=dframes.astype(float) # Convert to float values for later multiplications
    
    tau_d_dist=dframes-1 # 1) We have to substract -1 to get real dark frames, e.g. suppose it was bright at frame=2 and frame=4
                         #    4-2=2 but there was actually only one dark frame.
                         #
                         # 2) Be aware that counting of gaps starts with first bright localization and ends with last
                         #    since values before or after are anyway artificially shortened, i.e. one bright event means no dark time!
    
    tau_d_dist=tau_d_dist[tau_d_dist>(ignore)] # Remove all dark times>ignore
    tau_d_dist=np.sort(tau_d_dist) # Sorted tau_d distribution
 
    ############################# Bright time distribution
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
    
    ############################# Number of events
    n_events=float(np.size(tau_b_dist))

    return [tau_d_dist,tau_b_dist,n_events]
   
#%%     
def fit_times(tau_dist,mode='ralf'):
    ''' 
    Least square fit of function ``ECDF(t)=a*(1-exp(-t/tau))+off`` to experimental continuous distribution function (ECDF) of bright or dark times distribution ``tau_dist``.
    
    Args:
        tau_dist(numpy.array): Dark or bright times distribution as returned by darkbright_times()
        mode(str):             If mode is 'ralf' amplitude and offset will be floating freely, else fit will be performed with fixed parameters off=0 and a=1 as it was published
    
    Returns:
        list:
        - [0](float): Time fit result ``tau``
        - [1](float): Offset fit result ``off``. Set to 0 for non ``'ralf'`` mode.
        - [2](float): Amplitude fit result ``a``. Set to 1 for non ``'ralf'`` mode.
        - [3] (int):  Number of unique times in given bright or dark times distribution ``tau_dist``.
    '''
    
    #### Get number of unique values in tau distribution to decide if fitting makes sense
    ulen=len(np.unique(tau_dist))
    
    if ulen<=3: # Fit has 3 degrees of freedom hence more than 4 datapoints are necessary
        tau,off,a=np.mean(tau_dist),0,1
        return tau,off,a,ulen
    else:     
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
            
    return [tau,off,a,ulen]

#%%
def props_qpaint(df,ignore,mode='ralf'):
    '''
    Compute bright, dark-time distributions and number of events with allowed ``ignore`` value a la picasso.render. See darkbright_times().
    Least square fit of function ``ECDF(t)=a*(1-exp(-t/tau))+off`` to experimental continuous distribution function (ECDF) of bright or dark times distribution. See fit_times().
    
    '''
    
    ################ Get bright and dark time distributions
    tau_d_dist,tau_b_dist,n_events = darkbright_times(df,ignore=1)
    
    ################ Extract average bright and dark times
    tau_b,tau_b_off,tau_b_a,tau_b_ulen = fit_times(tau_b_dist,mode)  # Bright time
    tau_d,tau_d_off,tau_d_a,tau_d_ulen = fit_times(tau_d_dist,mode)  # Dark time

    ###################################################### Assignment to series 
    s_out=pd.Series({'tau_b':tau_b,'tau_b_off':tau_b_off,'tau_b_a':tau_b_a,'tau_b_mean':np.mean(tau_b_dist), # Bright times
                     'tau_d':tau_d,'tau_d_off':tau_d_off,'tau_d_a':tau_d_a,'tau_d_ulen':tau_d_ulen,'tau_d_mean':np.mean(tau_d_dist), # Dark times
                     'n_events':n_events}) # Events
    return s_out  
#%%
def props_other(df,NoFrames):
    '''
    Get other properties.

    '''
    ### Get all mean values
    s_out=df.mean()
    
    ### Localizations
    s_out['n_locs']=len(df)/NoFrames # append no. of locs. per frame
    
    ### Photons
    s_out[['photons','bg']]=df[['photons','bg']].mean()
    s_out['std_photons']=df['photons'].std(ddof=0)   # Biased standard deviation
    s_out['std_bg']=df['bg'].std(ddof=0)             # Biased standard deviation
    
    ### Set lpx and lpy to standard deviation in x,y for proper visualization in picasso.render
    s_out[['lpx','lpy',]]=df[['x','y']].std()
    
    ### Add field std_frame
    s_out['std_frame']=df['frame'].std(ddof=0) # Biased standard deviation
    
    return s_out

#%%
def get_props(df,NoFrames,ignore,mode='ralf'):
    ''' 
    Combine outputs of:
        
        - prop_fcs(df,NoFrames)
        - props_qpaint(df,ignore,mode)
        - props_other(df)
    
    '''
    
    ### Call individual functions   
    s_fcs=props_fcs(df,NoFrames)
    s_qpaint=props_qpaint(df,ignore,mode)
    s_other=props_other(df,NoFrames)
    
    ### Combine output
    s_out=pd.concat([s_fcs,s_qpaint,s_other])
    
    return s_out

#%%
def apply_props(df,conc,NoFrames,ignore,mode='ralf'): 
    """
    Applies pick_props.get_props(df,NoFrames,ignore) to each group in non-parallelized manner. Progressbar is shown under calculation.
    """
    df=df.set_index('group')
    tqdm.pandas() # For progressbar under apply
    df_props=df.groupby('group').progress_apply(lambda df: get_props(df,NoFrames,ignore,mode))

    return df_props

#%%
def apply_props_dask(df,conc,NoFrames,ignore,mode='ralf'): 
    """
    Applies pick_props.get_props(df,NoFrames,ignore) to each group in parallelized manner using dask by splitting df into 
    various partitions.
    """
     
    ########### Define apply_props for dask which will be applied to different partitions of df
    def apply_props_2part(df,NoFrames,ignore,mode): return df.groupby('group').apply(lambda df: get_props(df,NoFrames,ignore,mode))

    ########## Partinioning and computing
    t0=time.time() # Timing
    
    ### Set up DataFrame for dask
    df=df.set_index('group') # Set group as index otherwise groups will be split during partition!!! 
    NoPartitions=max(1,int(0.8 * mp.cpu_count()))
    df=dd.from_pandas(df,npartitions=NoPartitions)                
        
    ### Compute using running dask cluster, if no cluster is running dask will start one with default settings (maybe slow since not optimized for computation!)
    df_props=df.map_partitions(apply_props_2part,NoFrames,ignore,mode).compute()
    
    dt=time.time()-t0
    print('... Computation time %.1f s'%(dt)) 
    
    return df_props


#%%
def cluster_setup_howto():
    '''
    Print instruction howto start a DASK local cluster for efficient computation of apply_props_dask().
    Fixed ``scheduler_port=8787`` is used to easily reconnect to cluster once it was started.
    
    '''

    print('Please first start a DASK LocalCluster by running following command in directly in IPython shell:')
    print()
    print('Client(n_workers=max(1,int(0.8 * mp.cpu_count())),')
    print('       processes=True,')
    print('       threads_per_worker=1,')
    print('       scheduler_port=8787,')
    print('       dashboard_address=":1234")') 
    return


#%%
def main(locs,info,path,conc,**params):
    '''
    Get immobile properties for each group in _picked.hdf5 file (see `picasso.addon`_) and filter.
    
    
    Args:
        locs(pandas.DataFrame):    Grouped localization list, i.e. _picked.hdf5 as in `picasso.addon`_
        info(list):                Info _picked.yaml to _picked.hdf5 localizations as list of dictionaries.
        path(str):                 Path to _picked.hdf5 file.
        conc(float):               Imager concentration
        
    Keyword Args:
        ignore(int=1):             Maximum interruption (frames) allowed to be regarded as one bright time.
        parallel(bool=False):      Apply parallel computing using DASK? Local cluster should be started before according to cluster_setup_howto()
    
    Returns:
        list:
            
        - [0](dict):             Dict of keyword arguments passed to function.
        - [1](pandas.DataFrame): Immobile properties of each group in ``locs`` as calulated by apply_props()
    '''
    
    ### Path of file that is processed and number of frames
    path=os.path.splitext(path)[0]
    NoFrames=info[0]['Frames']
    
    ### Define standard 
    standard_params={'ignore': 3,
                     'parallel': False,
                     }
    ### Set standard if not contained in params
    for key, value in standard_params.items():
        try:
            params[key]
            if params[key]==None: params[key]=standard_params[key]
        except:
            params[key]=standard_params[key]
    
    ### Remove keys in params that are not needed
    delete_key=[]
    for key, value in params.items():
        if key not in standard_params.keys():
            delete_key.extend([key])
    for key in delete_key:
        del params[key]
        
    ### Procsessing marks: extension&generatedby
    params['generatedby']='lbfcs.pickprops.main()'
    
    ##################################### Calculate kinetic properties
    print('Calculating kinetic information ...')
    if params['parallel']==True:
        print('... in parallel')
        locs_props=apply_props_dask(locs,
                                    conc,
                                    NoFrames,
                                    params['ignore'],
                                    mode='ralf',
                                    )
    else:
        locs_props=apply_props(locs,
                               conc,
                               NoFrames,
                               params['ignore'],
                               mode='ralf',
                               )
    
    locs_props['conc']=conc
    locs_props['M']=NoFrames

    ##################################### Saving
    print('Saving _props ...')
    locs_props.reset_index(inplace=True) # Write group index into separate column
    info_props=info.copy()+[params]
    addon_io.save_locs(path+'_props.hdf5',
                       locs_props,
                       info_props,
                       mode='picasso_compatible')
           
    return [params,locs_props]










