import numpy as np
import pandas as pd
import scipy.optimize

# Import own modules
import lbfcs.varfuncs as varfuncs


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
    dframes=dframes.astype(float)   # Convert to float values for later multiplications
    
    tau_d_dist=dframes-1 # 1) We have to substract -1 to get real dark frames, e.g. suppose it was bright at frame=2 and frame=4
                                        #    4-2=2 but there was actually only one dark frame.
                                        #
                                        # 2) Be aware that counting of gaps starts with first bright localization and ends with last
                                        #    since values before or after are anyway artificially shortened, i.e. one bright event means no dark time!
    
    tau_d_dist=tau_d_dist[tau_d_dist>(ignore)] # Remove all dark times <= ignore
    tau_d_dist=np.sort(tau_d_dist)             # Sorted tau_d distribution
 
    ############################# Bright time distribution
    dframes[dframes<=(ignore+1)]=0 # Set (bright) frames to 0, i.e.those that have next neighbor distance <= ignore+1
    dframes[dframes>1]=1           # Set dark frames to 1
    dframes[dframes<1]=np.nan      # Set bright frames to NaN
    
    mask_end=np.concatenate([dframes,[1]],axis=0)      # Mask for end of events, add 1 at end
    frames_end=frames*mask_end                         # Apply mask to frames to get end frames of events
    frames_end=frames_end[~np.isnan(frames_end)]       # get only non-NaN values, removal of bright frames
    
    mask_start=np.concatenate([[1],dframes],axis=0)    # Mask for start of events, add one at start
    frames_start=frames*mask_start                     # Apply mask to frames to get start frames events
    frames_start=frames_start[~np.isnan(frames_start)] # get only non-NaN values, removal of bright frames
    
    tau_b_dist=frames_end-frames_start+1 # get tau_b distribution
    tau_b_dist=np.sort(tau_b_dist)       # sort tau_b distribution
    
    ############################# Number of events
    n_events=float(np.size(tau_b_dist))

    return [tau_d_dist,tau_b_dist,n_events]
   
#%%     
def fit_times(tau_dist,mode):
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
    tau_d_dist,tau_b_dist,n_events = darkbright_times(df,ignore)
    
    ################ Extract average bright and dark times
    tau_b,tau_b_off,tau_b_a,tau_b_ulen = fit_times(tau_b_dist,mode)  # Bright time
    tau_d,tau_d_off,tau_d_a,tau_d_ulen = fit_times(tau_d_dist,mode)  # Dark time

    ###################################################### Assignment to series 
    s_out=pd.Series({'tau_b':tau_b,
                     # 'tau_b_off':tau_b_off,'tau_b_a':tau_b_a, # Bright times additional info
                     'tau_d':tau_d,
                     # 'tau_d_off':tau_d_off,'tau_d_a':tau_d_a, # Dark times additional info
                     'events':n_events,                         # Events
                     'ignore':ignore,                           # Used ignore value 
                     }) 
    return s_out