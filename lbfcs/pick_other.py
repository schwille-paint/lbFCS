import numpy as np
import pandas as pd

#%%
def extract_eps(p):
    
    ### Get histogram of single photon counts
    y = np.histogram(p,800,(0,4000))[0]  # Only values returned in numba
    x = np.arange(2.5,4000,5)            # These are the bins
    
    ### Get histogram of double photon counts
    y2 = np.histogram(2*p,800,(0,4000))[0]  # Only values returned in numba
    y2 = y2 * 1.5                           # Multiply values to roughly equal heights!!!
    
    y = y.astype(np.float32)
    y2 = y2.astype(np.float32)
    ### Smooth substraction using lagtimes
    ### i.e. subtracting doubled photon histograms over and over by moving it to the right
    y_diff = y.copy()
    y2_lag = y2.copy()
    for l in range(0,100):
        y_diff -= y2_lag               # Substract y2_lag from y
        y2_lag = np.append(0,y2_lag)   # Add zero to start
        y2_lag = y2_lag[:-1]           # Remove last entry 
    
    y_diff[y_diff<0] = 0               # Asign zero to all negative entries after smooth substraction
    
    ### Calculate mean of y_diff
    eps_mean = np.sum(x * y_diff) / np.sum(y_diff)
    
    ### Calculate median of y_diff
    y_diff_cum = np.cumsum(y_diff / np.sum(y_diff))
    median_idx = np.argmin(np.abs(y_diff_cum-0.5))
    eps_median = x[median_idx]
    
    ### Which value is used for cut out of original distribution
    eps = (eps_median + eps_mean)/2
    
    ### Cut out first peak p based on eps
    in_first_peak = np.abs(p-eps) < 0.4 * eps
    eps = np.median(p[in_first_peak])
    
    return eps ,x, y, y2, y_diff



#%%
def props_other(df,NoFrames):
    '''
    Get other properties.
    '''
    
    ### Compute eps directly by smoothly subtracting doubled photon histogram from original photon histogram
    photons = df.photons_ck.values # Use Chung-Kennedy filtered photon values
    eps = extract_eps(photons)[0]
    
    s_out=pd.Series({'eps':eps,
                     'occ':len(df)/NoFrames,
                     'frame':np.mean(df.frame),
                     'std_frame':np.std(df.frame),
                     'photons': np.mean(photons), # Use Chung-Kennedy filtered photon values
                     'bg': np.mean(df.bg),
                     'x':np.mean(df.x),
                     'y':np.mean(df.y),
                     'lpx':np.std(df.x),
                     'lpy':np.std(df.y),
                     'sx':np.mean(df.sx),
                     'sy':np.mean(df.sy),
                     })
    return s_out