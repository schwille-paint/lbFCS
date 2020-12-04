import numpy as np 
import pandas as pd
import numba
import time
from tqdm import tqdm

### Test if GPU fitting can be used
try:
    from pygpufit import gpufit as gf
    gpufit_available=True
    gpufit_available=gf.cuda_available()
except ImportError:
    gpufit_available = False

import picasso.gausslq as gausslq
import picasso.postprocess as postprocess
import picasso_addon.io as addon_io
import picasso_addon.localize as localize
import picasso_addon.autopick as autopick

LOCS_DTYPE = [
    ('frame', 'u4'),
    ('x_in', 'f4'),
    ('y_in', 'f4'),
    ('x', 'f4'),
    ('y', 'f4'),
    ('imagers', 'f4'),
    ('photons', 'f4'),
    ('photons_ck', 'f4'),
    ('sx', 'f4'),
    ('sy', 'f4'),
    ('bg', 'f4'),
    ('lpx', 'f4'),
    ('lpy', 'f4'),
    ('net_gradient', 'f4'),
    ('group', 'u4'),
    ('M','u4'),
    ('exp','f4'),
    ('koff', 'f4'),
    ('kon', 'f4'),
    ('conc', 'f4'),
    ('N_in', 'u4'),
#    ('ellipticity', 'f4'),
#    ('n_id', "u4"),
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
def generate_locs(savepath,
                  reps,
                  M,
                  CycleTime,
                  N,
                  koff,
                  kon,
                  c,
                  box,
                  e_tot,
                  snr,
                  sigma = 0.9,
                  use_weight=True):
    
    ### Print statement
    print('%ix simulations of:'%reps)
    print('M:    %i'%M)
    print('exp:  %.2f [s]'%CycleTime)
    print('N:    %i'%N)
    print('koff: %.1e [1/s]'%koff)
    print('kon:  %.1e [1/Ms]'%kon)
    print('conc: %.1e [M]'%c)
    print('box: %i [ px]'%box)
    print('e_tot:  %i [e-]'%e_tot)
    print('snr: %.1f []'%snr)
    print()
    
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
        trace = generate_trace(M,CycleTime,N,koff,kon,c,10)
        
        ### Assign
        start=r*M
        end=(r+1)*M
        locs['frame'][start:end] = frames
        locs['x_in'][start:end]  = int(box/2) + np.random.rand(1) * (700 - box)
        locs['y_in'][start:end]  = int(box/2) + np.random.rand(1) * (700 - box)
        locs['imagers'][start:end] = trace
        locs['group'][start:end] = r 
    
    ### Assign input parameters
    locs['M']=M
    locs['exp']=CycleTime
    locs['N_in']=N
    locs['koff']=koff
    locs['kon']=kon
    locs['conc']=c
    
    ### Remove zeros in imager levels
    locs=locs[locs['imagers']>0]
    
    ### Copy input coordinates
    locs['x'] = locs['x_in']
    locs['y'] = locs['y_in']
    
    ### Add photon noise by spot creation and fitting
    locs, spots = noise_fitspots(locs, box, e_tot, snr, sigma, use_weight=True)
    
    ### Convert to DataFrame
    locs = pd.DataFrame(locs)
    
    ### Check for any negative values in photon, bg, sx, sy values
    positives = (locs.photons>0) & (locs.bg>0) & (locs.sx>0) & (locs.sy>0)
    locs = locs[positives]
    
    ### Apply photon step filter
    print('Applying Chung-Kennedy filter ...')
    tqdm.pandas()
    locs = locs.groupby('group').progress_apply(lambda df: df.assign( photons_ck = autopick.ck_nlf(df.photons.values).astype(np.float32) ) )
    
    ###
    print('Total time: %.2f [s]'%(time.time()-t0))
    print()
    
    ### Saving
    print('Saving ...')
    info=[{'Width':700,
            'Height':700,
            'reps':reps,
            'Frames':M,
            'exp':CycleTime,
            'N':N,
            'koff':koff,
            'kon':kon,
            'conc':c,
            'box':box,
            'e_tot':e_tot,
            'snr':snr,
            'sigma':sigma,
            'use_weight':use_weight,
            }]
    
    try:
        addon_io.save_locs(savepath,
                            locs,
                            info,
                            mode='picasso_compatible')
    except OSError:
        print('Could not save file to savepath!')
        
    return locs, info, spots 

#%%
def noise_fitspots(locs, box, e_tot, snr, sigma, use_weight):
    
    ### Generate spots
    spots = generate_spots(locs, box, e_tot, snr, sigma)
    ### Fit spots
    locs_noise = fit_spots(locs,box,spots[0],spots[1],use_weight)
    ### Compute and assing net-gradient
    print('Calculating net-gradient ...')
    locs_noise['net_gradient'] = netgradient_spots(spots[0])
    
    return locs_noise, spots

#%%
def gauss2D_on_mesh(p, xi, yi):
    arg = -(np.square(xi - p[:,1,np.newaxis,np.newaxis]) + np.square(yi - p[:,2,np.newaxis,np.newaxis]))
    arg /=  2 * p[:,3,np.newaxis,np.newaxis] * p[:,4,np.newaxis,np.newaxis]
    y = p[:,0,np.newaxis,np.newaxis] * np.exp(arg) + p[:,5,np.newaxis,np.newaxis]

    return y

#%%
def generate_spots(locs, box, e_tot, snr, sigma):
    
    print('Generating spots ... ')
    n_spots = len(locs)
    
    ### Generate coordinates for one spot
    px = np.arange(box)
    i,j = np.meshgrid(px, px, indexing='ij')
    
    ### Generate coordinates for all spots
    i_spots = np.repeat(i.reshape(1,box,box),n_spots,axis=0)
    j_spots = np.repeat(j.reshape(1,box,box),n_spots,axis=0)
    
    ### Generate true parameters for all spots
    p_spots = np.ones((n_spots,6),dtype = np.float32)
    p_spots[:,0] = e_tot * locs['imagers']              # Assign total e-
    p_spots[:,0] /= 2*np.pi*sigma**2                    # Convert to amplitude
    p_spots[:,1] = (box-1)/2 + locs['y_in'] - np.round(locs['y_in'])
    p_spots[:,2] = (box-1)/2 + locs['x_in'] - np.round(locs['x_in'])                     
    p_spots[:,3] = sigma                                       # Same width in x and y direction!
    p_spots[:,4] = sigma                                       
    p_spots[:,5] = e_tot / (2*np.pi*sigma**2*snr) # Background = amplitde/snr!
    
    ### Generate gauss2D spots without noise
    spots = gauss2D_on_mesh(p_spots, i_spots, j_spots)
    spots = spots.reshape(n_spots,box*box) # Flatten 
    
    ### Generate gauss2D spots with shot noise
    spots_shotnoise = np.random.poisson(spots)
    
    ### Generate spots readout noise 
    spots_readvar = localize.cut_spots_readvar(locs,box)
    spots_readvar = spots_readvar.reshape(n_spots,box*box)
    spots_readnoise = np.random.normal(0,np.sqrt(spots_readvar))
    
    ### Combine noise distributions
    spots_noise = spots_shotnoise + spots_readnoise
    spots_noise = spots_noise.astype(np.float32)
    
    ### Reshape spots 
    spots_noise = spots_noise.reshape(n_spots,box,box)
    spots_readvar = spots_readvar.reshape(n_spots,box,box)
    spots = spots.reshape(n_spots,box,box)
    spots_shotnoise = spots_shotnoise.reshape(n_spots,box,box)
    spots_shotnoise = spots_shotnoise - spots
    spots_readnoise = spots_readnoise.reshape(n_spots,box,box)
    
    return [spots_noise, spots_readvar, spots, spots_shotnoise, spots_readnoise]

#%%
def fit_spots(locs,box,spots,spots_readvar, use_weight):
    locs_out = locs.copy()
    init_shape = np.shape(spots)
    
    if gpufit_available:
            if use_weight:
                print('Weighted least square fitting (GPU) ...')
                theta = localize.weightfit_spots_gpufit(spots,spots_readvar)
                ### Reshape spots after fitting
                spots.shape = init_shape
                spots_readvar.shape = init_shape
            else:
                print('Non-weighted least square fitting (GPU) ...')
                theta = gausslq.fit_spots_gpufit(spots)
                ### Reshape spots after fitting
                spots.shape = init_shape
            
            ### Convert fit parameters to final output
            locs_out['photons'] = theta[:,0]
            locs_out['bg'] = theta[:,-1]
            locs_out['x'] = np.round(locs_out['x_in']) + theta[:,1] - int(box/2)
            locs_out['y'] = np.round(locs_out['y_in']) + theta[:,2] - int(box/2)
            locs_out['sx'] = theta[:,3]
            locs_out['sy'] = theta[:,4]
            locs_out['lpx'] = postprocess.localization_precision(theta[:, 0], theta[:, 3], theta[:, 5], em = False)
            locs_out['lpy'] = postprocess.localization_precision(theta[:, 0], theta[:, 4], theta[:, 5], em = False)
            
    else:
        print('Only GPU fitting implemented assigning imagers ...')
    
    return locs_out

#%%
@numba.jit(nopython=True, nogil=True, cache=False)
def netgradient_spots(spots):
    box = np.shape(spots)[1]
    box_half = int(box / 2)
    
    ### We create some kind of a meshgrid
    ux = np.zeros((box, box), dtype=np.float32)
    uy = np.zeros((box, box), dtype=np.float32)
    for i in range(box):
        val = box_half - i
        ux[:, i] = uy[i, :] = val
    unorm = np.sqrt(ux ** 2 + uy ** 2)
    ux /= unorm
    uy /= unorm
     
    ### Loop through spots and calculate net-gradient
    ng = np.zeros(len(spots), dtype=np.float32)
    for spot_idx, spot in enumerate(spots):
        for i in range(1,box-1):
            for j in range(1,box-1):
                if not (i == box_half and j == box_half):
                    gj = spot[j + 1, i] - spot[j - 1, i]
                    gi = spot[j, i + 1] - spot[j, i - 1]
                    ng[spot_idx] += (gj * uy[j,i] + gi * ux[j,i])
                    
    return ng