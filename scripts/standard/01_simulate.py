import os
import numpy as np

import lbfcs
import lbfcs.simulate as simulate

### Define system constants
reps = 3000
M = [4500]*3
CycleTime = 0.4

N = 1
koff = [1.10e-1, 1.12e-1, 1.16e-1]
kon = [2.55e7, 3.27e7, 3.10e7]
cs = [10000e-12, 5000e-12, 2500e-12]

box = 7
e_tot = [564,570,500]
snr = [3.5,5.1,8.4]
sigma = 0.85
use_weight = False

savedir = '/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-01-22_Simulation/N1_5xCTC_exact20201217'
#%%
for i,c in enumerate(cs):
    ### Path and naming
    N_str=('%i'%(N)).zfill(2)
    c_str=('%i'%(c*1e12)).zfill(5)
    
    savename='N%s_c%s.hdf5'%(N_str,c_str)
    savepath=os.path.join(savedir,savename)
    
    ### Generate simulation
    locs = simulate.generate_locs(savepath,
                                  reps,
                                  M[i],
                                  CycleTime,
                                  N,
                                  koff[i],
                                  kon[i],
                                  c,
                                  box,
                                  e_tot[i],
                                  snr[i], #lbfcs.snr_from_conc(c),
                                  sigma,
                                  use_weight)
