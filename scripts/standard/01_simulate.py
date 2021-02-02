import os
import numpy as np

import lbfcs
import lbfcs.simulate as simulate

### Define system constants
reps = 1000
M = [4500]*3 + [9000]*2
CycleTime = 0.4

N = 3
koff = 1.15e-1
kon = 30e6
cs = [5000e-12, 2500e-12, 1250e-12, 625e-12, 313e-12]

box = 7
e_tot = 400
sigma = 0.9
use_weight = False

savedir = '/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-01-22_Simulation/5xCTC_exp400_T23_kon30e6_new/N03'
#%%
for i,c in enumerate(cs):
    ### Path and naming
    N_str=('%i'%(N)).zfill(2)
    c_str=('%i'%(c*1e12)).zfill(4)
    
    savename='N%s_c%s.hdf5'%(N_str,c_str)
    savepath=os.path.join(savedir,savename)
    
    ### Generate simulation
    locs = simulate.generate_locs(savepath,
                                  reps,
                                  M[i],
                                  CycleTime,
                                  N,
                                  koff,
                                  kon,
                                  c,
                                  box,
                                  e_tot,
                                  lbfcs.snr_from_conc(c),
                                  sigma,
                                  use_weight)
