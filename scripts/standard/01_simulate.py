import os
import numpy as np

import lbfcs
import lbfcs.simulate as simulate

### Define system constants
reps = 1000
M = [9000]*1
CycleTime = 0.2

N = 1
koff = 2.66e-1
kon = 6.5e6
cs = [5000e-12]

box = 7
e_tot = 350
sigma = 0.9
use_weight = False

savedir = r'C:\Data\p17.lbFCS2\21-07-13_sim_Pm2_exp200'

#%%
for i,c in enumerate(cs):
    ### Path and naming
    N_str=('%i'%(N)).zfill(2)
    c_str=('%i'%(c*1e12)).zfill(4)
    
    savename='N%s_c%s_picked.hdf5'%(N_str,c_str)
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
