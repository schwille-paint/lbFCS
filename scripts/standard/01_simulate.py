import importlib
import os
import numpy as np

import lbfcs.simulate as simulate
importlib.reload(simulate)

### Define system constants
reps = 1000
M = 9000
CycleTime = 0.2

N = 10
koff = 0.15
kon = 6e6
# cs = [5e-9,10e-9,20e-9]
cs = [2e-9]

box = 9
e_tot = 250
snr = 3
sigma = 0.9
use_weight = True

savedir=r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-03_SimulationsNoise/exp200_koff15_kon6/e250_snr3_box9'
#%%
for c in cs:
    ### Path and naming
    N_str=('%i'%(N)).zfill(2)
    c_str=('%i'%(c*1e12)).zfill(4)
    
    savename='N%s_c%s_picked.hdf5'%(N_str,c_str)
    savepath=os.path.join(savedir,savename)
    
    ### Generate simulation
    locs = simulate.generate_locs(savepath,
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
                                  sigma,
                                  use_weight)
