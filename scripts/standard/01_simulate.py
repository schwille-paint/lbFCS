import importlib
import os
import numpy as np

import lbfcs.simulate as simulate
importlib.reload(simulate)

### Define system constants
reps=1000
exp=0.2
kon=6.5e6
c=4000e-12
M=9000
koff=0.28

Ns=[1,2,3,4,8,12,20]

savedir=r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-10-09_Simulation-Ringberg/cseries_koff28_kon06'

#%%
for N in Ns:
    ### Path and naming
    N_str=('%i'%(N)).zfill(2)
    koff_str='%i'%(koff*1e2)
    konc_str=('%i'%(kon*c*1e3)).zfill(3)
    
    
    savename='N%s_koff%s_konc%s.hdf5'%(N_str,koff_str,konc_str)
    savepath=os.path.join(savedir,savename)
    
    ### Generate simulation
    locs = simulate.generate_locs(savepath,reps,M,exp,N,koff,kon,c)