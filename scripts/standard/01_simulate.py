import importlib
import os
import numpy as np

import lbfcs.simulate as simulate
importlib.reload(simulate)

### Define system constants
reps=4000
exp=0.2
koff=0.28
kon=6.5e6

Ms=[1000,2000,3000,4500,6000,7500,9000,18000]
Ns=[12]
cs=[i*1e-9 for i in [5,10,20]]



savedir=r'C:\Data\p04.lb-FCS\20-06-22_Simulation\scan_run01\N03'

#%%
for M in Ms:
    for N in Ns:
        for c in cs:
            
            ### Path and naming
            M_str=('%i'%(M)).zfill(5)
            N_str=('%i'%(N)).zfill(2)
            c_str=('%i'%(c*1e9)).zfill(2)
            
            savename='M%s_N%s_c%s.hdf5'%(M_str,N_str,c_str)
            savepath=os.path.join(savedir,savename)
            
            ### Generate simulation
            locs = simulate.generate_locs(savepath,reps,M,exp,N,koff,kon,c)