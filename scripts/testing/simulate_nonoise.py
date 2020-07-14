import matplotlib.pyplot as plt
import numpy as np 
import importlib
import os

import lbfcs.simulate as simulate
import lbfcs.pickprops as pickprops
importlib.reload(simulate)

### Define system
reps=1000
M=18000
CycleTime=0.05
N=3
koff=0.15
kon=6.5e6*2
c=10e-9

savedir=r'C:\Data\p04.lb-FCS\20-06-22_Simulation\tests'
savename='exp%i_N%i_c%i.hdf5'%(CycleTime*1e3,N,c*1e12)
savepath=os.path.join(savedir,savename)

### Simulate single traces and convert to _picked
locs = simulate.generate_locs(savepath,reps,M,CycleTime,N,koff,kon,c)


#%%
### Get props
props=pickprops.apply_props(locs,c,M,ignore=1,mode='ralf')

######################### Plotting
g=0
glocs=locs[locs.group==g]
trace,ac=pickprops.trace_and_ac(glocs,M)
    
ACs=np.zeros((reps,len(ac)))
for g in range(1,reps):
    glocs=locs[locs.group==g]
    ACs[g,:]=pickprops.trace_and_ac(glocs,M)[1][:,1]

#%%
######################### Trace & AC
f=plt.figure(num=1,figsize=[8,6])
f.subplots_adjust(bottom=0.1,top=0.95,left=0.15,right=0.95)
f.clear()
ax=f.add_subplot(311)
ax.errorbar(ac[:,0],
           ACs.mean(axis=0),
           yerr=ACs.std(axis=0)
           )
ax.set_xscale('log')
ax.set_xlim(0.9,3e2)

ax=f.add_subplot(312)
ax.plot(ac[:,0],
        (np.std(ACs,axis=0)/np.nanmean(ACs,axis=0))*100,
        )
ax.set_xscale('log')
ax.set_xlim(0.9,3e2)

ax=f.add_subplot(313)
ax.plot(trace)

######################### tau
f=plt.figure(num=2,figsize=[6,4])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
f.clear()
ax=f.add_subplot(111)
ax.hist(props.tau_lin*CycleTime,
        bins='fd',
        fc='grey',
        ec='k')

ax.axvline(1/(koff+kon*c),
           lw=3,
           ls='-',
           c='r',
           label='in')

ax.axvline(props.tau_lin.mean()*CycleTime,
           lw=3,
           ls='--',
           c='k',
           label='out')

ax.legend()
ax.set_xlabel(r'Correlation time $\tau_{lin}$ [s]')
ax.set_ylabel('Counts')

######################### A
f=plt.figure(num=3,figsize=[6,4])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
f.clear()
ax=f.add_subplot(111)
ax.hist(props.A_lin,
        bins='fd',
        fc='grey',
        ec='k')

ax.axvline(koff/(N*kon*c),
           lw=3,
           ls='-',
           c='r',
           label='in')

ax.axvline(props.A_lin.median(),
           lw=3,
           ls='--',
           c='k',
           label='out')

ax.legend()
ax.set_xlabel(r'Correlation amplitude $A_{lin}$ []')
ax.set_ylabel('Counts')

######################### B
f=plt.figure(num=4,figsize=[6,4])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
f.clear()
ax=f.add_subplot(111)

koff_out=props.B.mean()/(props.tau_lin.mean()*CycleTime)
konc_out=1/(props.tau_lin.mean()*CycleTime)-koff_out
N_out=koff_out/(props.A_lin*konc_out)

ax.hist(N_out,
        bins=np.linspace(0,5,50),
        fc='grey',
        ec='k')

ax.axvline(3,#/(koff+kon*c),
            lw=3,
            ls='-',
            c='r',
            label='in')

# ax.axvline(props.B.mean(),
#            lw=3,
#            ls='--',
#            c='k',
#            label='out')

ax.legend()
ax.set_xlabel(r'Brightness $B$ []')
ax.set_ylabel('Counts')

