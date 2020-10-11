import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import glob
import importlib
import re
import lbfcs.solveseries as solve

#%%
def convert_sol(sol,exp):
    
    cols = list(sol.columns)
    cols_konc = [c for c in cols if bool(re.search('konc',c))]
    cols_konc = [c for c in cols_konc if not bool(re.search('_std',c))]
    
    koff  = float(sol.koff/exp)
    koncs = sol[cols_konc].values/exp
    cs    = np.array([int(c[4:]) for c in cols_konc])*1e-12
    kons  = (koncs/cs).flatten()
    kon   = np.mean(kons)
    N     = float(sol.N)
    
    return koff,kon,N,kons,cs
    
#%%
def compare_old(obs,sol,exp):
    
    x = obs.vary.values
    xlim = [0,max(x)+0.2*max(x)]
    x_ref = np.linspace(xlim[0]+0.1,xlim[1],100)
    
    koff,kon,N = convert_sol(sol,exp)[:-2]
    c          = x_ref*1e-12
    
    tau     = solve.tau_func(koff,kon*c,0)
    Ainv    = 1/solve.A_func(koff,kon*c,N,0)
    taudinv = 1/solve.taud_func(kon*c,N,0)
    
    def plotter(x,field,convert,ref):
        y = obs[field]*convert
        if field != 'tau': y = 1/y
        
        ylim = [min(y)-0.3*min(y),max(y)+0.3*max(y)]
        if field != 'tau': ylim[0] = 0
        ax.plot(x,
                y,
                'o',
                mfc='r',
                alpha=0.3)
        ax.plot(x_ref,
                ref,
                '-',
                c='r',
                lw=2)
        
        ax.set_xlabel('Concentration [pM]')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    
        
    f=plt.figure(num=11,figsize=[4,9])
    f.subplots_adjust(bottom=0.08,top=0.98,left=0.25,right=0.95,hspace=0.25)
    f.clear()
    
    ax = f.add_subplot(311)
    plotter(x,'tau',exp,tau)
    ax.set_ylabel(r'$\tau$  (s)')
    
    ax = f.add_subplot(312)
    plotter(x,'A',1,Ainv)
    ax.set_ylabel('1/A ()')
    
    ax = f.add_subplot(313)
    plotter(x,'taud',exp,taudinv)
    ax.set_ylabel(r'$1/\tau_{d}$ (Hz)')
    
#%%
def print_sol(sol,exp):
    koff,kon,N,kons,cs = convert_sol(sol,exp)
    print('koff = %.2e [1/s]'%koff)
    for i,c in enumerate(cs): print('kon  =  %.1e [1/Ms]@ %i'%(kons[i],int(c*1e12)))
    print('N    = %.2f'%N)
    print()