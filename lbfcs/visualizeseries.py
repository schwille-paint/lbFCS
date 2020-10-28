import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import glob
import importlib
import re
import traceback

import lbfcs.solveseries as solve

#%%
def create_savesubpath(save_path,do_save,obs):
    
    if not os.path.exists(save_path): 
        print('Not a valid save_path!');print()
        do_save = False
        return '', do_save
    
    else:
        setting_name = 'N' + ( '%i'%( np.unique(obs.setting)[0] )).zfill(2) + '_'
        vary_names   = [( '%i'%(vary) ).zfill(5) for vary in np.unique(obs.vary)]
        
        save_subdir  = setting_name
        for name in vary_names: save_subdir += name + '-'
        save_subdir = save_subdir[:-1]
        
        save_subpath = os.path.join(save_path,save_subdir)
        
        try: os.mkdir(save_subpath)
        except FileExistsError: print('Subdir already exists');print() 
    
        return save_subpath, do_save
    
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
    
    x = obs.vary.values                             # [pM]
    xlim = np.array([0,max(x)+0.2*max(x)])
    x_ref = np.linspace(xlim[0]+0.1,xlim[1],100)
    
    koff,kon,N = convert_sol(sol,exp)[:-2]
    c          = x_ref*1e-12                        # [M]
    
    tau     = solve.tau_func(koff,kon*c,0)
    Ainv    = 1/solve.A_func(koff,kon*c,N,0)
    occ     = solve.occ_func(koff*(exp),kon*(exp)*c,N,0)
    p01     = solve.pk_func(0,koff*(exp),kon*(exp)*c,N,0)
        
    def plotter(x,field,convert,ref):
        y = obs[field]*convert
        if field == 'A': y = 1/y
        
        ylim = [min(y)-0.3*min(y),max(y)+0.3*max(y)]
        if field != 'tau': ylim[0] = 0
        ax.plot(x*1e-3,                             # [nM]
                y,
                'o',
                ms=8,
                mfc='r',
                alpha=0.7)
        ax.plot(x_ref*1e-3,                         # [nM]
                ref,
                '-',
                c='k',
                lw=1)
        
        ax.set_xlabel('Concentration [nM]')
        ax.set_xlim(xlim*1e-3)                      # [nM]
        ax.set_ylim(ylim)
    
        
    f=plt.figure(num=11,figsize=[8,6])
    f.subplots_adjust(bottom=0.1,top=0.9,left=0.1,right=0.95,hspace=0.25,wspace=0.4)
    f.clear()
    
    f.suptitle(r'$k_{off}$ = ' +
               '%.2e [1/s];     '%koff +
               r'$k_{on}$ = ' +
               '%.1e [1/Ms];     '%kon +
               r'$N$ = ' +
               '%.2f    '%N 
               )
    
    ax = f.add_subplot(221)
    plotter(x,'tau',exp,tau)
    ax.set_ylabel(r'$\tau$  [s]')
    
    ax = f.add_subplot(223)
    plotter(x,'A',1,Ainv)
    ax.set_ylabel('1/A []')
    
    ax = f.add_subplot(222)
    plotter(x,'occ',1,occ)
    ax.set_ylabel(r'occ []')
    
    ax = f.add_subplot(224)
    plotter(x,'p01',1,p01)
    ax.set_ylabel(r'p01 []')

#%%
def obs_relresidual(obs,obs_ref,exclude = 'taud|events'):
    
    ### Prep data
    cols = list(obs.columns)
    cols = [col for col in cols if not bool(re.search('_std',col))]
    cols = [col for col in cols if not bool(re.search('setting|rep|M|ignore',col))]
    cols = [col for col in cols if not bool(re.search(exclude,col))]
    

    y     = obs.loc[:,cols].values                            # Observables
    y0    = obs_ref.loc[:,cols].values                        # Reference observables
    delta = (y[:,1:]-y0[:,1:]) * (100/y0[:,1:])               # Relative residual
    conc  = np.unique(y[:,0]).flatten()  
    
    for i in range(np.shape(delta)[0]): 
        delta[i,-10:][ y[i,-10:] < 0.1* np.max(y[i,-10:]) ] = np.inf  # Remove Pk values < 5% of maximum Pk since they are not considered in fit
    
    last_col   = np.sum(np.any(np.isfinite(delta),axis=0))+1  # Up to which column do we expect valid data?
    cols_valid = cols[1:last_col]
    
    ## Plotting specs
    colors = plt.get_cmap('magma')
    colors = [colors(i) for i in np.linspace(0.1,0.9,5)]

    f=plt.figure(12,figsize=[6,3])
    f.subplots_adjust(bottom=0.3,top=0.85,left=0.15,right=0.95,hspace=0)
    f.clear()
    ax=f.add_subplot(111)
    
    for i,c in enumerate(conc):
        d = delta[y[:,0]==c]   # Select concentrations
        d = np.mean(d,axis=0)     # Average over concentrations       
  
        ax.plot(range(len(d)),
                d,
                '-o',
                c=colors[i],
                mec=colors[i],
                mfc='none',
                label=int(c),
                )
    
    ax.axhline(0,lw=1,ls='--',c='k')
    
    ax.legend(loc='upper left', bbox_to_anchor = (0.08,1.2), ncol=4)
    ax.set_xticks(range(len(cols_valid)))
    ax.set_xticklabels(cols_valid,rotation=60)
    ax.set_ylabel(r'$\Delta_{rel}$ [%]')
    ax.set_ylim(-15,15)
    
    return y,y0,delta

#%%
def show_levels(levels):
    
    df_group = levels.groupby(['setting','vary','rep'])
    groups = list(df_group.groups)
    
    f=plt.figure(13,figsize=[8,3])
    f.subplots_adjust(bottom=0.2,top=0.9,left=0.1,right=0.95,hspace=0,wspace=0)
    f.clear()
    
    for i,g in enumerate(groups):
        xdata,ydata,p =solve.fit_levels(df_group.get_group(g))[:-1]
        yfit = solve.gauss_comb(xdata,p)
        ax=f.add_subplot(1,4,i+1)
        ax.fill_between(xdata,
                        np.zeros(len(xdata)),
                        ydata,
                        fc='grey',
                        ec='none',
                        alpha=0.8,
                        )
        ax.plot(xdata,
                yfit,
                )
        
        ax.set_title(int(g[1]),fontsize=11)
        ax.set_xlim(0.2,5.8)
        ax.set_xticks([1,2,3,4,5])
        
        ax.set_ylim(0.4,1e2)
        ax.set_yscale('log')
        if i+1 in [1,6]: ax.set_yticks([1,10,100]);ax.set_ylabel(r'[%]')
        else: ax.set_yticks([])
        
#%%
def print_sol(sol,exp):
    koff,kon,N,kons,cs = convert_sol(sol,exp)
    print('koff = %.2e [1/s]'%koff)
    for i,c in enumerate(cs): print('kon  =  %.1e [1/Ms]@ c=%ipM'%(kons[i],int(c*1e12)))
    print('N    = %.2f'%N)
    print()