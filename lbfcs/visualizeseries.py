import os 
import numpy as np
import matplotlib.pyplot as plt
import re

import lbfcs.solveseries as solve

OBS_COLS = ['tau','A','occ','I','var_I','B','taud','events','eps_direct']
OBS_COLS_LABELS = [r'$\tau$','A','occ','I',r'$I_{var}$','B',r'$\tau_d$','events',r'$\epsilon_{direct}$']

#%%
def residual_violinplot_toax(ax,obs_res,show_violin=False,show_cols=[0,1,2,3,8]):
    
    ### Remove non-finite entries
    arefinite = np.all(np.isfinite(obs_res[['tau','A','occ','I','eps_direct']]),axis=1)
    obs_res = obs_res[arefinite]
    
    ### Select columns to use
    cols = [OBS_COLS[i] for i in show_cols]
    cols_labels = [OBS_COLS_LABELS[i] for i in show_cols]
    
    if show_violin:
        ### Violin plot
        parts = ax.violinplot(obs_res[cols],
                      showmeans = False,
                      showextrema = False,
                      points = 20,
                      widths = 0.7,
                      )
        for pc in parts['bodies']:
            pc.set_facecolor('lightgrey')
            pc.set_edgecolor('black')
            pc.set_linewidth(1)
            pc.set_alpha(1)
    
    ### Zero line as guide for the eye
    ax.axhline(0,ls='--',c='k')
    
    ### Centers and quantile bars as errorbar on top of violin
    means = np.mean(obs_res[cols], axis=0)
    quartile1, medians, quartile3 = np.percentile(obs_res[cols], [10, 50, 90], axis=0)
    centers = medians.copy()                      # Center corresponds to median ...
    centers[0] = means[0]                           # ... except for tau
    quartile1 = np.abs(quartile1 - centers)
    quartile3 = np.abs(quartile3 - centers)
    
    ax.errorbar(range(1,len(cols)+1),
                centers,
                yerr=(quartile1,quartile3),
                fmt='o',mfc='w',mec='k',c='r',lw=1.5,
                )

    ax.set_xticks(range(1,len(cols)+1))
    ax.set_xticklabels(cols_labels,rotation=70)
    ax.set_ylabel(r'$\Delta_{rel}$ [%]')
    ax.set_ylim([-50,50])

