# Import modules
import numpy as np
import pandas as pd
import importlib


#%%
def _stats(df,CycleTime):
    """ 
    Parameters
    ---------
    df : pandas.DataFrame
        Stacked '_props' DataFrame using pandas.concat([props for props in props],labels=labels,key='expID')
    Returns
    -------
    df_stats: pandas.DataFrame
        df will be grouped by expID and statistical quantities are calulated and returned as multi-index DataFrame
        
    """
    #### List of observables
    out_vars=['expID','mono_tau','mono_tau_lin','mono_A','mono_A_lin','tau_b','tau_d','mono_taub','mono_taud','n_events','n_locs']
    #### List of statistical quantities
    out_stats=['25%','50%','75%','mean','std']
    #### List of observables that will be multiplied with CycleTime to convert from frame to seconds
    out_converts=['mono_tau','mono_tau_lin','mono_taub','mono_taud','tau_b','tau_d']
    
    #### Get statistics defined in out_stats in all variables defined in out_vars
    df_stats=df.groupby('expID').describe()
    df_stats=df_stats.loc[:,(out_vars,out_stats)]
    #### Apply unit conversion from frame to [s] to all tau values
    df_stats.loc[:,(out_converts,slice(None))]=df_stats.loc[:,(out_converts,slice(None))]*CycleTime
    #### Get deviation from percentiles to median
    df_stats.loc[:,(slice(None),'25%')]=df_stats.loc[:,(slice(None),'50%')].values-df_stats.loc[:,(slice(None),'25%')].values
    df_stats.loc[:,(slice(None),'75%')]=df_stats.loc[:,(slice(None),'75%')].values-df_stats.loc[:,(slice(None),'50%')].values
    df_stats.loc[:,(slice(None),'std')]=df_stats.loc[:,(slice(None),'std')].values/2
    ##### Add number of groups used for generation of stats
    df_stats.loc[:,'groups']=df.groupby('expID').size()
    ##### Add imager concentration
    df_stats.loc[:,'c']=df.groupby('expID').apply(lambda df: df.conc.mean()).values
    df_stats.rename(columns={'c':'conc'},inplace=True)
    
    return df_stats

#%%
def _fit(df,df_stats):
        
    from scipy.optimize import curve_fit
    import lbfcs.varfuncs as varfuncs
    importlib.reload(varfuncs)
    
    #### Get ariables for start paramter estimates
    minc=df_stats.conc.min()
    maxc=df_stats.conc.max()
    tau_minc=df_stats.loc[df_stats.conc==minc,('mono_tau','mean')].mean()
    tau_maxc=df_stats.loc[df_stats.conc==maxc,('mono_tau','mean')].mean()
    minc=minc*1e-9 # Convert: [nM]->[M]
    maxc=maxc*1e-9 # Convert: [nM]->[M]
    
    koff=1/tau_minc
    kon=-koff**2*((tau_minc-tau_maxc)/(minc-maxc))
    
    #### mono_tau vs c (lbFCS)
    p0=[koff,kon]
    popt_lbfcs,pcov_lbfcs=curve_fit(varfuncs.tau_of_c,
                             df_stats.conc*1e-9,
                             df_stats.loc[:,('mono_tau','mean')],
                             p0=p0,
                             sigma=None,
                             absolute_sigma=False)
    #### 1/mono_A vs c
    p0=[kon/koff,1e-3]
    popt_lbfcsA,pcov_lbfcsA=curve_fit(varfuncs.Ainv_of_c,
                          df_stats.conc*1e-9,
                          1/df_stats.loc[:,('mono_A','50%')],
                          p0=p0,
                          sigma=None,
                          absolute_sigma=True)
    
    ##### tau_d vs c (Picasso)
    p0=[kon,1e-3]
    try:
        popt_qpaint,pcov_qpaint=curve_fit(varfuncs.taudinv_of_c,
                          df_stats.conc*1e-9,
                          1/df_stats.loc[:,('tau_d','50%')],
                          p0=p0,
                          sigma=None,
                          absolute_sigma=False)
    except ValueError: 
        popt_qpaint=[0,0]
        
    #### Assign fit results to df_fit
    df_fit=pd.DataFrame([],index=['lbfcs','lbfcsA','qpaint'])
    df_fit=df_fit.assign(popt0=[popt_lbfcs[0],popt_lbfcsA[0],popt_qpaint[0]])
    df_fit=df_fit.assign(popt1=[popt_lbfcs[1],popt_lbfcsA[1],popt_qpaint[1]])
        
    #### Assign number of docking sites to df
    koff=df_fit.loc['lbfcs','popt0'] # [Hz]
    kon=df_fit.loc['lbfcs','popt1'] # [Hz/M]
    
    #### Assign number of docking sites to df_stats
    for expID in df_stats.index:
        c=df_stats.loc[expID,('conc','')]*1e-9 # Concentration [M]
        df.loc[expID,'N']=(koff/(kon*c))*(1./df.loc[expID,'mono_A'].values)
        df_stats.loc[expID,('N','50%')]=df.loc[expID,'N'].median()
        df_stats.loc[expID,('N','25%')]=np.percentile(df.loc[expID,'N'],25)
        df_stats.loc[expID,('N','75%')]=np.percentile(df.loc[expID,'N'],75)
        
    return df,df_stats,df_fit

#%%
def _plot(df_stats,df_fit):
       
    import matplotlib.pyplot as plt
    
    f=plt.figure(num=11,figsize=[4,9])
    f.subplots_adjust(bottom=0.08,top=0.98,left=0.25,right=0.95,hspace=0.25)
    f.clear()
    
    ################################################ mono_tau vs concentration
    ax1=f.add_subplot(311)
    ax1=_mono_tau_onax(ax1,df_stats,df_fit)
    
    ################################################ 1/mono_A vs concentration
    ax2=f.add_subplot(312)
    ax2=_mono_A_onax(ax2,df_stats,df_fit)
      
    ###############################################################  1/fit_taud vs conc
    ax3=f.add_subplot(313)
    ax3=_tau_d_onax(ax3,df_stats,df_fit,color='b')
    
    return [ax1,ax2,ax3]
    
#%%
def _prep_mono_tau(df_stats,df_fit):
    import lbfcs.varfuncs as varfuncs
    
    field='mono_tau'
    
    x=df_stats.conc
    y=df_stats.loc[:,(field,'mean')]
    yerr=df_stats.loc[:,(field,'std')]
    
    popt=df_fit.loc['lbfcs',:]
    xfit=np.arange(0,max(x)+50,0.01)
    yfit=varfuncs.tau_of_c(xfit*1e-9,*popt)
    
    return x,y,yerr,xfit,yfit
    
#%% 
def _mono_tau_onax(ax,df_stats,df_fit,color='red',label_data='data',label_fit='fit'):
    import lbfcs.pyplot_wrap as plt_wrap
    
    #### Get data to plot
    x,y,yerr,xfit,yfit=_prep_mono_tau(df_stats,df_fit)
    
    #### Plot data
    ax.errorbar(x,y,yerr=yerr,fmt='o',label=label_data,c=color)
    ax.plot(xfit,yfit,'-',label=label_fit,c=color,lw=2)
    
    plt_wrap.ax_styler(ax)
    ax.set_xlim(0,max(x)+0.2*max(x))
    ax.set_ylim(min(y)-0.3*min(y),max(y)+0.3*max(y))
    ax.set_xlabel('Concentration (nM)')
    ax.set_ylabel(r'$\langle\tau\rangle$ (s)')
    ax.legend(loc='upper right')
    
    return ax
       
#%%
def _prep_mono_A(df_stats,df_fit):
    import lbfcs.varfuncs as varfuncs
    
    field='mono_A'
    
    x=df_stats.conc
    y=1/df_stats.loc[:,(field,'50%')]
    yerr=[df_stats.loc[:,(field,'25%')]/df_stats.loc[:,(field,'50%')]**2,
          df_stats.loc[:,(field,'75%')]/df_stats.loc[:,(field,'50%')]**2]
    
    popt=df_fit.loc['lbfcsA',:]
    xfit=np.arange(0,max(x)+50,0.01)
    yfit=varfuncs.Ainv_of_c(xfit*1e-9,*popt)
    
    return x,y,yerr,xfit,yfit
 
#%%
def _mono_A_onax(ax,df_stats,df_fit,color='red',label_data='data',label_fit='fit'):
    import lbfcs.pyplot_wrap as plt_wrap
    
    #### Get data to plot
    x,y,yerr,xfit,yfit=_prep_mono_A(df_stats,df_fit)
    
    #### Plot data
    ax.errorbar(x,y,yerr=yerr,fmt='o',label=label_data,c=color)
    ax.plot(xfit,yfit,'-',label=label_fit,c=color,lw=2)
    
    plt_wrap.ax_styler(ax)
    ax.set_xlim(0,max(x)+0.2*max(x))
    ax.set_ylim(0,max(y)+0.3*max(y))
    ax.set_xlabel('Concentration (nM)')
    ax.set_ylabel(r'$1/A$ (a.u.)')
    ax.legend(loc='upper left')
    
    return ax
    
#%%
def _prep_tau_d(df_stats,df_fit):
    import lbfcs.varfuncs as varfuncs
    
    field='tau_d'
    
    x=df_stats.conc
    y=1/df_stats.loc[:,(field,'50%')]
    yerr=[df_stats.loc[:,(field,'25%')]/df_stats.loc[:,(field,'50%')]**2,
          df_stats.loc[:,(field,'75%')]/df_stats.loc[:,(field,'50%')]**2]
    
    popt=df_fit.loc['qpaint',:]
    xfit=np.arange(0,max(x)+50,0.01)
    yfit=varfuncs.taudinv_of_c(xfit*1e-9,*popt)
    
    return x,y,yerr,xfit,yfit
 
#%%
def _tau_d_onax(ax,df_stats,df_fit,color='red',label_data='data',label_fit='fit'):
    import lbfcs.pyplot_wrap as plt_wrap
    
    #### Get data to plot
    x,y,yerr,xfit,yfit=_prep_tau_d(df_stats,df_fit)
    
    #### Plot data
    ax.errorbar(x,y,yerr=yerr,fmt='o',label=label_data,c=color)
    ax.plot(xfit,yfit,'-',label=label_fit,c=color,lw=2)
    
    plt_wrap.ax_styler(ax)
    ax.set_xlim(0,max(x)+0.2*max(x))
    ax.set_ylim(0,max(y)+0.3*max(y))
    ax.set_xlabel('Concentration (nM)')
    ax.set_ylabel(r'$1/\tau_d$ (Hz)')
    ax.legend(loc='upper left')
    
    return ax
