import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import glob
import importlib
import re
import time

import picasso.io as io
import lbfcs.solveseries as solve
import lbfcs.visualizeseries as visualize

plt.style.use('~/lbFCS/styles/paper.mplstyle')

#%%
############################### Define data
# dir_name=r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-17_FS_id192'
dir_name=r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-17_FS_id180'

paths = sorted( glob.glob(os.path.join(dir_name,'*_props.hdf5')) )
props_init = pd.concat([pd.DataFrame(io.load_locs(p)[0]) for p in paths],keys=range(len(paths)),names=['rep'])
props_init = props_init.reset_index(level=['rep'])

files_props = [os.path.split(path)[-1] for path in paths]

paths = sorted( glob.glob(os.path.join(dir_name,'*.hdf5')) )
paths = [path for path in paths if not bool(re.search('props',path))]
locs_init = pd.concat([pd.DataFrame(io.load_locs(p)[0]) for p in paths],keys=range(len(paths)),names=['rep'])
locs_init = locs_init.reset_index(level=['rep'])  

files_locs = [os.path.split(path)[-1] for path in paths]
#%%
############################################
'''
Part 2: Find solutions
'''
############################################

################ Set parameters
params = {'field' : 'vary',
                   'select' : [10000],
                   'parts' : 84,
                   'samples' : 1,
                   'weights' : [1,1,1,0,0,1],
                   'exp' : 0.4,
                   'intensity':'raw',
                   }

################ Filter props&locs @(setting,vary,rep)
props, locs = solve.prefilter(props_init, locs_init,
                                             params['field'], 
                                             params['select'],
                                             )

################ Bootstrap props (->part) and find 1st iteration solution @(setting,part) 
props_part, obs_part, sol_part,sol = solve.bootstrap_firstsolve(props,
                                                                                                   params['exp'],
                                                                                                   parts = params['parts'] ,
                                                                                                   samples = params['samples'] ,
                                                                                                   weights = params['weights'],
                                                                                                   )


################ Get bound imager probabilities and use for 2nd iteration solution @(setting,part)
props_part, levels_part, obs_part, sol_part_levels, sol_levels = solve.levels_secondsolve(props_part,
                                                                                                                                           sol,
                                                                                                                                           locs,
                                                                                                                                           obs_part,
                                                                                                                                           params['exp'],
                                                                                                                                           field = params['intensity'],
                                                                                                                                           weights = params['weights'],
                                                                                                                                           )


################ Mean observables @(setting,vary,rep,:) & expected observables @ solutions
obs, levels, obs_expect = solve.combine(props_part,
                                                                 obs_part,
                                                                 levels_part,
                                                                 [sol,sol_levels],
                                                                 )


#%%
############################### Assign N to props
props = props.groupby(['setting','vary']).apply(lambda df: solve.assign_N(df,sol_levels))
props = props.droplevel(['setting','vary'])


#%%
############################################
'''
Part 3: Visualize results
'''
############################################
do_save      = False
save_path    = r'/fs/pool/pool-schwille-paint/Analysis/p17.lbFCS2/cseries/experiments/plots/N01/20-12-09_5xCTC_ibidi'

# save_subpath, do_save = visualize.create_savesubpath(save_path,do_save,obs)

######################################## Plot subset and field histogram
field  = 'N'
subset = sol_part.setting == 1
bins   = np.linspace(0,5,40)
# bins = 'fd'

f=plt.figure(1,figsize=[4,3])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
f.clear()
ax = f.add_subplot(111) 
ax.hist(sol_part.loc[subset,field],bins=bins,fc='grey',ec='k')
ax.set_xlabel(field)
ax.set_ylabel('Counts')

# if do_save: plt.savefig(os.path.join(save_subpath,'N.pdf'),transparent=True)

#%%

# ########################################### Standardized results visualization
# Compare with old procedure
visualize.compare_old(obs,sol,params['exp'])
# if do_save: plt.savefig(os.path.join(save_subpath,'obs.pdf'),transparent=True)

# Observables residual from solution
ax = visualize.obs_relresidual(obs,obs_expect[0])
ax.set_ylim(-20,20)
# if do_save: plt.savefig(os.path.join(save_subpath,'res.pdf'),transparent=True)

# Levels
visualize.show_levels(levels,logscale=True)
# if do_save: plt.savefig(os.path.join(save_subpath,'levels.pdf'),transparent=True)

