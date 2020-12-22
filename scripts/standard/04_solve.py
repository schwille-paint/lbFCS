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
# dir_name=r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-17_N1-2x5xCTC_cseries/20-12-17_FS_id180'
dir_name = '/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-18_N2-5xCTC_cseries/20-12-18_FS_id194'

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
params = {'in_field' : 'vary',
                   'select_values' : [2500],
                   'parts' :1000,
                   'samples' : 1,
                   'weights' : [1,1,1,0,0,1],
                   'exp' : 0.4,
                   'intensity':'raw',
                   }

################ Filter props&locs @(setting,vary,rep) and set random parts
props, locs, parts = solve.prefilter_partition(props_init, locs_init,
                                                                     params['in_field'],
                                                                     params['select_values'],
                                                                     parts = params['parts'] ,
                                                                     samples = params['samples'] ,
                                                                     )

################ Compute observables @(setting,vary,rep,part) and solve (1st iteration)
obs_part, sol_part, sol = solve.draw_solve(props,
                                                                   parts,
                                                                   params['exp'],
                                                                   weights = params['weights'],
                                                                   )

################ Assign solution to props
props = solve.assign_sol_to_props(props,
                                                       sol,
                                                       locs,
                                                       params['intensity'])

################ Compute observables @(setting,vary,rep,part) and solve (2nd iteration, including bound imager probabilities)
obs_part, sol_part_levels, sol_levels = solve.draw_solve(props,
                                                                                        parts,
                                                                                        params['exp'],
                                                                                        weights = params['weights'],
                                                                                        )

#%%
bins = np.linspace(0,6,60)

f=plt.figure(1,figsize=[4,3])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
f.clear()
ax = f.add_subplot(111)
ax.hist(sol_part.N,bins=bins,histtype='step',ec='k')
ax.hist(sol_part_levels.N,bins=bins,histtype='step',ec='r')

#%%
################ Mean observables @(setting,vary,rep,:) & expected observables @ solutions
obs, levels, obs_expect = solve.combine(props_part,
                                                                 obs_part,
                                                                 levels_part,
                                                                 [sol,sol_levels],
                                                                 )
