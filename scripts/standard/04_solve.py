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
dir_name=r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-11-04_N12_T23_Buffer-Repeat/20-11-04_FS'

paths = sorted( glob.glob(os.path.join(dir_name,'*_props.hdf5')) )
props_init = pd.concat([pd.DataFrame(io.load_locs(p)[0]) for p in paths],keys=range(len(paths)),names=['rep'])
props_init = props_init.reset_index(level=['rep'])

files_props = [os.path.split(path)[-1] for path in paths]

paths = sorted( glob.glob(os.path.join(dir_name,'*.hdf5')) )
paths = [path for path in paths if not bool(re.search('props',path))]
locs = pd.concat([pd.DataFrame(io.load_locs(p)[0]) for p in paths],keys=range(len(paths)),names=['rep'])
locs = locs.reset_index(level=['rep'])  

files_locs = [os.path.split(path)[-1] for path in paths]

#%%

############################################
'''
Part 1: Find solution using props only
'''
############################################
props = props_init.copy()
# props = props[props.vary == 10000]

############################### Filter props @(setting,vary,rep)
props = props.groupby(['setting','vary','rep']).apply(solve.prefilter)
props = props.droplevel(level=['setting','vary','rep'])

############################### Bootstrap props @(setting,vary,rep) -> part
props_part = props.groupby(['setting','vary','rep']).apply(lambda df: solve.bootstrap_props(df,parts=20,samples=500))
props_part = props_part.droplevel(['setting','vary','rep'])

############################### (Statistical) observables @(setting,vary,rep,part)
obs_part = props_part.groupby(['setting','vary','rep','part']).apply(solve.extract_observables)
obs_part = obs_part.reset_index(level=['setting','vary','rep','part'])

############################### 1st iteration solution @(setting,part) using tau,A,occ,taud,events
sol_part = obs_part.groupby(['setting','part']).apply(solve.solve_eqs)
sol_part = sol_part.reset_index(level=['setting','part'])

############################### Combine 1st iteration solutions @(setting)
sol = sol_part.groupby(['setting']).apply(solve.combine_solutions)
sol = sol.reset_index(level=['setting'])

#%%
############################################
'''
Part 2: Find new solution by using 1st solution to get imager levels 
'''
############################################

############################## Get imager levels @(setting,vary,part)
levels = props.groupby(['setting','vary','rep']).apply(lambda df: solve.get_levels(df,locs,sol,filtered=True))
levels = levels.reset_index(level=['setting','vary','rep'])

############################## Fit levels and assign result to obs_part @(setting,vary,part)
obs_part_levels = levels.groupby(['setting','vary','rep']).apply(lambda df: solve.assign_levels(df,obs_part))
obs_part_levels = obs_part_levels.droplevel(['setting','vary','rep'])

############################### Combine observables for different parts @(setting,vary,rep)
obs_levels = obs_part_levels.groupby(['setting','vary','rep']).apply(solve.combine_observables)
obs_levels = obs_levels.reset_index(level=['setting','vary','rep'])

############################### 2nd iteration solution @(setting,part) using tau,A,occ,taud,events and bound imager probability
sol_part_levels = obs_part_levels.groupby(['setting','part']).apply(solve.solve_eqs)
sol_part_levels = sol_part_levels.reset_index(level=['setting','part'])

############################### Troubleshooting
# #%%
# importlib.reload(solve)
# df_group = obs_part_levels.groupby(['setting','part'])
# groups = list(df_group.groups)
# df = df_group.get_group(groups[1])

# data = solve.prep_data(df)

# # x0   = np.array([float(sol.konc20000),float(sol.koff),float(sol.N)])
# # x0   = np.array([float(sol_levels.konc20000),float(sol_levels.koff),float(sol_levels.N)])
# x0   = np.array([float(sol_group.konc20000),float(sol_group.koff),float(sol_group.N)])

# eqs = solve.create_eqs_konc(x0,data,model=model)

# print(eqs[0:5])
# print()
# sol_group  = solve.solve_eqs_konc(data,model=model)
# print()
# print(sol_group)

# # %%

############################### Combine 2nd iteration solutions @(setting)
sol_levels = sol_part_levels.groupby(['setting']).apply(solve.combine_solutions)
sol_levels = sol_levels.reset_index(level=['setting'])


############################### Observables according to 1st iteration solution
obs_to_sol = obs_levels.groupby('setting').apply(lambda df: solve.expected_observables(df,sol))
obs_to_sol = obs_to_sol.droplevel(['setting'])

############################### Observables according to 2nd iteration solution
obs_to_sol_levels = obs_levels.groupby('setting').apply(lambda df: solve.expected_observables(df,sol_levels))
obs_to_sol_levels = obs_to_sol_levels.droplevel(['setting'])

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
save_path    = r'path-to some-folder'

save_subpath, do_save = visualize.create_savesubpath(save_path,do_save,obs_levels)

######################################## Plot subset and field histogram
field  = 'n_locs'
subset = props.vary == 10000
bins   = np.linspace(0,1,100)

f=plt.figure(1,figsize=[4,3])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
f.clear()
ax = f.add_subplot(111) 
ax.hist(props.loc[subset,field],bins=bins,ec='k',histtype='step')
ax.set_xlabel(field)
ax.set_ylabel('Counts')

if do_save: plt.savefig(os.path.join(save_subpath,'N.pdf'),transparent=True)


########################################### Print results
exp    = 0.2 # Conversion factor for real units

print('-----> No levels')
visualize.print_sol(sol,exp)
print('-----> Levels')
visualize.print_sol(sol_levels,exp)

########################################### Standardized results visualization
# Compare with old procedure
visualize.compare_old(obs_levels,sol_levels,exp)
if do_save: plt.savefig(os.path.join(save_subpath,'obs.pdf'),transparent=True)

# Observables residual from solution
visualize.obs_relresidual(obs_levels,obs_to_sol_levels)
if do_save: plt.savefig(os.path.join(save_subpath,'res.pdf'),transparent=True)

# Levels
visualize.show_levels(levels)                       
if do_save: plt.savefig(os.path.join(save_subpath,'levels.pdf'),transparent=True)

