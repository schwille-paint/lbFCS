import numpy as np
import matplotlib.pyplot as plt

import lbfcs.solveseries as solve
import lbfcs.visualizeseries as visualize

plt.style.use('~/lbFCS/styles/paper.mplstyle')

#%%
#################### Define parameters
params = {}
params['dir_name'] = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-17_N1-2x5xCTC_cseries/20-12-17_FS_id192_sample2'

params['exp'] = 0.4
params['exclude_rep'] = []

#%%
#################### Load, filter, solve, save
### Load 
props_init, file_list = solve.load_props_in_dir(params['dir_name'])
print();print(file_list)
### Filter
props = solve.exclude_filter(props_init, params['exclude_rep'] )
### Solve
obsol = solve.get_obsol(props, params['exp'],[1,1,1,1,0,0,0,0])
## Save
# solve.save_series(file_list, obsol, params)


#%%
#################### Combine and ensemble solution
v_upp = 1250
v_low = int(np.ceil(v_upp/2))

### Combine
obsol_combined = solve.combine_obsol(obsol)
print()
print('Combined solutions:')
solve.print_solutions(obsol_combined)

### Ensemble solution
obsol_ensemble, obsol_ensemble_combined = solve.get_obsols_ensemble(obsol.query('vary == @v_low or vary == @v_upp '))
print()
print('Ensemble solution:')
solve.print_solutions(obsol_ensemble_combined)


#################### Plot anything
N_up = obsol_combined.setting.iloc[0] * 6
bins = np.linspace(0,N_up,50)
# bins = np.linspace(0,100,50)
# bins=100

field = 'N'
query_str = 'vary == @v_upp and nn_d > 5 and success > 90'

f = plt.figure(0,figsize = [4,3])
f.clear()
ax = f.add_subplot(111)
ax.hist(obsol.query(query_str)[field],bins=bins,histtype='step')
# ax.hist(obsol.query(query_str)[field + '_direct'],bins=bins,histtype='step')


#################### Plot residuals
f = plt.figure(1,figsize = [4,3])
f.clear()
ax = f.add_subplot(111)
visualize.residual_violinplot_toax(ax,
                                   solve.compute_residuals(obsol.query(query_str)),
                                   )