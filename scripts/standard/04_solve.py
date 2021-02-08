import numpy as np
import matplotlib.pyplot as plt

import lbfcs.solveseries as solve

plt.style.use('~/lbFCS/styles/paper.mplstyle')

#%%
#################### Define parameters
params = {}
params['dir_name'] = '/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-02-07_N2_a20_var/21-02-07_JS_id206'

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
obsol = solve.get_obsol(props, params['exp'])
## Save
solve.save_series(file_list, obsol, params)


#%%
#################### Combine and ensemble solution
v_upp = 5000
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
N_up = obsol_combined.setting.iloc[0] * 2
bins = np.linspace(0,N_up,50)
# bins = np.linspace(0,0.1,70)
# bins = 40

field = 'N'
query_str = 'vary <= @v_upp and nn_d > 5 and success == 1'

f = plt.figure(0,figsize = [4,3])
f.clear()
ax = f.add_subplot(111)
ax.hist(obsol.query(query_str)[field],bins=bins,histtype='step')
ax.hist(obsol_ensemble.query(query_str)[field],bins=bins,histtype='step')