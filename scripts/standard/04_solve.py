import numpy as np
import matplotlib.pyplot as plt

import lbfcs.solveseries as solve

plt.style.use('~/lbFCS/styles/paper.mplstyle')

#%%
#################### Define parameters
params = {}
### 5xCTC @exp400 & kon17e6
# params['dir_name'] = '/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-01-22_Simulation/5xCTC_exp400_T23_kon17e6/N02'
### 5xCTC @exp400 & kon30e6
params['dir_name'] = '/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-01-28_Simulation_meanvarI-test/5xCTC_exp400_T23_kon30e6/N12'
### 5xCTC @exp200 & kon30e6
# params['dir_name'] = '/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-01-28_Simulation_meanvarI-test/5xCTC_exp200_T23_kon30e6/N04'

### Experimental data
# params['dir_name'] = '/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-18_N2-5xCTC_cseries/20-12-18_FS_id194_meanvarI-test'

params['exp'] = 0.4
params['exclude_rep'] = []
params['weights']        = [1,1,1,0,0]
params['solve_mode'] = 'single'


#%%
#################### Load, filter, solve, combine & save
### Load 
props_init, file_list = solve.load_props_in_dir(params['dir_name'])
print();print(file_list)

### Filter
props = solve.exclude_filter(props_init, params['exclude_rep'] )

### Solve
obsol = solve.get_obsol(props, params['exp'], params['weights'], params['solve_mode'])

#%%
### Combine
obsol_combined = solve.combine_obsol(obsol)
print()
print('Combined solutions:')
solve.print_solutions(obsol_combined)

### Ensemble solution
obsol_ensemble, obsol_ensemble_combined = solve.get_obsols_ensemble(obsol.query('vary <= 625'),[1,1,0,0,0])
print()
print('Ensemble solution:')
solve.print_solutions(obsol_ensemble_combined)

## Save
# solve.save_series(file_list, obsol, params)


#################### Plot anything
bins = np.linspace(3,30,100)
bins = 'fd'

field = 'snr'
query_str = 'vary <= 625'

f = plt.figure(0,figsize = [4,3])
f.clear()
ax = f.add_subplot(111)
ax.hist(obsol.query(query_str)[field],bins=bins,histtype='step')
ax.hist(obsol_ensemble.query(query_str)[field],bins=bins,histtype='step')
