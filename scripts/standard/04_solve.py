import numpy as np
import matplotlib.pyplot as plt

import lbfcs.solveseries as solve

plt.style.use('~/lbFCS/styles/paper.mplstyle')

#%%
#################### Define parameters
params = {}
### N=2, 5xCTC
# params['dir_name'] = '/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-18_N2-5xCTC_cseries/20-12-18_FS_id194_new'
### N=4, 5xCTC
# params['dir_name'] = '/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-01-28_higherN-5xCTC_cseries/20-01-28_JS_N4_new'
### N=4, 5xCTC
params['dir_name'] = '/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-01-28_higherN-5xCTC_cseries/21-01-28_JS_N6_new'

params['exp'] = 0.4
params['exclude_rep'] = []
params['weights']        = [1,1,1,1,1,0,0]
params['solve_mode'] = 'single'


#%%
#################### Load, filter, solve, save & combine + ensemble solution
### Load 
props_init, file_list = solve.load_props_in_dir(params['dir_name'])
print();print(file_list)
### Filter
props = solve.exclude_filter(props_init, params['exclude_rep'] )
### Solve
obsol = solve.get_obsol(props, params['exp'], params['weights'], params['solve_mode'])
## Save
solve.save_series(file_list, obsol, params)

#%%
### Combine
obsol_combined = solve.combine_obsol(obsol)
print()
print('Combined solutions:')
solve.print_solutions(obsol_combined)

### Ensemble solution
obsol_ensemble, obsol_ensemble_combined = solve.get_obsols_ensemble(obsol.query('vary <= 1251 and nn_d > 5'),[1,1,0,0,0,0,0])
print()
print('Ensemble solution:')
solve.print_solutions(obsol_ensemble_combined)

#################### Plot anything
bins = np.linspace(0,8,70)
# bins = 'fd'

field = 'N'
query_str = 'vary == 1251 and nn_d > 5'

f = plt.figure(0,figsize = [4,3])
f.clear()
ax = f.add_subplot(111)
ax.hist(obsol.query(query_str)[field],bins=bins,histtype='step')
ax.hist(obsol_ensemble.query(query_str)[field],bins=bins,histtype='step')
