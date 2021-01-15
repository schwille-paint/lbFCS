import numpy as np
import matplotlib.pyplot as plt

import lbfcs.solveseries as solve

plt.style.use('~/lbFCS/styles/paper.mplstyle')

#%%
#################### Define parameters
params = {}

params['dir_name']  = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-17_N1-2x5xCTC_cseries/20-12-17_FS_id180'
# params['dir_name']  = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-18_N2-5xCTC_cseries/20-12-18_FS_id194'
# params['dir_name']  = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-17_N1-2x5xCTC_cseries/20-12-17_FS_id192_sample2'

params['exp'] = 0.4
params['exclude_rep'] = [0,1,2,4,5,6]
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

### Combine
obsol_combined = solve.combine_obsol(obsol)
solve.print_solutions(obsol_combined)

### Save


#%%
# ################ Plot anything else ...
x = obsol.loc[(obsol.success==True)&(obsol.n_points==1)]
bins = np.linspace(0,5,60)
# bins = 'fd'

f=plt.figure(1,figsize=[4,3])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
f.clear()
ax = f.add_subplot(111)
ax.hist(x.N,bins=bins)