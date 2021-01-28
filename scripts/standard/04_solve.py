import numpy as np
import matplotlib.pyplot as plt

import lbfcs.solveseries as solve

plt.style.use('~/lbFCS/styles/paper.mplstyle')

#%%
#################### Define parameters
params = {}

### Old Pm2@200ms & T=21C
# params['dir_name'] = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/z.olddata/N1_T21/20-01-19_FS'

### New Pm2@200ms
# params['dir_name'] = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-09_N1_T23_ibidi_cseries/21-01-19_FS_id181'

### 5xCTC@200ms&400ms
# params['dir_name'] = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-09_N1_T23_ibidi_cseries/21-01-19_FS_id180'
# params['dir_name'] = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-10_N1-5xCTC_cseries_varexp/21-01-19_FS_id180_exp200'
# params['dir_name'] = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-10_N1-5xCTC_cseries_varexp/21-01-19_FS_id180_exp400'
# params['dir_name']  = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-17_N1-2x5xCTC_cseries/20-12-17_FS_id180'
# params['dir_name']  = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-17_N1-2x5xCTC_cseries/20-12-17_FS_id180_occ-newcorr'

### N != 1, 5xCTC@400ms
# params['dir_name']  = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-18_N2-5xCTC_cseries/20-12-18_FS_id194'
params['dir_name']  = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-18_N2-5xCTC_cseries/20-12-18_FS_id194_occ-newcorr'

### S1_5xCTC adapter@200ms&400ms
# params['dir_name'] = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-10_N1-5xCTC_cseries_varexp/21-01-19_FS_id163_exp200'
# params['dir_name'] = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-10_N1-5xCTC_cseries_varexp/21-01-19_FS_id163_exp400'

### N=2, JHK_5xCTC adapter@200ms&400ms
# params['dir_name']  = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-17_N1-2x5xCTC_cseries/20-12-17_FS_id192_sample2'

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

### Combine
obsol_combined = solve.combine_obsol(obsol)
solve.print_solutions(obsol_combined)

### Save
solve.save_series(file_list, obsol, params)



#%%
#################### Plot anything
bins = np.linspace(0,5,100)
# bins = 'fd'

f = plt.figure(0,figsize = [4,3])
f.clear()
ax = f.add_subplot(111)
ax.hist(obsol.query('nn_d > 5').N,bins=bins)
