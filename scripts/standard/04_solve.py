import numpy as np
import matplotlib.pyplot as plt
import os

import lbfcs.solveseries as solve
import lbfcs.visualizeseries as visualize
import picasso_addon.io as io

plt.style.use('~/lbFCS/styles/paper.mplstyle')

#%%
#################### Define parameters
params = {}
# params['dir_name'] = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-02-23_EGFR-aGFP-5xCTC/02_w2_FOV2_561_p40uW_c1250pM_1/box5_mng600'
# params['dir_name'] = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-02-23_EGFR-aGFP-5xCTC/01_w1_FOV2_561_pos15um_p40uW_c2500pM_POC_1/box5_mng600'
# params['dir_name'] = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-18_N2-5xCTC_cseries/20-12-18_FS_id194'
# params['dir_name'] = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-18_N2-5xCTC_cseries/20-12-18_FS_id194_epsv02'
params['dir_name'] = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-01-28_higherN-5xCTC_cseries/21-02-24_FS_N4_epsv02'

params['exp'] = 0.4
params['exclude_rep'] = []

params['weights'] = [1,1,1,1,0,0,0,0]
params['eps_unknown'] = False

#%%
#################### Load, filter, solve, save
### Load 
props_init, file_list = solve.load_props_in_dir(params['dir_name'])

print();print(file_list)
### Filter
# props = solve.exclude_filter(props_init, params['exclude_rep'])
props = props_init.copy()

### Solve
obsol = solve.get_obsol(props, 
                        params['exp'],
                        params['weights'],
                        params['eps_unknown'],
                        )

## Save
solve.save_series(file_list, obsol, params)

#%%
#################### Plotting
save_picks = True
exp = params['exp']
v = 5000


query_str = 'vary >= @v ' 
query_str += 'and success >= 99'
# query_str += 'and nn_d >= 5'
query_str += 'and abs(frame-M/2)*(2/M) < 0.15'
query_str += 'and std_frame - 0.85*M/4 > 0'
# query_str += 'and koff > 0.07*0.4'
# query_str += 'and konc*(1e-6/(@exp*vary*1e-12)) > 20'

data = obsol.query(query_str)
if save_picks: io.save_picks(data,3,os.path.join(params['dir_name'],'%spicks.yaml'%(str(v).zfill(4))))

### Combine
data_combined = solve.combine_obsol(data)
print()
print('Combined solutions:')
solve.print_solutions(data_combined)


####################
field = 'success'
bins = np.linspace(90,100,100)

f = plt.figure(0,figsize = [4,3])
f.clear()
ax = f.add_subplot(111)
ax.hist(data[field],
        bins=bins,histtype='step',ec='k')

####################
field = 'N'
bins = np.linspace(0,7,70)

f = plt.figure(1,figsize = [4,3])
f.clear()
ax = f.add_subplot(111)
ax.hist(data[field]*0.9,
        bins=bins,histtype='step',ec='k')

####################
field = 'koff'
bins = np.linspace(0,0.3,40)

f = plt.figure(2,figsize = [4,3])
f.clear()
ax = f.add_subplot(111)
ax.hist(data[field]/exp,
        bins=bins,histtype='step',ec='k')

####################
field = 'konc'
bins = np.linspace(0,50,40)

f = plt.figure(3,figsize = [4,3])
f.clear()
ax = f.add_subplot(111)
ax.hist(data[field]*(1e-6/(exp*data.vary*1e-12)),
        bins=bins,histtype='step',ec='k')


#################### Plot residuals
# eps_field = 'eps_direct'
# if params['eps_unknown']: eps_field = 'eps'

# f = plt.figure(1,figsize = [4,3])
# f.clear()
# ax = f.add_subplot(111)
# visualize.residual_violinplot_toax(ax,
#                                     solve.compute_residuals(obsol.query(query_str),eps_field=eps_field),
#                                     )
# ax.set_ylim(-3,3)