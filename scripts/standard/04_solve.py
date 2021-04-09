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


params['dir_name'] = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-04-08_origami-checks/01_s1_5000pM_p30uW_exp400_1/20-04-08_FS'

params['exp'] = 0.4
params['exclude_rep'] = []
params['weights'] = [1,1,1,1,0,0,0,0]
params['eps_unknown'] = False

#%%
#################### Load, solve, save
### Load 
props_init, file_list = solve.load_props_in_dir(params['dir_name'])
props = props_init.copy()
print();print(file_list)


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


query_str = 'vary == @v ' 
query_str += 'and success >= 99 '
query_str += 'and abs(frame-M/2)*(2/M) < 0.2'
query_str += 'and std_frame - 0.8*M/4 > 0'

query_str += 'and N < 3.5 '
# query_str += 'and koff > 0.065*@exp '
query_str += 'and konc*(1e-6/(@exp*vary*1e-12)) > 10'

data = obsol.query(query_str)
if save_picks: io.save_picks(data,3,os.path.join(params['dir_name'],'%spicks.yaml'%(str(v).zfill(4))))

### Combine
data_combined = solve.combine_obsol(data)
print()
print('Combined solutions:')
solve.print_solutions(data_combined)


################### Plot results

f = plt.figure(0,figsize = [5,9])
f.clear()
f.subplots_adjust(bottom=0.1,wspace=0.3)
#################### success
field = 'success'
bins = np.linspace(90,100,100)
ax = f.add_subplot(411)

ax.hist(data[field],
        bins=bins,histtype='step',ec='k')

#################### N
field = 'N'
bins = np.linspace(0,5,50)
ax = f.add_subplot(412)

ax.hist(data[field],
        bins=bins,histtype='step',ec='k')

#################### koff
field = 'koff'
bins = np.linspace(0,0.45,50)
ax = f.add_subplot(413)

ax.hist(data[field]/exp,
        bins=bins,histtype='step',ec='k')

#################### kon
field = 'konc'
bins = np.linspace(0,30,50)
ax = f.add_subplot(414)

ax.hist(data[field]*(1e-6/(exp*data.vary*1e-12)),
        bins=bins,histtype='step',ec='k')


#################### Plot residuals
eps_field = 'eps_direct'
if params['eps_unknown']: eps_field = 'eps'

f = plt.figure(1,figsize = [5,3])
f.clear()
ax = f.add_subplot(111)
visualize.residual_violinplot_toax(ax,
                                    solve.compute_residuals(data.query(query_str),eps_field=eps_field),
                                    )
ax.set_ylim(-5,5)