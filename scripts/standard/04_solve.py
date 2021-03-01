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
### EGFR, 2nd try
# params['dir_name'] = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-02-25_EGFR-aGFP-controls/w6_03_561_FOV1_Pm2-8nt-c1250_p40uW_1/box5_mng600_pd12_use-eps'
# params['dir_name'] = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-02-25_EGFR-aGFP-controls/w6_05_561_FOV2_Pm2-8nt-c1250_p40uW_1/box5mng600_pd12_use-eps'
params['dir_name'] = r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/21-02-25_EGFR-aGFP-controls/w7_10_561_FOV4_Pm2-8nt-c1250_p40uW_POCGX_1/box_mng600_pd12_use-eps'


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
save_picks = False
exp = params['exp']
v = 1250


query_str = 'vary == @v ' 
query_str += 'and success >= 98'
query_str += 'and abs(frame-M/2)*(2/M) < 0.15'
query_str += 'and std_frame - 0.85*M/4 > 0'

# query_str += 'and occ < 0.15 '
# query_str += 'and N < 1.4 '
# query_str += 'and koff > 0.16*@exp '
# query_str += 'and konc*(1e-6/(@exp*vary*1e-12)) > 2.5'

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
bins = np.linspace(0,4,60)
ax = f.add_subplot(412)

ax.hist(data[field]*0.9,
        bins=bins,histtype='step',ec='k')

#################### koff
field = 'koff'
bins = np.linspace(0,0.3,50)
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