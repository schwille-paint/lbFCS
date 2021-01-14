import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import glob
import importlib
import re
import time
from tqdm import tqdm
tqdm.pandas()

import picasso.io as io
import lbfcs.solveseries as solve
import lbfcs.visualizeseries as visualize

plt.style.use('~/lbFCS/styles/paper.mplstyle')

#%%
############################### Define data
dir_names = []
dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-17_N1-2x5xCTC_cseries/20-12-17_FS_id180'])
# dir_names.extend([r'/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-18_N2-5xCTC_cseries/20-12-18_FS_id194'])
# dir_name = '/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-17_N1-2x5xCTC_cseries/20-12-17_FS_id192_sample2'

paths = []
for dir_name in dir_names:
    files = sorted( glob.glob( os.path.join( dir_name,'*_props.hdf5') ) )
    paths.extend(files)
                      
props_init = pd.concat([pd.DataFrame(io.load_locs(p)[0]) for p in paths],keys=range(len(paths)),names=['rep'])
props_init = props_init.reset_index(level=['rep'])

#%%
select_field = 'vary'
select_values = [1250]
weights = [1,1,1,0,0]

### Select subset of props
props = props_init.query( select_field + ' in @select_values' )

### Filter props
props = props.groupby(['setting','vary','rep']).apply(solve.prefilter)
props = props.droplevel( level = ['setting','vary','rep'])

#%%
### Test individual fitting
obsol = props.groupby(['setting','group']).progress_apply(lambda df: solve.obsol(df,weights))
obsol = obsol.droplevel( level = ['setting','group'])

#%%
### Test ensemble fitting
obsol_ensemble = obsol.groupby(['setting']).apply(lambda df: solve.obsol_ensemble(df,weights))
obsol_ensemble = obsol_ensemble.droplevel( level = ['setting'])

#%%
# ################ Plot anything else ...
x = obsol.loc[(obsol.success==True)&(obsol.n_points==1)]
bins = np.linspace(0,3,60)
# bins = 'fd'

f=plt.figure(1,figsize=[4,3])
f.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.95)
f.clear()
ax = f.add_subplot(111)
ax.hist(x.N,bins=bins)