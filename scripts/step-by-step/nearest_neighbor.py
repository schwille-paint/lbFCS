import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial

import picasso.io as io


### High density data
dir_name = '/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-09_N1_T23_ibidi_cseries/21-01-19_FS_id180'
file_name = 'ibidi_id180_Pm2-05nM_p40uW_1_MMStack_Pos0.ome_locs_render_picked_props.hdf5'

### Low density data
# dir_name = '/fs/pool/pool-schwille-paint/Data/p17.lbFCS2/20-12-10_N1-5xCTC_cseries_varexp/21-01-19_FS_id180_exp400'
# file_name = 'id180_Pm2-2d5nM_p40uW_exp400_1_MMStack_Pos0.ome_locs_render_picked_props.hdf5'

### Load props
props = pd.DataFrame(io.load_locs(os.path.join(dir_name,file_name))[0]) 

### Build KDtree and ask for nearest neighbor
data = props[['x','y']].values.astype(np.float32)
tree = spatial.cKDTree(data)
d,idx = tree.query(data,k=[2])

### Plot nearest neighbor distance distribution in data
f = plt.figure(0,figsize = [5,4])
f.clear()
ax = f.add_subplot(111)
ax.hist(d,bins=np.linspace(0,12,100))

