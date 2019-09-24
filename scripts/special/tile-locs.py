# Import modules
import os
import importlib

#### Define path to file
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p04.lb-FCS/19-01-10_c-series_SDS-Pm2-8nt-NoPEG/SDS-Pm2-8nt-NoPEG_c50nM_1/19-01-20_JS']*3)

file_names=[]
file_names.extend(['SDS-Pm2-8nt-NoPEG_c50nM_1_MMStack_Pos0.ome_locs_render.hdf5'])
file_names.extend(['SDS-Pm2-8nt-NoPEG_c50nM_1_MMStack_Pos1.ome_locs_render.hdf5'])
file_names.extend(['SDS-Pm2-8nt-NoPEG_c50nM_1_MMStack_Pos2.ome_locs_render.hdf5'])

#### Create list of paths
path=[os.path.join(dir_names[i],file_names[i]) for i in range(0,len(file_names))]

#%%
#### Segment
import pickprops_calls as props_call
# Reload modules
importlib.reload(props_call)

noTiles=8
center=[512,512]
width=512

for p in path:
    print(p)
    locs=props_call.tile_locs(p,noTiles,center,width)

#%% 
#import matplotlib.pyplot as plt
#import mpl_scatter_density # density module for fast scatter plots
#
##### x-y scatter after
#f=plt.figure(num=2)
#f.clear()
#tile=62
#x=locs.loc[locs.group==tile,'x'].values
#y=locs.loc[locs.group==tile,'y'].values
#
#ax = f.add_subplot(1, 1, 1, projection='scatter_density')
#plt_map=ax.scatter_density(x, y,cmap='jet')
#plt.colorbar(plt_map,ax=ax)
##plt_map.set_clim(0,5e3)
#ax.set_xlim(0,1024)
#ax.set_ylim(0,1024)

