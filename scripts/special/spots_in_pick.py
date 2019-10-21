import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import lbfcs.io as io
import lbfcs.picasso_wrap as pic_wrap
import picasso.render as render
plt.style.use('~/lbFCS/styles/paper.mplstyle')

############################################################## Define data

locs_dir='/fs/pool/pool-schwille-paint/Data/p11.lbFCSnew/19-10-10_S2-adapter_Pm2/id164_S2_Pm2-10nM_p1800uW_1/19-10-10_FS'
locs_name='id164_S2_Pm2-10nM_p1800uW_1_MMStack_Pos0.ome_locs_render_align.hdf5'

props_dir='/fs/pool/pool-schwille-paint/Data/p11.lbFCSnew/19-10-10_S2-adapter_Pm2/id164_S2_Pm2-10nM_p35uW_2/19-10-10_FS'
props_name='id164_S2_Pm2-10nM_p35uW_2_MMStack_Pos0.ome_locs_render_picked_props_ig1.hdf5'

############################################################## Load data
locs,info_locs=io.load_locs(os.path.join(locs_dir,locs_name)) 
props,info_props=io.load_locs(os.path.join(props_dir,props_name)) 
pid=info_props[3]['Pick diameter'] 

#%%
############################################################## Cluster detection and analysis in locs corresponding to group in props
importlib.reload(pic_wrap)
group_centers=props.loc[:,['mean_x','mean_y']].values # Query locs for group centers
picks_idx=pic_wrap._query_locs_for_centers(locs,
                                          group_centers,
                                          pick_radius=pid*0.5)

oversampling=30
mng=30
box=7

from tqdm import tqdm
for i in tqdm([0]):
    group=props.group[i] # which group?
    pick_locs=locs.loc[picks_idx[i],:].reset_index(drop=True) # Picked localizations corresponding to group in props
    
    if len(pick_locs)>0: # Check if there are locs in pick
        viewport=[(group_centers[i,1]-pid/2,group_centers[i,0]-pid/2), # Define viewport for rendering of pick
                  (group_centers[i,1]+pid/2,group_centers[i,0]+pid/2)]
        image=render.render(pick_locs.to_records(index=False), # Render pick
                    oversampling=oversampling,
                    viewport=viewport,
                    )[1]
        spot_centers=pic_wrap._spots_in_image(image,mng,box,fit=False)
        
        if len(spot_centers)>0: # Check if spot was detected
            cluster_centers=pic_wrap._coordinate_convert(spot_centers, # Coordinate conversion
                                                         viewport[0],
                                                         oversampling)
            clusters_idx=pic_wrap._query_locs_for_centers(pick_locs, # Query locs for cluster center
                                                          cluster_centers.loc[:,['x','y']].values,
                                                          pick_radius=0.21)
            pick_locs_picked=pic_wrap._get_picked(pick_locs,clusters_idx,'cluster') # Assign cluster index
            pick_locs_picked['group']=group # Assign group index
            
            
            
############################################################## View result
try:
    ax=pic_wrap._image_preview(image,0,10,12)
    pic_wrap._spots_preview(spot_centers,ax,1500)
    
    f=plt.figure(num=13,figsize=[4,4])
    f.subplots_adjust(bottom=0.,top=1.,left=0.,right=1.)
    f.clear()
    ax=f.add_subplot(111)
    color_palette=['r','b','magenta','grey']
    cluster_colors = [color_palette[x] if x >= 0
                      else 'k'
                      for x in pick_locs_picked.cluster]
    
    ax.scatter(pick_locs_picked.x,pick_locs_picked.y,s=40,c=cluster_colors,marker='x')
    ax.set_xlim(viewport[0][1],viewport[1][1])
    ax.set_ylim(viewport[0][0],viewport[1][0])
except NameError:
    print('\nNo locs or spots in group!')
    


