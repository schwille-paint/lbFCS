
#Script to autopick on image basis and index locs accordingly
import os
from lbfcs.lbfcs import _master
import picasso.render as render
import lbfcs.picasso_wrap as pic_wrap
#%%

############################################################## Load data
dir_name='/fs/pool/pool-schwille-paint/Data/p11.lbFCSnew/19-10-21_c-series_N1_R1-9/id161_R1-9_20nM_p35uW_1'
file_name='id161_R1-9_20nM_p35uW_1_MMStack_Pos0.ome.tif'

path=os.path.join(dir_name,file_name) 

movie,info,locs,locs_render,centers,locs_picked,_props,conc=master(path,
                                                                   box=5,
                                                                   mng=400,
                                                                   segments=500,
                                                                   oversampling=5,
                                                                   box_pick=9,
                                                                   mng_pick=300,
                                                                   pick_diameter=1.6,
                                                                   ignore=1,
                                                                   NoPartitions=30)

#%%
############################################################## Render
oversampling=5
image=render.render(locs_render,
                    info,
                    oversampling=oversampling,
                    )[1]


############################################################## Preview picks
#### Spot detection settings
box=9
mng=300

#### Image settings
contrast_min=0
contrast_max=100
view=[(700,700),(200,200)]

#### Show preview
ax=pic_wrap._image_preview(image,
                           view,
                           contrast_min=contrast_min,
                           contrast_max=contrast_max,
                           fignum=13)
spots=pic_wrap._spots_in_image(image,mng,box)
pic_wrap._spots_preview(spots,
                       ax)
