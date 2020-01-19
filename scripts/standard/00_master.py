# Localize, undrift autopick and kinetic anaylisis in one stroke!
import os
from lbfcs.__main__ import _lbfcs
import picasso.render as render
import lbfcs.picasso_wrap as pic_wrap
import lbfcs.pyplot_wrap as plt_wrap
#%%

############################################################## Load data
dir_name='/fs/pool/pool-schwille-paint/Data/p05.crowding/19-10-28_8000_PEG_15%/id154_1_25nM_20%_8000_PEG_38uW_1'
file_name='id154_1_25nM_20%_8000_PEG_38uW_1_MMStack_Pos0.ome.tif'

path=os.path.join(dir_name,file_name) 

movie,info,locs,locs_render,centers,locs_picked,_props,conc=_lbfcs(path,
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

#### Show preview
ax=plt_wrap._image_preview(image,
                           contrast_min=contrast_min,
                           contrast_max=contrast_max,
                           fignum=13)
ax.set_xlim(250,450)
ax.set_ylim(250,450)

spots=pic_wrap._spots_in_image(image,mng,box)
plt_wrap._spots_preview(spots,ax)
