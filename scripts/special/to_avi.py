##################################### Import
import numpy as np
import os
import matplotlib.pyplot as plt
# import cv2

import picasso.io as io

##################################### Path definitions
dir_names=[]
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p15.RiboCounting/19-11-22/03_FOV1_GFP_p019uW_1'])
dir_names.extend(['/fs/pool/pool-schwille-paint/Data/p15.RiboCounting/19-11-22/04_FOV1_Cy3B_p038uW_R1s1-8-100pM_POC_1'])

file_names=[]
file_names.extend(['03_FOV1_GFP_p019uW_1_MMStack_Pos0.ome.tif'])
file_names.extend(['04_FOV1_Cy3B_p038uW_R1s1-8-100pM_POC_1_MMStack_Pos0.ome.tif'])

pick_dir='/fs/pool/pool-schwille-paint/Data/p15.RiboCounting/19-11-22/04_FOV1_Cy3B_p038uW_R1s1-8-100pM_POC_1/19-11-22_FS'
save_dir='/fs/pool/pool-schwille-paint/Analysis/p15.RiboCounting/plots/19-11-29'

##################################### Load movie and picks
path=[os.path.join(dir_names[i],file_names[i]) for i in range(0,len(file_names))]
movie=[io.load_tif(p)[0] for p in path]
info=[io.load_tif(p)[1] for p in path]
picks=io.load_info(os.path.join(pick_dir,'FOV1_allcells_picks.yaml'))[0]

#%%
### Function to draw picks
def draw_centerline(ax,picks,dx,dy):
    lines=picks['Center-Axis-Points']
    for line in lines:
        ax.plot([line[0][0]+dx,line[1][0]+dx],
                [line[0][1]+dy,line[1][1]+dy],
                c='r',
                lw=1,
                )
    return ax

### Settings
frame_min=0
frame_max=150
hlim=[400,200]
wlim=[0,200]

f=plt.figure(1,figsize=[6,6])
f.clear()
f.subplots_adjust(bottom=0,top=1,left=0,right=1)
ax=f.add_subplot(111)
ax.imshow(movie[0][0],
              vmin=0,
              vmax=40e3,
              cmap='gray',
              interpolation='nearest',
              )
draw_centerline(ax,picks,2.5,7)

ax.set_xlim(wlim[0],wlim[1])
ax.set_ylim(hlim[1],hlim[0])
# plt.savefig(os.path.join(save_dir,'FOV1_GFP.pdf'))

frame=21
f=plt.figure(2,figsize=[6,6])
f.clear()
f.subplots_adjust(bottom=0,top=1,left=0,right=1)
ax=f.add_subplot(111)
ax.imshow(movie[1][frame],
          vmin=100,
          vmax=500,
          cmap='gray',
          interpolation='nearest',
          )

draw_centerline(ax,picks,0,0)

ax.set_xlim(wlim[0],wlim[1])
ax.set_ylim(hlim[1],hlim[0])
ax.text(20,220,'%2.1f s'%(frame*0.2),color='w',fontsize=25)

#%%
f=plt.figure(2,figsize=[6,6])
f.clear()
f.subplots_adjust(bottom=0,top=1,left=0,right=1)
ax=f.add_subplot(111)

for frame in range(frame_min,frame_max):
    ax.clear()
    ### Image
    ax.imshow(movie[1][frame],
                  vmin=100,
                  vmax=500,
                  cmap='gray',
                  interpolation='nearest',
                  ) 
    ### Picks
    ax=draw_centerline(ax,picks,0,0)   
    ### Timestamp
    ax.text(20,220,'%2.1f s'%(frame*0.2),color='w',fontsize=25)
    ### Limits
    ax.set_xlim(wlim[0],wlim[1])
    ax.set_ylim(hlim[1],hlim[0])
    
    plt.savefig(os.path.join(save_dir,'gif','FOV1_100pM-%i.pdf'%(frame)),dpi=100)


#%%
############################################## Video file export, under progress
#### Normalize to maximum
img_max=np.max(img_array.flatten())
img_array=img_array/img_max
### Adjust contrast
img_array[img_array>=c_max]=c_max
### Convert to 8bit
img_array=((img_array/c_max)*2**8-1).astype('uint8')


### Export video
out = cv2.VideoWriter(os.path.join(dir_names[i],'test_wmv2.avi'),
                      cv2.VideoWriter_fourcc(*'WMV2'), #Video codec
                      5, #fps
                      (hlim[1]-hlim[0],wlim[1]-wlim[0]),
                      isColor=False,
                      )
 


### Adjust brightess

for j in range(len(img_array)):
    out.write(img_array[j])
out.release()