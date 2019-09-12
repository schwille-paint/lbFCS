import picasso.io as io
import picasso.localize as localize
import picasso.gausslq as gausslq

path = '/fs/pool/pool-schwille-paint/Data/p12.ACAB/19-08-29_id147/id147_20nM_p35uW_1/id147_20nM_p35uW_1_MMStack_Pos0.ome.tif'


def fit_from_path(path,box=5,mng=400):
    
    #### Load raw movie
    print('Loading movie ...')
    movie, info = io.load_movie(path)
    
    #### Set camera specs for photon conversion
    camera_info = {}
    camera_info['baseline'] = 70
    camera_info['gain'] = 1
    camera_info['sensitivity'] = 0.56
    camera_info['qe'] = 0.82
    
    #### Spot detection
    print('Spot detection ...')
    current, futures = localize.identify_async(movie,
                                               mng,
                                               box,
                                               None)

    identifications = localize.identifications_from_futures(futures)
    spots = localize.get_spots(movie, identifications, box, camera_info)

    #### Gauss-Fitting
    print('Fitting ...')
    fs = gausslq.fit_spots_parallel(spots, asynch=True)
    theta = gausslq.fits_from_futures(fs)
    em = camera_info['gain'] > 1
    locs = gausslq.locs_from_fits(identifications, theta, box, em)
    
    assert len(spots) == len(locs)
    
    return spots, locs

#%%
#### Load raw movie
movie, info = io.load_movie(path)

#### Get one image of movie at frame
frame=100
image=movie[frame]
#### Detect spots in image 
box=5
mng=400
y,x,ng= localize.identify_in_frame(image,mng,box,None)

#### Show spots
view_x=info[0]['Height']/2
view_y=info[0]['Height']/2
view_width=info[0]['Height']/2

import matplotlib.pyplot as plt
f=plt.figure(num=12,figsize=[7,7])
f.subplots_adjust(bottom=0.,top=1.,left=0.,right=1.)
f.clear()
ax=f.add_subplot(111)
ax.imshow(image,cmap='gray',vmin=100,vmax=200,interpolation='nearest')
ax.scatter(x,y,s=100,marker='o',alpha=1,facecolor='None',edgecolor='w',linewidths=1)
ax.set_xlim(view_x,(view_x+view_width))
ax.set_ylim(view_y,(view_y+view_width))
ax.grid(False)
plt.show()