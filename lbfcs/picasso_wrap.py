import numpy as np

#%%
def _spot_in_image(image,mng,box,fit=False):
    ### Import
    import picasso.localize
    import picasso.gausslq 

    #### Identify spots in image
    y,x,ng= picasso.localize.identify_in_frame(image,mng,box,None)
    
    #### Remove spots closer than int(box/2) to image boarder
    crit_d=int(box/2)
    im_size_x=np.shape(image)[0]
    im_size_y=np.shape(image)[1]
    
    y=y[(y>crit_d)|((im_size_y-y)>crit_d)]
    x=x[(x>crit_d)|((im_size_x-x)>crit_d)]
    
    #### Prepare identifactions   
    N=len(x) # Number of detected spots
    identifications=np.rec.array( # Prepare output
            (1*np.ones(N), x, y, ng),
            dtype=[("frame", "i"), ("x", "i"), ("y", "i"), ("net_gradient", "f4")])
    
    if fit==True:
        #### Cut spots
        spots = np.zeros((N, box, box), dtype=image.dtype) # Initialize spots
        picasso.localize._cut_spots_frame(image,0,np.zeros(N),x,y,int(box/2),0,N,spots)
        spots=spots.astype(float) # Convert to float for fitting

        #### Fit spots
        theta=picasso.gausslq.fit_spots_parallel(spots,asynch=False)

        #### Convert fit results to localizations
        locs=picasso.gausslq.locs_from_fits(identifications, theta, box, 1)
    else:
        locs=identifications.copy()
        locs.x=x
        locs.y=y
        locs.ng=ng
        
    return locs


#%%
def _spot_preview(movie,info,frame,box,mng,contrast_min=100,contrast_max=200):
    '''

    '''
    #### Get image of movie at frame
    image=movie[frame]
    
    #### Detect spots in image 
    locs=_spot_in_image(image,mng,box,fit=False)
    
    #### Set viewing dimensions
    view_x=info[0]['Width']/2
    view_y=info[0]['Height']/2
    view_width=info[0]['Height']/4
    
    import matplotlib.pyplot as plt
    f=plt.figure(num=12,figsize=[4,4])
    f.subplots_adjust(bottom=0.,top=1.,left=0.,right=1.)
    f.clear()
    ax=f.add_subplot(111)
    ax.imshow(image,cmap='gray',vmin=contrast_min,vmax=contrast_max,interpolation='nearest')
    ax.scatter(locs.x,locs.y,s=50,marker='o',alpha=1,facecolor='None',edgecolor='y',linewidths=1)    
    ax.set_xlim((view_x-view_width),(view_x+view_width))
    ax.set_ylim((view_y-view_width),(view_y+view_width))
    ax.grid(False)
    plt.show()
    
    return ax

#%%
def _render_preview(locs,info,oversampling=2,blur_method=None,min_blur_width=0.02,contrast_min=100,contrast_max=200,fignum=13):
    '''

    '''
    import picasso.render as render
    
    #### Define viewport
    view_x=info[0]['Height']/2
    view_y=info[0]['Width']/2
    view_width=info[0]['Width']/4
    viewport=[(view_x-view_width,view_y-view_width),(view_width*2,view_width*2)]
    
    #### Render image
    n_locs,locs_render=render.render(
        locs,
        info,
        oversampling=oversampling,
        viewport=viewport,
        blur_method=blur_method,
        min_blur_width=min_blur_width
        )

    import matplotlib.pyplot as plt
    f=plt.figure(num=fignum,figsize=[4,4])
    f.subplots_adjust(bottom=0.,top=1.,left=0.,right=1.)
    f.clear()
    ax=f.add_subplot(111)
    ax.imshow(locs_render,cmap='magma',vmin=contrast_min,vmax=contrast_max,interpolation='nearest')
    ax.grid(False)
    plt.show()
    
    return ax
#%%
def _localize_movie(movie,box=5,mng=400,):
    '''
    
    '''
    import picasso.localize as localize
    import picasso.gausslq as gausslq
    #### Set camera specs for photon conversion
    camera_info = {}
    camera_info['baseline'] = 70
    camera_info['gain'] = 1
    camera_info['sensitivity'] = 0.56
    camera_info['qe'] = 0.82
    
    #### Spot detection
    print('Identifying spots ...')
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
def _locs2picks(locs,pick_diameter,path):
    x=locs.x
    y=locs.y
    centers = []
    for index, element in enumerate(range(len(x))):
        centers.append([x[index],y[index]])

    picks = {'Diameter': pick_diameter, 'Centers': centers}
 
    import yaml  as yaml
    with open(path, 'w') as f:
        yaml.dump(picks, f)
        
