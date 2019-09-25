import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Test if GPU fitting can be used
try:
    from pygpufit import gpufit as gf

    gpufit_installed = True
except ImportError:
    gpufit_installed = False
    
#%%
def _spots_in_image(image,mng,box,fit=False):
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
def _image_preview(image,view=None,contrast_min=100,contrast_max=200,fignum=12):
    '''

    '''
    f=plt.figure(num=fignum,figsize=[4,4])
    f.subplots_adjust(bottom=0.,top=1.,left=0.,right=1.)
    f.clear()
    ax=f.add_subplot(111)
    
    ax.imshow(image,cmap='gray',vmin=contrast_min,vmax=contrast_max,interpolation='nearest')   
    ax.grid(False)
    
    if view is None:
        print('No viewport')
    else:
        ax.set_xlim(view[0][0]-view[1][0]/2,
                    view[0][0]+view[1][0]/2)
        ax.set_ylim(view[0][1]-view[1][1]/2,
                    view[0][1]+view[1][1]/2)
  
    return ax

#%%
def _spots_preview(spots,ax):
    ax.scatter(spots.x,spots.y,s=50,marker='o',alpha=1,facecolor='None',edgecolor='y',linewidths=1)

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
    em = camera_info['gain'] > 1
    if gpufit_installed:
        print('GPU fitting ...')
        theta = gausslq.fit_spots_gpufit(spots)
        locs=gausslq.locs_from_fits_gpufit(identifications, theta, box, em)
    else:
        print('Fitting ...')
        fs = gausslq.fit_spots_parallel(spots, asynch=True)
        theta = gausslq.fits_from_futures(fs)   
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

#%%
def _get_picked(locs,centers,pick_diameter=2):
    '''
    Parameters
    ---------
    locs : numpy.recarray
        locs as created by picasso.localize
    centers: np.array of shape (m,2)
        x and y pick center positions
    pick_diameter: float
        Pick diameter in px
    Returns
    -------
    locs_picked: numpy.recarray
        locs_picked file as created by picasso.render 
    '''
    from scipy.spatial import cKDTree
    
    #### Prepare data for KDtree
    data=np.vstack([locs.x,locs.y]).transpose()
    
    #### Build up KDtree
    tree=cKDTree(data,compact_nodes=False,balanced_tree=False)

    #### Query KDtree for indices belonging to picks
    picks_idx=tree.query_ball_point(centers,r=pick_diameter,p=1)

    #### Init locs_picked as DataFrame
    locs_picked=pd.DataFrame(locs).assign(group=np.nan)
    
    #### Assign group index
    groups=np.full(len(locs),np.nan)
    for g,idx in enumerate(picks_idx): groups[idx]=g
    locs_picked['group']=groups
    
    #### Drop all unassigned localizations (NaNs)
    locs_picked=locs_picked.loc[np.isfinite(locs_picked.group),:]
    
    #### Convert to right dtype
    locs_picked=locs_picked.astype({'group':np.uint32})
    
    #### Sort
    locs_picked.sort_values(['group','frame'],inplace=True)
    
    return locs_picked




