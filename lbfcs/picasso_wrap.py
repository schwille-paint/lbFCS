import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Test if GPU fitting can be used
try:
    from pygpufit import gpufit as gf
    gpufit_available=True
    gpufit_available=gf.cuda_available()
except ImportError:
    gpufit_available = False

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
    if gpufit_available:
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
def _spots_in_image(image,mng,box,fit=False):
    '''
    Identify spots in image and return center of spots.
    
    Parameters
    ---------
    image : np.array of shape (m,n)
    mng: int
        Minimal net gradient for spot detection as in picasso.localize
    box: uneven int
        Box size (uneven integer) for spot detections as in picasso.localize
    fit: bool
        If 'True' spots will be fitted by 2D gaussian to define center, if 'False' centers will be center of mass position of box. Defaults to 'False'. 
    Returns
    -------
    spot_centers: pandas.DataFrame
        Spot centers of identified spots. Columns are 'x', 'y' and  'ng' corresponding to spot center postittion and net gradient.
    '''
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
    identifications=np.rec.array((1*np.ones(N), x, y, ng),
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
    
    spot_centers=pd.DataFrame(locs)
    spot_centers.drop(columns=['frame'],inplace=True)    
    
    return spot_centers

#%%
def _coordinate_convert(spot_centers,viewport_min,oversampling):
    '''
    Convert spot centers as detected in rendered localizations by given viewport minima and oversampling back to original localization coordinates.
    
    Parameters
    ---------
    spot_centers : pandas.DataFrame
        Output of pic_wrap._spots_in_image function.
    viewport_min: tuple of len=2
        First tuple of viewport as in picasso.render: viewport=[(min_y,min_x),(max_y,max_x)]
    oversampling: int
        Pixel oversampling as in picasso.render
    
    Returns
    -------
    spot_centers_convert: pandas.DataFrame
        Spot centers converted back to orignal localization coordinates.
    
    '''
    spot_centers_convert=spot_centers.copy()
    spot_centers_convert.x=spot_centers.x/oversampling+viewport_min[1]
    spot_centers_convert.y=spot_centers.y/oversampling+viewport_min[0]
    
    return spot_centers_convert
#%%
def _query_locs_for_centers(locs,centers,pick_radius=0.8):
    '''
    Builds up KDtree for locs, queries the tree for localizations within pick_radius (norm of p=2) aorund centers (xy coordinates).
    Output will be list of indices as result of query.
    Parameters
    ---------
    locs : numpy.recarray or pandas.DataFrame
        locs as created by picasso.localize with fields 'x' and 'y'
    centers: np.array of shape (m,2)
        x and y pick center positions
    pick_diameter: float
        Pick diameter in px
    Returns
    -------
    picks_idx: list
        List of len(centers). Single list entries correpond to indices in locs within pick_radius around centers.
    '''
    from scipy.spatial import cKDTree
    
    #### Prepare data for KDtree
    data=np.vstack([locs.x,locs.y]).transpose()
    #### Build up KDtree
    tree=cKDTree(data,compact_nodes=False,balanced_tree=False)
    #### Query KDtree for indices belonging to picks
    picks_idx=tree.query_ball_point(centers,r=pick_radius,p=2)
    
    return picks_idx
#%%
def _get_picked(locs,picks_idx,field='group'):
    
    locs_picked=locs.copy()

    #### Assign group index
    groups=np.full(len(locs),-1)
    for g,idx in enumerate(picks_idx): groups[idx]=g
    
    locs_picked[field]=groups
    
    return locs_picked
