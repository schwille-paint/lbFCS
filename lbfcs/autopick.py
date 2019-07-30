import numpy as np

def _autopick(image,mng,box,fit=False):
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