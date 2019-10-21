import matplotlib.pyplot as plt

#%%
def _image_preview(image,contrast_min=100,contrast_max=200,fignum=12):
    '''

    '''
    f=plt.figure(num=fignum,figsize=[4,4])
    f.subplots_adjust(bottom=0.,top=1.,left=0.,right=1.)
    f.clear()
    ax=f.add_subplot(111)
    
    ax.imshow(image,cmap='gray',vmin=contrast_min,vmax=contrast_max,interpolation='nearest',origin='lower')   
    ax.grid(False)
    
    return ax

#%%
def _spots_preview(spots,ax,msize=50):
    ax.scatter(spots.x,spots.y,s=msize,marker='o',alpha=1,facecolor='None',edgecolor='y',linewidths=1)
    
#%%
def ax_styler(ax):
    ax.xaxis.set_tick_params(top='off')
    ax.yaxis.set_tick_params(right='off')