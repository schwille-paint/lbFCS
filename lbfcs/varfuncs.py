import numpy as np

#%%
def ac_monoexp(l,A,tau):
    """
    Fit function for mono-exponential fit of autocorrelation function: ``g(l)=A*exp(-l/tau)+1``
    """
    g=A*np.exp(-l/tau)+1.
    return g

#%%
def ac_monoexp_lin(l,A,tau):
    '''
    Linearized fit function for mono-exponential fit of autocorrelation function: ``-log(g(l)-1)=l/tau-log(A)``
    '''
    g=l/tau-np.log(A)
    return g

#%%
def ac_biexp(t,A1,tau1,A2,tau2):
    """
    Fit function for bi-exponential fit of autocorrelation function:
        g(t)=A1*exp(-t/tau1)+A2*exp(-t/tau2)+1
    """
    g=A1*np.exp(-t/tau1)+A2*np.exp(-t/tau2)+1.
    return g

#%%
def get_ecdf(x):
    """
    Calculate experimental continuous distribution function (ECDF) of random variable x
    so that counts(value)=probability(x<=value). I.e last value of counts=1.
    
    Equivalent to :
        matplotlib.pyplot.hist(tau_dist,bins=numpy.unique(tau_dist),normed=True,cumulative=True)
    
    Parameters
    ---------
    x : numpy.ndarray
        1 dimensional array of random variable  
    Returns
    -------
    values : numpy.ndarray
         Bins of ECDF corresponding to unique values of x.
    counts : numpy.ndarray
        counts(value)=probability(x<=value).
    """
    values,counts=np.unique(x,return_counts=True) # Give unique values and counts in x
    counts=np.cumsum(counts) # Empirical cfd of dist without bias from binning
    counts = counts/counts[-1] # normalize that sum(counts) = 1
    return (values,counts)

#%%
def ecdf_exp(t,tau,off=0,A=1):
    """
    Fit function for exponential fit of ECDF:
        ecdf=1-np.exp(-t/tau)
    """
    ecdf=A*(1-np.exp(-t/tau))+off
    return ecdf


#%%
def gaussian_2D(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple                                                        
    xo = float(xo)                                                              
    yo = float(yo)                                                              
    a = (np.cos(theta)**2)/(1*sigma_x**2) + (np.sin(theta)**2)/(1*sigma_y**2)   
    b = -(np.sin(2*theta))/(2*sigma_x**2) + (np.sin(2*theta))/(2*sigma_y**2)    
    c = (np.sin(theta)**2)/(1*sigma_x**2) + (np.cos(theta)**2)/(1*sigma_y**2)   
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)         
                        + c*((y-yo)**2)))                                   
    return g.ravel()

#%%
#### Hybridization kinetics for c-series
def tau_of_c(c,koff,kon):
    tauc=1/(koff+kon*c)
    return tauc

def Ainv_of_c(c,slope,offset=0.):
    Ainv=slope*c+offset
    return Ainv

def taudinv_of_c(c,slope,offset=0.):
    taudinv=slope*c+offset
    return taudinv




