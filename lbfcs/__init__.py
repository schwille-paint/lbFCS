"""
    lbfcs/__init__.py
    ~~~~~~~~~~~~~~~~~~~~

    :authors: Florian Stehr 2019
"""
import numpy as np


def snr_from_conc(conc):
    '''
    Returns signal to noise ratio [a.u.], i.e. amplitude of gaussian spot divided by background fluorescence for specific concentration [M].
    The snr values correspond to empirical values as measured on D042 TIRF setup for 5XCTC docking strand (kon = 30e6 1/Ms) and Pm2Cy3B imager sequence.
    The function is designed to give some exemplary snr values for the lbfcs simulation module.
    '''
    kon = 30e6
    a = 13.014
    b = 0.077
    c = 3.503
    
    snr = a * np.exp(-(kon*conc)/b) + c
    
    return snr