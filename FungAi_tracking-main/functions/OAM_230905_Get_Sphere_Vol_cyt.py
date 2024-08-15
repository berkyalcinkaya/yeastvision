#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:49:40 2024

@author: samarth
"""

import numpy as np
from skimage.measure import label, regionprops

"""
TODO: requires discussion
"""

def OAM_230905_Get_Sphere_Vol_cyt(mask_cyt):
    """
    Uses a binary mask of a cell or cellular structures as a matrix to calculate the volume of a sphere with the same equivalent diameter.
    """
    mask_cyt1 = label(mask_cyt)

    Spherical_vol_cyt = 0
    for it in range(1, np.max(mask_cyt1) + 1):
        Ic = (mask_cyt1 == it)
        ed = regionprops(Ic.astype(int), 'equivalent_diameter')
        Spherical_vol_cyt += 0.523 * (ed[0].equivalent_diameter ** 3)
    
    return Spherical_vol_cyt
