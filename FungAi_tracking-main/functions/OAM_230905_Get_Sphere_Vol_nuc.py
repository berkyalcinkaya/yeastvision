#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:51:53 2024

@author: samarth
"""

import numpy as np
from skimage.measure import label, regionprops

def OAM_230905_Get_Sphere_Vol_nuc(mask_nuc):
    """
    Uses a binary mask of a cell or cellular structures as a matrix to calculate the volume of a sphere with the same equivalent diameter.
    """
    mask_nuc = label(mask_nuc)
    max_label = np.max(mask_nuc)
    
    if max_label == 0:
        Spherical_vol_nuc = np.nan
    elif max_label > 1:
        Spherical_vol_nuc = 0
        for it in range(1, max_label + 1):
            Ic = (mask_nuc == it)
            ed = regionprops(Ic.astype(int), 'equivalent_diameter')[0]
            Spherical_vol_nuc += 0.523 * (ed.equivalent_diameter ** 3)
    else:
        ed = regionprops(mask_nuc, 'equivalent_diameter')[0]
        Spherical_vol_nuc = 0.523 * (ed.equivalent_diameter ** 3)
    
    return Spherical_vol_nuc
