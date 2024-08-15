#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:42:57 2024

@author: samarth
"""

import numpy as np
from skimage.measure import regionprops

def get_wind_coord1(Ihere, cell_margin):
    x_size, y_size = Ihere.shape
    
    if np.sum(Ihere) > 0:  # not empty object
        s_f = regionprops(Ihere)
        if s_f:  # not empty
            bbox = np.round(s_f[0].bbox).astype(int)
            lower_x_limit = max(0, bbox[1] - cell_margin)
            upper_x_limit = min(y_size, bbox[3] + cell_margin)
            lower_y_limit = max(0, bbox[0] - cell_margin)
            upper_y_limit = min(x_size, bbox[2] + cell_margin)
            x_cn = np.arange(lower_x_limit, upper_x_limit)
            y_cn = np.arange(lower_y_limit, upper_y_limit)
        else:
            print('empty or multiple object given - error')
            x_cn, y_cn = None, None
    else:
        x_cn = np.arange(y_size)
        y_cn = np.arange(x_size)
    
    return x_cn, y_cn
