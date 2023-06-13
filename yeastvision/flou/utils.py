import numpy as np
from skimage.draw import disk
from skimage.feature import blob_log as bl
from skimage.exposure import equalize_adapthist
import math


def normalize_im(im, clip = True):
    """
    Normalizes a given image such that the values range between 0 and 1.     
    
    Parameters
    ---------- 
    im : 2d-array
        Image to be normalized.
    clip: boolean
        Whether or not to set im values <0 to 0
        
    Returns
    -------
    im_norm: 2d-array
        Normalized image ranging from 0.0 to 1.0. Note that this is now
        a floating point image and not an unsigned integer array. 
    """ 
    assert not np.any(np.isnan(im))
    if clip:
        im[im<0] = 0
    if im.max()==0:
        return im.astype(np.float32)
    im_norm = (im - im.min()) / (im.max() - im.min())
    return im_norm.astype(np.float32)


def draw_blobs(ims, min_size, max_size, threshold):
    masks = np.zeros_like(ims)
    for idx, im in enumerate(ims):
        im = equalize_adapthist(im)
        blobs = bl(ims, min_sigma = min_size, max_sigma = max_size, threshold = threshold)
        blobs[:, 2] = blobs[:, 2] * math.sqrt(2)

        label = 1
        for blob in blobs:
            r,c,radius = blob
            rr, cc = disk((int(r),int(c)), int(radius), shape = im.shape)
            masks[idx, rr, cc] =  label
            label+=1
    return masks

