from yeastvision.models.cp import CustomCPWrapper
from yeastvision.utils import resize_image_OAM, divide_and_round_up, resize_image_scipy, rescaleBySize
import numpy as np
import torch
from skimage.io import imread 
import cv2

class MatSeg(CustomCPWrapper):
    hyperparams  = { 
    "mean_diameter":30,
    "flow_threshold":0.4, 
    "cell_probability_threshold": 0.5,
    "half_image_size": True}
    types = [float, float, float, bool]
    
    def __init__(self, params, weights):
        super().__init__(params, weights)
    
    @classmethod
    @torch.no_grad()
    def run(cls, ims, params, weights):
        if params["half_image_size"]:
            original_size = ims[0].shape # (rows, cols)            
            ims_downsized = [resize_image_scipy(im, 0.5) for im in ims]
            out_small = super().run(ims_downsized, params, weights) # run matSeg on downsized images
            out =  tuple(np.array(rescaleBySize(original_size, ims, interpolation=cv2.INTER_NEAREST_EXACT)) for ims in out_small)
            return out
        else:
            return super().run(ims, params, weights)