from yeastvision.models.cp import CustomCPWrapper
from yeastvision.utils import resize_image_scipy, rescaleBySize
from yeastvision.track.fiest.utils import resize_image
import numpy as np
import torch

class MatSeg(CustomCPWrapper):
    hyperparams  = { 
    "mean_diameter":30,
    "flow_threshold":0.4, 
    "cell_probability_threshold": 0.0,
    "half_image_size": True}
    types = [float, float, float, bool]
    
    def __init__(self, params, weights):
        super().__init__(params, weights)
    
    @classmethod
    @torch.no_grad()
    def run(cls, ims, params, weights):
        if params["half_image_size"]:
            original_size = ims[0].shape
            ims_resized = [resize_image_scipy(im, 0.5) for im in ims]
            out_small = super().run(ims_resized, params, weights)
            out =  tuple(np.array([resize_image(im, original_size) for im in ims]) for ims in out_small)
            return out
        else:
            return super().run(ims, params, weights)