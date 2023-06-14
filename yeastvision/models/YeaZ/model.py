import os
import numpy as np
from skimage.exposure import equalize_adapthist
from .segment import segment
import skimage
from yeastvision.models.model import Model as CustomModel
import matplotlib.pyplot as plt
from .unet import unet

class YeaZ(CustomModel):
    hyperparams = {"Threshold":0.50, "Minimum Cell Distance":5}
    types = [None, None]
    loss = "binary crossentropy"

    def __init__(self, params, weights):
        super().__init__(params, weights)
        self.model = self.load()

    def load(self):  
        # WHOLE CELL PREDICTION
        model = unet(pretrained_weights = self.weights+".hdf5",
                    input_size = (None,None,1))
        return model
    
    def getProbIm(self, im):
        im = equalize_adapthist(im)
        (nrow, ncol) = im.shape
        row_add = 16-nrow%16
        col_add = 16-ncol%16
        padded = np.pad(im, ((0, row_add), (0, col_add)))
        results = self.model(padded[np.newaxis,:,:,np.newaxis])
        res = np.array(results[0,:,:,0])
        print(res.max())
        return res[:nrow, :ncol]
    
    def threshold(self, im):
        """
        Binarize an image with a threshold given by the user, or if the threshold is None, calculate the better threshold with isodata
        Param:
            im: a numpy array image (numpy array)
            th: the value of the threshold (feature to select threshold was asked by the lab)
        Return:
            bi: threshold given by the user (numpy array)
        """
        th = float(self.params["Threshold"])
        im2 = np.array(im).copy()
        if th == None:
            th = skimage.filters.threshold_isodata(im2)
        bi = im2
        bi[bi > th] = 255
        bi[bi <= th] = 0
        return bi

    def getMask(self, pred):
        th = self.threshold(pred)
        seg  = segment(th,pred, min_distance = int(self.params["Minimum Cell Distance"]))
        return seg








