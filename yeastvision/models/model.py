from os.path import join
import numpy as np
from os.path import join
from skimage.exposure import equalize_adapthist as ea
from yeastvision.utils import *
from yeastvision.models.utils import MODEL_DIR, patchify_for_train
import tqdm

class Model():
    hyperparams = {"Threshold":0.50}
    loss = None
    trainparams = {"learning_rate": 0.001, "n_epochs": 100}
    types = [None]
    prefix = ""

    def __init__(self, params, weights):
        self.params = params 
        self.weights = weights
    
    def getProbIm(self,im):
        return im
    
    def getMask(self, probIm):
        return (probIm>self.params["Threshold"]).astype(np.uint8)
    
    def preprocess(self, im):
        return ea(im).astype(np.float32)
    
    def preprocess_masks(self, mask):
        return (mask>0).astype(np.float32)
    
    def name(self):
        pass
    
    def train(self, ims, masks, params):
        batchsize = 8
        weightPath = join(params["dir"], "models", params["model_name"])+".h5"
        ims = [self.preprocess(im) for im in ims]
        masks = [self.preprocess_masks(mask) for mask in masks]
        ims = patchify_for_train(ims)
        masks = patchify_for_train(masks)
        ims = np.array(ims)
        masks = np.array(masks)
        model = self.model
        model.fit(ims, masks,
            batch_size=batchsize,
            epochs=int(params["n_epochs"]))

        # Save the trained model (optional)
        model.save(weightPath)

    @classmethod
    def run(cls, ims, params, weights):
        params = params if params else cls.hyperparams
        model = cls(params, weights)
        probIms, masks = [], []
        for im in tqdm(ims):
            probIm = model.getProbIm(im)
            probIms.append(probIm)
            masks.append(model.getMask(probIm))
        del model.model
        del model


        masks, probIms =  np.array(masks, dtype = np.uint16), (np.array(probIms)*255).astype(np.uint8)
        return masks, probIms



