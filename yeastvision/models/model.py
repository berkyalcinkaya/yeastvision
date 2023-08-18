from os.path import join
import glob
from audioop import avg
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
import skimage
import skimage.measure
import skimage.morphology
import tensorflow as tf
from skimage import io
from patchify import patchify, unpatchify
import os
from os.path import join
import glob
import json
from tensorflow.keras.optimizers import Adam
import pandas as pd
from tensorflow.keras.models import model_from_json
from skimage.io import imsave, imread
import shutil
from skimage.morphology import binary_erosion
from skimage.morphology import disk  # noqa
from skimage.io import imread,imsave
from skimage.measure import label
from skimage.exposure import equalize_adapthist as ea
from scipy.signal import medfilt2d
from yeastvision.utils import *
from yeastvision.models.utils import jaccard_coef_loss, MODEL_DIR, patchify_for_train
import tensorflow as tf

class Model():
    hyperparams = {"Threshold":0.50}
    loss = jaccard_coef_loss
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
        weightPath = join(params["dir"], "models", params["model_name"])
        ims = [self.preprocess(im) for im in ims]
        masks = [self.preprocess_masks(mask) for mask in masks]
        ims = patchify_for_train(ims)
        masks = patchify_for_train(masks)
        ims = np.array(ims)
        masks = np.array(masks)
        model = self.model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
                loss=self.loss)
        model.fit(ims, masks,
            batch_size=batchsize,
            epochs=int(params["n_epochs"]))

        # Save the trained model (optional)
        print(weightPath)
        model.save(weightPath)

    @classmethod
    def run(cls, ims, params, weights):
        params = params if params else cls.hyperparams
        model = cls(params, weights)
        probIms, masks = [], []
        for im in ims:
            probIm = model.getProbIm(im)
            probIms.append(probIm)
            masks.append(model.getMask(probIm))
        del model.model
        del model
        return np.array(masks, dtype = np.uint8), np.array(probIms, dtype = np.float32) 


