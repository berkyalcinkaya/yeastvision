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

class Model():
    hyperparams = {"Threshold":0.50}
    types = [None]
    prefix = ""

    def __init__(self, params, weights):
        self.params = params 
        self.weights = weights
    
    def preprocess(self,im):
        return im
    
    def getProbIm(self,im):
        return im
    
    def getMask(self, probIm):
        return (probIm>self.params["Threshold"]).astype(np.uint8)

    @classmethod
    def run(cls, ims, params, weights):
        params = params if params else cls.hyperparams
        model = cls(params, weights)
        probIms, masks = [], []
        for im in ims:
            probIm = model.getProbIm(im)
            probIms.append(probIm)
            masks.append(model.getMask(probIm))
        return np.array(masks, dtype = np.uint8), np.array(probIms, dtype = np.float32) 


