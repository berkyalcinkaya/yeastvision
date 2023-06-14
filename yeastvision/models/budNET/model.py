import os
import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.measure import label
from yeastvision.models.model import Model
from yeastvision.models.utils import prediction
from yeastvision.models.loss import jaccard_coef_loss, jaccard_coef
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tqdm import tqdm


class BudNET(Model):
    hyperparams = {"Threshold":0.50,
                    "Test Time Augmentation": True}
    types = [None, bool]
    prefix = ".h5"
    loss = "jaccard coefficient"

    def __init__(self, params, weights):
        super().__init__(params, weights)
        custom_objects = {"jaccard_coef_loss":jaccard_coef_loss,
                        "jaccard_coef": jaccard_coef}
        self.model = load_model(weights, custom_objects = custom_objects)
    
    @classmethod
    def run(cls, ims, params, weights):
        params = params if params else cls.hyperparams
        model = cls(params, weights)

        probIms = [prediction(model.model, im, do_thresh=False, do_aug = model.params["Test Time Augmentation"]) for im in tqdm(ims)]
        threshIms = [label((im>model.params["Threshold"]).astype(np.uint16)) for im in probIms]

        return np.array(threshIms, dtype = np.uint16), np.array(probIms, dtype = np.float32)



    
    
    
