from tensorflow.keras.models import load_model
import glob
import os
from yeastvision.models.utils import prediction, normalizeIm, produce_weight_path


class BudSeg():
    def __init__(self):
        modelPath = produce_weight_path("artilife/budSeg","Bud_Seg.h5")
        self.model = load_model(modelPath)

    def prediction(self, ims):
        if type(ims) != list:
            ims =[ims]
        return [prediction(self.model, im, input_shape = 50, 
                            thresh = 0.50, preprocess=normalizeIm, do_aug = False) 
                            for im in ims]
        