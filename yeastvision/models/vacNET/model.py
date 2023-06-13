import os
import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.measure import label
from models.budNET.model import BudNET
from models.utils import prediction
from models.loss import jaccard_coef_loss, jaccard_coef
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


class VacNET(BudNET):
    def __init__(self, params, weights):
        super().__init__(params, weights)
    