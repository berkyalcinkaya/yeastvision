import torch
import numpy as np
from time import process_time
tic = process_time()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QApplication, QStyle, QMainWindow, QGroupBox, QPushButton, QDialog,
                            QDialogButtonBox, QLineEdit, QFormLayout, QMessageBox,QErrorMessage, QStatusBar, 
                             QFileDialog, QVBoxLayout, QCheckBox, QFrame, QSpinBox, QLabel, QWidget, QComboBox, QSizePolicy, QGridLayout)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np
import matplotlib.pyplot as plt
from yeastvision.parts.canvas import ImageDraw, ViewBoxNoRightDrag
from yeastvision.parts.guiparts import *
from yeastvision.parts.dialogs import *
from yeastvision.track.track import track_to_cell, trackYeasts
from yeastvision.track.data import LineageData
from yeastvision.track.lineage import LineageConstruction
from yeastvision.track.cell import LABEL_PROPS, IM_PROPS, EXTRA_IM_PROPS, getCellData, exportCellData, getHeatMaps, getDaughterMatrix
import cv2
from yeastvision.disk.reader import loadPkl, ImageData, MaskData
import importlib
import yeastvision.parts.menu as menu
from yeastvision.models.utils import MODEL_DIR
import glob
from os.path import join
from datetime import datetime
import pandas as pd
import yeastvision.plot.plot as plot
from yeastvision.flou.blob_detect import Blob
from yeastvision.utils import *
import yeastvision.ims as im_funcs
import math
import pickle
from skimage.io import imread, imsave
from cellpose.metrics import average_precision
from tqdm import tqdm


class SegmentWorker(QtCore.QObject):
    '''Handles Multithreading'''
    finished = QtCore.pyqtSignal(object, object, object, object, object, object)
    def __init__(self, parent):
        super(SegmentWorker,self).__init__()
        self.parent = parent
    
    def run(self, modelClass,ims, params, weightPath, modelType):
        row, col = ims[0].shape
        newImTemplate = np.zeros((len(ims), row, col))
        outputTup  = (newImTemplate, newImTemplate.copy())
        tStart, tStop = int(params["T Start"]), int(params["T Stop"])

        output = modelClass.run(ims[tStart:tStop+1],params, weightPath)
        self.finished.emit(output, modelClass,ims, params, weightPath, modelType)

class TrackWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, object)

    def __init__(self, parent, func, z):
        self.parent = parent
        self.trackfunc = func
        self.z
    
    def run(self, labels):
        self.finished.emit(self.func(labels))

# class ImageWorker(QtCore.QObject):

#     finished = QtCore.pyqtSignal(object, object)

#     def __init__(self, parent, func, z):
