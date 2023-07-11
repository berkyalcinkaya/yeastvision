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
import yeastvision.ims.im_funcs as im_funcs
import math
import pickle
from skimage.io import imread, imsave
from cellpose.metrics import average_precision
from tqdm import tqdm
import time
from PyQt5.QtCore import Qt, QThread, QMutex


class SegmentWorker(QtCore.QObject):
    '''Handles Multithreading'''
    finished = QtCore.pyqtSignal(object, object, object, object, object, object)
    def __init__(self, modelClass,ims, params, weightPath, modelType):
        super(SegmentWorker,self).__init__()
        self.mc = modelClass
        self.ims = ims
        self.params = params
        self.weight = weightPath
        self.mType = modelType

    def run(self):
        row, col = self.ims[0].shape
        newImTemplate = np.zeros((len(self.ims), row, col))
        tStart, tStop = int(self.params["T Start"]), int(self.params["T Stop"])

        output = self.mc.run(self.ims[tStart:tStop+1],self.params, self.weight)
        self.finished.emit(output, self.mc,newImTemplate, self.params, self.weight, self.mType)

class TrackWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, object)

    def __init__(self, func, cells, z, obj = None):
        super(TrackWorker,self).__init__()
        self.trackfunc = func
        self.z = z
        self.cells = cells
        self.obj = obj
    
    def run(self):
        if self.obj is not None:
            out =  self.trackfunc(self.obj, self.cells)
        else:
            out = self.trackfunc(self.cells)
        self.finished.emit(self.z,out)

class InterpolationWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, object, object, object)

    def __init__(self, ims, files, newname, func, outdtype, *args):
        super(InterpolationWorker, self).__init__()
        self.ims = ims
        self.files = files
        self.name = newname
        self.func = func
        self.funcargs = args
        self.outtype = outdtype

    def run(self):
        out = self.func(self.ims, *self.funcargs)
        self.finished.emit(out, self.files, self.name, self.outtype)

