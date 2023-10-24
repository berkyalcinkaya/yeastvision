#https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
import torch
import numpy as np
from time import process_time
tic = process_time()
import os
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QApplication, QStyle, QMainWindow, QGroupBox, QPushButton, QDialog,
                            QDialogButtonBox, QLineEdit, QFormLayout, QMessageBox,QErrorMessage, QStatusBar, 
                             QFileDialog, QVBoxLayout, QCheckBox, QFrame, QSpinBox, QLabel, QWidget, QComboBox, QSizePolicy, QGridLayout)
from PyQt5.QtCore import Qt, QThread, QMutex, pyqtSignal
from QSwitchControl import SwitchControl
import pyqtgraph as pg
import numpy as np
import matplotlib.pyplot as plt
from yeastvision.parts.canvas import ImageDraw, ViewBoxNoRightDrag
from yeastvision.parts.guiparts import *
from yeastvision.parts.workers import SegmentWorker, TrackWorker, InterpolationWorker
from yeastvision.parts.dialogs import *
from yeastvision.track.track import track_to_cell, trackYeasts
from yeastvision.track.data import LineageData, TimeSeriesData
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
from yeastvision.ims.interpolate import interpolate
import math
import pickle
from skimage.io import imread, imsave
from cellpose.metrics import average_precision
from tqdm import tqdm
from memory_profiler import profile
from functools import partial
from yeastvision.models.artilife.model import ArtilifeFullLifeCycle
from yeastvision.data.ims import Experiment, ChannelNoDirectory, InterpolatedChannel
torch.cuda.empty_cache() 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import warnings
import copy
#warnings.filterwarnings("ignore")
configure_tf_memory_growth()
from collections import OrderedDict

# global logger
# logger, log_file = logger_setup()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, thread, imPaths = None, labelPaths = None):
        super(MainWindow, self).__init__()
        pg.setConfigOption('imageAxisOrder', 'row-major')
        self.setGeometry(100, 100, 900, 1000)
        self.setAcceptDrops(True)

        #self.idealThreadCount  = thread.idealThreadCount()//2
        self.idealThreadCount = 2

        self.sessionId = self.getCurrTimeStr()
        
        self.goldColor = [255,215,0,255]

        self.setStyleSheet("QMainWindow {background-color: rgb(24,25,26);}")
        self.stylePressed = ("QPushButton {Text-align: left; "
                             "background-color: rgb(200,10,10); "
                             "border-color: white;"
                             "color:white;}")
        self.styleCheckable = ("QPushButton {Text-align: left; "
                               "background-color: rgb(50,50,50); "
                                "border-color: white;"
                               "color:white;}"
                            "QPushButton:checked {Text-align: left; "
                             "background-color: rgb(150,50,150); "
                             "border-color: white;"
                             "color:white;}"
                             "QPushButton:checked")
        self.styleUnpressed = ("QPushButton {Text-align: left; "
                               "background-color: rgb(50,50,50); "
                                "border-color: white;"
                               "color:white;}")
        self.styleInactive = ("QPushButton {Text-align: left; "
                              "background-color: rgb(30,30,30); "
                             "border-color: white;"
                              "color:rgb(80,80,80);}")
        self.firstMaskLoad = True
        self.getModelNames()
        self.cwidget = QWidget(self)
        self.l = QGridLayout()
        self.cwidget.setLayout(self.l)
        self.setCentralWidget(self.cwidget)
        self.l.setSpacing(1)

        self.win = pg.GraphicsLayoutWidget()
        self.mainViewRows = 30
        self.mainViewCols  = 21
        self.l.addWidget(self.win, 0,0, self.mainViewRows, self.mainViewCols)
        self.make_viewbox()

        self.labelX, self.labelY = None, None
        self.currMaskDtype = "uint8"
        self.currImDtype = "uint8"

        self.threads = []
        self.workers = []
        
        self.setEmptyDisplay(initial=True)
        self.build_widgets()
        self.l.setHorizontalSpacing(15)
        self.l.setVerticalSpacing(15)
        
        self.drawType = ""
        self.cellToChange = 0

        self.annotationList = []

        self.addRegionStrokes = []
        self.currPointSet = []

        menu.menubar(self)

        self.cellDataFormat = { "Cell":[],
                                "Birth":[],
                                "Death":[]}
        self.cellLineageDataFormat = {"Mother":[],
                                    "Daughters": []}
        
        self.plotTypes = ["single", "population", "heatmap"]
        self.populationPlotTypes = ["population", "heatmap"]
            

        self.pWindow = None
        self.lineageWindow = None
        self.plotPropsBeenChecked = False
        self.toPlot = None
        self.emptying = False

        self.overrideNpyPath = None

        self.trainModelName = None
        self.trainModelSuffix = None

        self.experiments = []
        self.experiment_index = -1

        self.newInterpolation = False

        self.numObjs = 0

        self.win.show()
    
    @property
    def maxT(self):
        return self.channel().max_t()
    @maxT.setter
    def maxT(self, num):
        return
    def computeMaxT(self):
        self.tIndex =  min(self.tIndex, self.maxT)
    
    @property
    def maskLoaded(self):
        if not self.experiments:
            return False
        else:
            return self.experiment().has_labels()
    @maskLoaded.setter
    def maskLoaded(self, b):
        return


    @property
    def imZ(self):
        return self._imZ
    @imZ.setter
    def imZ(self, num):
        try:
            if num != self.imZ:
                self.imChanged = True
                self._imZ = num
        except AttributeError:
            self.imChanged = True
            self._imZ  = num
        
        if num>=0:
            try:
                self.checkInterpolation()
            except IndexError:
                pass
        if self.experiments:
            if self.experiment().channels[self._imZ].max_t() < self.tIndex:
                self.tIndex = 0
    
    @property
    def maskZ(self):
        return self._maskZ
    @maskZ.setter
    def maskZ(self, num):
        try:
            if num != self.maskZ:
                self.maskChanged = True
                self._maskZ = num
        except AttributeError:
            self._maskZ  = num
        if num>=0:
            try:
                self.checkProbability()
                self.checkDataAvailibility()
            except IndexError:
                pass
        if self.experiments and self.experiment().has_labels():
            if self.experiment().labels[self._maskZ].max_t() < self.tIndex:
                self.tIndex = 0
        
    
    @property
    def tIndex(self):
        return self._tIndex
    @tIndex.setter
    def tIndex(self, num):
        try:
            if num != self.tIndex:
                self.imChanged, self.maskChanged, self.tChanged = True, True, True
                self._tIndex = num
        except AttributeError:
            self._tIndex = num
    

    def handleFinished(self, error = False):
        self.closeThread(self.threads[-1])
        self.updateThreadDisplay()
    
    def clearCurrMask(self):
        self.currMask[self.currMask>0] = 0
        contours  = self.getCurrContours()
        contours[:,:] =0 
        self.label().save()
        self.drawMask()
    
    def labelCurrMask(self):
        mask = label(self.currMask)
        contours = self.getCurrContours()
        contours = get_mask_contour(self.currMask)
        self.getCurrMaskSet()[self.tIndex] = mask
        self.getCurrContourSet()[self.tIndex] = contours
        self.label().save()
        self.drawMask()
    
    def setEmptyDisplay(self, initial = False):
        self.tIndex = 0
        self.bigTStep = 3
        self.maxT = 0
        self.tChanged = False
        self.experiments = []
        self.experiment_index = -1
        if not initial:
            self.emptying = True
            self.experimentSelect.clear()

        self.setEmptyIms(initial = initial)
        self.setEmptyMasks(initial = initial)

        if not initial:
            self.updateDataDisplay()
            self.emptying = False
        
        self.saveData()
    
    def setEmptyIms(self, initial = False):
        # if self.maskLoaded:
        #     self.errorDialog = QtWidgets.QErrorMessage()
        #     self.errorDialog.showMessage("Images cannot be removed with masks present")
        #     return

        self.saturation = [0,255]
        self.imLoaded = False
        self.imZ = -1
        self.currIm =  np.zeros((512,512), dtype = np.uint8)
        self.imChanged = False
        self.pg_im.setImage(self.currIm, autoLevels=False, levels = [0,0])

        if not initial:
            self.channelSelect.clear()
            self.brushTypeSelect.setEnabled(False)
            self.brushTypeSelect.setCurrentIndex(-1)

    def setEmptyMasks(self, initial = False):
        self.firstMaskLoad = True
        self.maskLoaded = False
        self.selectedCells = []
        self.selectedCellContourColors = []
        self.cmap = self.getMaskCmap()
        self.floatCmap = self.getFloatCmap()
        self.maskColors = self.cmap.copy()
        self.maskChanged = False
        self.prevMaskOn = True
        self.maskOn = True
        self.contourOn = False
        self.probOn = False
        self.maskZ = -1
        self.currMask = np.zeros((512,512), dtype = np.uint8)
        self.pg_mask.setImage(self.currMask, autolevels  =False, levels =[0,0], lut = self.maskColors)


        if not initial:
            self.labelSelect.clear()
            self.maskOnCheck.setChecked(False)
            self.probOnCheck.setChecked(False)
            self.contourButton.setChecked(False)
            self.cellNumButton.setChecked(False)

    def getCurrTimeStr(self):
        now = datetime.now()
        return now.strftime("%d-%m-%Y|%H-%M")
    
    def newData(self):
        self.computeMaxT()
        self.updateDisplay()
        #self.saveData()

    # def newMasks(self, name = None):
    #     if not self.maskLoaded:
    #         self.maskLoaded = True
    #         self.enableMaskOperations()
    #     self.maskChanged = True

    #     if name:
    #         self.labelSelect.addItem(self.getNewLabelName(name))
    #     else:
    #         self.labelSelect.addItem("Label " + str(self.maskData.maxZ))
        
    #     self.maskTypeSelect.checkMaskRemote()

    #     self.cellData.append(None)
    #     self.newData()
    
    def addBlankMasks(self):
        if not self.imLoaded:
            self.showError("Load Images First")
            return
        blankMasks = np.zeros_like(self.getCurrImSet())
        self.loadMasks(blankMasks, name = "Blank")

    def make_viewbox(self):
        self.view = ViewBoxNoRightDrag(
            parent=self,
            lockAspect=True,
            name="plot1",
            border=[0, 0, 0],
            invertY=True
        )
        self.view.setCursor(QtCore.Qt.CrossCursor)
        self.brush_size =3

        self.win.addItem(self.view, row = 0, col = 0, rowspan = 20, colspan = 20)
        self.view.setMenuEnabled(False)
        self.view.setMouseEnabled(x=True, y=True)

        self.pg_im = pg.ImageItem()
        self.pg_mask = ImageDraw(viewbox = self.view, parent = self)

        self.view.addItem(self.pg_im)
        self.view.addItem(self.pg_mask)
        self.pg_mask.setZValue(10)
    

    def getCellColors(self,im):
        return np.take(self.maskColors, np.unique(im), 0)

    def build_widgets(self):
        rowspace = self.mainViewRows+1
        cspace = 2
        self.labelstyle = """QLabel{
                            color: white
                            } 
                         QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }"""
        self.statusbarstyle = ("color: white;" "background-color : black")
        self.boldfont = QtGui.QFont("Arial", 14, QtGui.QFont.Bold)
        self.medfont = QtGui.QFont("Arial", 12)
        self.smallfont = QtGui.QFont("Arial", 10)
        self.headings = ('color: rgb(200,10,10);')
        self.dropdowns = ("color: white;"
                        "background-color: rgb(40,40,40);"
                        "selection-color: white;"
                        "selection-background-color: rgb(50,100,50);")
        self.checkstyle = "color: rgb(190,190,190);"

        self.statusBar = QStatusBar()
        self.statusBar.setFont(self.medfont)
        self.statusBar.setStyleSheet(self.statusbarstyle)
        self.setStatusBar(self.statusBar)

        self.gpuDisplayTorch = ReadOnlyCheckBox("gpu - torch  |  ")
        self.gpuDisplayTF= ReadOnlyCheckBox("gpu - tf")
        self.gpuDisplayTF.setFont(self.smallfont)
        self.gpuDisplayTorch.setFont(self.smallfont)
        self.gpuDisplayTF.setStyleSheet(self.checkstyle)
        self.gpuDisplayTorch.setStyleSheet(self.checkstyle)
        self.gpuDisplayTF.setChecked(False)
        self.gpuDisplayTorch.setChecked(False)
        self.checkGPUs()
        if self.tf:
            self.gpuDisplayTF.setChecked(True)
        if self.torch:
            self.gpuDisplayTorch.setChecked(True)
        self.statusBarLayout = QGridLayout()
        self.statusBarWidget = QWidget()
        self.statusBarWidget.setLayout(self.statusBarLayout)

        self.cpuCoreDisplay = QLabel("")
        self.cpuCoreDisplay.setFont(self.smallfont)
        self.cpuCoreDisplay.setStyleSheet(self.labelstyle)
        self.updateThreadDisplay()

        self.hasLineageBox = ReadOnlyCheckBox("lineage data")
        self.hasCellDataBox = ReadOnlyCheckBox("cell data")
        for display in [self.hasLineageBox,self.hasCellDataBox]:
            display.setFont(self.smallfont)
            display.setStyleSheet(self.checkstyle)
            display.setChecked(False)

        self.statusBarLayout.addWidget(self.cpuCoreDisplay, 0, 1, 1,1, alignment=(QtCore.Qt.AlignCenter))
        self.statusBarLayout.addWidget(self.gpuDisplayTF, 0, 2, 1, 1)
        self.statusBarLayout.addWidget(self.gpuDisplayTorch, 0, 3, 1, 1)
        self.statusBarLayout.addWidget(self.hasLineageBox, 0,4,1,1)
        self.statusBarLayout.addWidget(self.hasCellDataBox,0,5,1,1)
        self.statusBar.addWidget(self.statusBarWidget)
        
        self.dataDisplay = QLabel("")
        self.dataDisplay.setMinimumWidth(300)
        # self.dataDisplay.setMaximumWidth(300)
        self.dataDisplay.setStyleSheet(self.labelstyle)
        self.dataDisplay.setFont(self.medfont)
        self.dataDisplay.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.updateDataDisplay()
        self.l.addWidget(self.dataDisplay, rowspace-1,cspace,1,20)

        self.experimentLabel = QLabel("Experiment:")
        self.experimentLabel.setStyleSheet(self.labelstyle)
        self.experimentLabel.setFont(self.smallfont)
        self.l.addWidget(self.experimentLabel, 0,1,1,2)
        self.experimentSelect= QComboBox()
        self.experimentSelect.setStyleSheet(self.dropdowns)
        self.experimentSelect.setFocusPolicy(QtCore.Qt.NoFocus)
        self.experimentSelect.setFont(self.medfont)
        self.experimentSelect.currentIndexChanged.connect(self.experimentChange)
        self.l.addWidget(self.experimentSelect, 0, 3, 1,2)
        
        self.channelSelectLabel = QLabel("Channel: ")
        self.channelSelectLabel.setStyleSheet(self.labelstyle)
        self.channelSelectLabel.setFont(self.smallfont)
        self.l.addWidget(self.channelSelectLabel, 0, self.mainViewCols-7,1,1)
        self.channelSelect = QComboBox()
        self.channelSelect.setStyleSheet(self.dropdowns)
        self.channelSelect.setFont(self.medfont)
        self.channelSelect.currentIndexChanged.connect(self.channelSelectIndexChange)
        self.channelSelect.setEditable(True)
        self.channelSelect.editTextChanged.connect(self.channelSelectEdit)
        self.channelSelect.setEnabled(False)
        self.l.addWidget(self.channelSelect, 0, self.mainViewCols-6,1, 2)
        self.l.setAlignment(self.channelSelect, QtCore.Qt.AlignLeft)
        self.channelSelect.setMinimumWidth(100)
        self.channelSelect.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.channelSelect.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.channelSelect.setMinimumWidth(200)
        setattr(self.channelSelect, "items", lambda: [self.channelSelect.itemText(i) for i in range(self.channelSelect.count())])


        self.labelSelectLabel = QLabel("Label: ")
        self.labelSelectLabel.setStyleSheet(self.labelstyle)
        self.labelSelectLabel.setFont(self.smallfont)
        self.l.addWidget(self.labelSelectLabel, 0, self.mainViewCols-4,1,1)
        self.labelSelect = QComboBox()
        self.labelSelect.setStyleSheet(self.dropdowns)
        self.labelSelect.setFont(self.medfont)
        self.labelSelect.currentIndexChanged.connect(self.labelSelectIndexChange)
        self.labelSelect.setEditable(True)
        self.labelSelect.editTextChanged.connect(self.labelSelectEdit)
        self.labelSelect.setMinimumWidth(100)
        self.labelSelect.setEnabled(False)
        self.l.addWidget(self.labelSelect, 0, self.mainViewCols-3,1,2)
        self.l.setAlignment(self.labelSelect, QtCore.Qt.AlignLeft)
        setattr(self.labelSelect, "items", lambda: [self.labelSelect.itemText(i) for i in range(self.labelSelect.count())])
        self.labelSelect.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.labelSelect.setMinimumWidth(200)
        self.labelSelect.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        label = QLabel('Drawing:')#[\u2191 \u2193]')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.l.addWidget(label, rowspace,1,1,5)

        label = QLabel("Brush Type:")
        label.setStyleSheet(self.labelstyle)
        label.setFont(self.medfont)
        self.l.addWidget(label,rowspace+1,1,1,2)
        self.brushTypeSelect = QComboBox()
        self.brushTypeSelect.addItems(["Eraser", "Brush", "Outline"])
        self.brushTypeSelect.setCurrentIndex(-1)
        self.brushTypeSelect.currentIndexChanged.connect(self.brushTypeChoose)
        self.brushTypeSelect.setStyleSheet(self.dropdowns)
        self.brushTypeSelect.setFont(self.medfont)
        self.brushTypeSelect.setCurrentText("")
        self.brushTypeSelect.setFocusPolicy(QtCore.Qt.NoFocus)
        self.brushTypeSelect.setEnabled(False)
        self.brushTypeSelect.setFixedWidth(90)
        self.brushTypeSelect.setEnabled(False)
        self.l.addWidget(self.brushTypeSelect, rowspace+1,3,1,1)

        
        label = QLabel("Brush Size")
        label.setStyleSheet(self.labelstyle)
        label.setFont(self.medfont)
        self.l.addWidget(label, rowspace+2,1,1,2)
        self.brush_size = 3
        self.brushSelect = QSpinBox()
        self.brushSelect.setMinimum(1)
        self.brushSelect.setValue(self.brush_size)
        self.brushSelect.valueChanged.connect(self.brushSizeChoose)
        self.brushSelect.setFixedWidth(90)
        self.brushSelect.setStyleSheet(self.dropdowns)
        self.brushSelect.setFont(self.medfont)
        edit = self.brushSelect.lineEdit()
        edit.setFocusPolicy(QtCore.Qt.NoFocus)
        self.brushSelect.setEnabled(False)
        self.l.addWidget(self.brushSelect, rowspace+2,3,1,1)

        line = QVLine()
        line.setStyleSheet('color:white;')
        self.l.addWidget(line, rowspace,4,6,1)

        label = QLabel('Tracking')#[\u2191 \u2193]')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.l.addWidget(label, rowspace,5,1,4)

        self.trackButton = QPushButton('track cells')
        #self.trackButton.setFixedWidth(90)
        #self.trackButton.setFixedHeight(20)
        self.trackButton.setStyleSheet(self.styleInactive)
        self.trackButton.setFont(self.medfont)
        self.trackButton.clicked.connect(self.trackButtonClick)
        self.trackButton.setEnabled(False)
        self.trackButton.setToolTip("Track current cell labels")
        self.l.addWidget(self.trackButton, rowspace+1, 5,1,2)

        self.trackObjButton = QPushButton('track label to cell')
        #self.trackObjButton.setFixedWidth(90)
        #self.trackObjButton.setFixedHeight(20)
        self.trackObjButton.setFont(self.medfont)
        self.trackObjButton.setStyleSheet(self.styleInactive)
        self.trackObjButton.clicked.connect(self.trackObjButtonClick)
        self.trackObjButton.setEnabled(False)
        self.trackObjButton.setToolTip("Track current non-cytoplasmic label to a cellular label")
        self.l.addWidget(self.trackObjButton, rowspace+1, 7,1,2)

        self.lineageButton = QPushButton("get lineages")
        self.lineageButton.setStyleSheet(self.styleInactive)
        #self.lineageButton.setFixedWidth(90)
        #self.lineageButton.setFixedHeight(18)
        self.lineageButton.setFont(self.medfont)
        self.lineageButton.setToolTip("Use current budNET mask to assign lineages to a cellular label")
        self.lineageButton.setEnabled(False)
        self.showMotherDaughters = False
        self.showLineages = False
        self.lineageButton.clicked.connect(self.getLineages)
        self.l.addWidget(self.lineageButton, rowspace+3, 5,1,2)

        self.interpolateButton = QPushButton("interpolate movie")
        self.interpolateButton.setStyleSheet(self.styleInactive)
        self.interpolateButton.setFont(self.medfont)
        self.interpolateButton.setEnabled(False)
        self.interpolateButton.clicked.connect(self.interpolateButtonClicked)
        self.l.addWidget(self.interpolateButton, rowspace+2, 5,1,2)

        self.interpRemoveButton = QPushButton("remove interpolation")
        self.interpRemoveButton.setStyleSheet(self.styleInactive)
        self.interpRemoveButton.setFont(self.medfont)
        self.interpRemoveButton.setEnabled(False)
        self.interpRemoveButton.clicked.connect(self.interpRemoveButtonClicked)
        self.l.addWidget(self.interpRemoveButton, rowspace+2, 7,1,2)

        line = QVLine()
        line.setStyleSheet('color:white;')
        self.l.addWidget(line, rowspace,9,6,1)
    
        label = QLabel('Segmentation')#[\u2191 \u2193]')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.l.addWidget(label, rowspace,10,1,5)
        
        #----------UNETS-----------
        self.artiButton = QPushButton(u'artilife full lifecycle')
        self.artiButton.setEnabled(False)
        self.artiButton.setFont(self.medfont)
        self.artiButton.clicked.connect(self.computeArtilifeModel)
        self.artiButton.setStyleSheet(self.styleInactive)
        self.l.addWidget(self.artiButton, rowspace+1, 10, 1,5, Qt.AlignTop)

        self.GB = QGroupBox("Unets")
        self.GB.setStyleSheet("QGroupBox { border: 1px solid white; color:white; padding: 10px 0px;}")
        self.GBLayout = QGridLayout()
        self.GB.setLayout(self.GBLayout)
        self.GB.setToolTip("Select Unet(s) to be used for segmenting channel")
    
        self.getModels()
        self.modelChoose = QComboBox()
        self.modelChoose.addItems(sorted(self.modelNames, key = lambda x: x[0]))
            #self.modelChoose.setItemChecked(i, False)
        #self.modelChoose.setFixedWidth(180)
        self.modelChoose.setStyleSheet(self.dropdowns)
        self.modelChoose.setFont(self.medfont)
        self.modelChoose.setFocusPolicy(QtCore.Qt.NoFocus)
        self.modelChoose.setCurrentIndex(-1)
        self.GBLayout.addWidget(self.modelChoose, 0,0,1,7)

        self.modelButton = QPushButton(u'run model')
        self.modelButton.clicked.connect(self.computeModels)
        self.GBLayout.addWidget(self.modelButton, 0,7,1,2)
        self.modelButton.setEnabled(False)
        self.modelButton.setStyleSheet(self.styleInactive)
        self.l.addWidget(self.GB, rowspace+2,10,3,5, Qt.AlignTop | Qt.AlignHCenter)


        #------Flourescence Segmentation -----------pp-p-
        # self.segButton = QPushButton(u'blob detection')
        # self.segButton.setEnabled(False)
        # self.segButton.clicked.connect(self.doFlou)
        # self.segButton.setStyleSheet(self.styleInactive)
        # self.l.addWidget(self.segButton, rowspace+4,10,3,5, Qt.AlignTop)

        #----------------------------------s-------------

        line = QVLine()
        line.setStyleSheet('color:white;')
        self.l.addWidget(line, rowspace,15,6,1)

        label = QLabel('Display')#[\u2191 \u2193]')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.l.addWidget(label, rowspace,16,1,5)

        self.contourButton = QCheckBox("Show Contours")
        self.contourButton.setStyleSheet(self.checkstyle)
        self.contourButton.setFont(self.medfont)
        self.contourButton.stateChanged.connect(self.toggleContours)
        self.contourButton.setShortcut(QtCore.Qt.Key_C)
        self.contourButton.setEnabled(False)
        self.l.addWidget(self.contourButton, rowspace+1, 16,1,2)

        self.plotButton = QCheckBox("Show Plot Window")
        self.plotButton.setStyleSheet(self.checkstyle)
        self.plotButton.setFont(self.medfont)
        self.plotButton.stateChanged.connect(self.togglePlotWindow)
        self.plotButton.setShortcut(QtCore.Qt.Key_P)
        self.l.addWidget(self.plotButton, rowspace+1, 18, 1,2)

        self.maskOnCheck = QCheckBox("Mask")
        self.maskOnCheck.setStyleSheet(self.checkstyle)
        self.maskOnCheck.setFont(self.medfont)
        self.maskOnCheck.setEnabled(False)
        self.maskOnCheck.setShortcut(QtCore.Qt.Key_Space)
        self.maskOnCheck.stateChanged.connect(self.toggleMask)
        self.l.addWidget(self.maskOnCheck, rowspace+2, 16,1,2)

        self.probOnCheck = QCheckBox("Probability")
        self.probOnCheck.setStyleSheet(self.checkstyle)
        self.probOnCheck.setFont(self.medfont)
        self.probOnCheck.setEnabled(False)
        self.probOnCheck.setShortcut(QtCore.Qt.Key_F)
        self.probOnCheck.stateChanged.connect(self.toggleProb)
        self.l.addWidget(self.probOnCheck, rowspace+2, 18,1,2)

        self.cellNumButton = QCheckBox("cell nums")
        self.cellNumButton.setStyleSheet(self.checkstyle)
        self.cellNumButton.setFont(self.medfont)
        self.cellNumButton.stateChanged.connect(self.toggleCellNums)
        self.cellNumButton.setEnabled(False)
        self.l.addWidget(self.cellNumButton, rowspace+3, 16, 1,2)

        self.showLineageButton = QCheckBox("lineages")
        self.showLineageButton.setStyleSheet(self.checkstyle)
        self.showLineageButton.setFont(self.medfont)
        self.showLineageButton.stateChanged.connect(self.toggleLineages)
        self.showLineageButton.setEnabled(False)
        self.l.addWidget(self.showLineageButton, rowspace+3, 18, 1,2)

        self.showMotherDaughtersButton = QCheckBox("mother-daughters")
        self.showMotherDaughtersButton.setStyleSheet(self.checkstyle)
        self.showMotherDaughtersButton.setFont(self.medfont)
        self.showMotherDaughtersButton.stateChanged.connect(self.toggleMotherDaughters)
        self.showMotherDaughtersButton.setEnabled(False)
        self.l.addWidget(self.showMotherDaughtersButton, rowspace+4, 16, 1,2)

        self.showTreeButton = QCheckBox("lineage tree")
        self.showTreeButton.setStyleSheet(self.checkstyle)
        self.showTreeButton.setFont(self.medfont)
        self.showTreeButton.stateChanged.connect(self.toggleLineageTreeWindow)
        self.showTreeButton.setEnabled(False)
        self.l.addWidget(self.showTreeButton, rowspace+4, 18, 1,2)

        # self.autoSaturationButton = QPushButton("Auto")
        # self.autoSaturationButton.setFixedWidth(45)
        # self.autoSaturationButton.setEnabled(True)
        # self.autoSaturationButton.setStyleSheet(self.styleInactive)
        # self.autoSaturationButton.setEnabled(False)
        # self.autoSaturationButton.clicked.connect(self.resetAutoSaturation)
        # self.l.addWidget(self.autoSaturationButton,rowspace+4, 22,1,1)
        # self.saturationSlider = RangeSlider(self)
        # self.saturationSlider.setMinimum(0)
        # self.saturationSlider.setMaximum(255)
        # self.saturationSlider.setLow(self.saturation[0])
        # self.saturationSlider.setHigh(self.saturation[-1])
        # self.saturationSlider.setTickPosition(QSlider.TicksRight)
        # self.saturationSlider.setFocusPolicy(QtCore.Qt.NoFocus)
        # self.saturationSlider.setEnabled(False)
        # self.l.addWidget(self.saturationSlider, rowspace+4, 22,1,3)
        for i in range(self.mainViewRows):
            self.l.setRowStretch(i, 10)

        self.l.setColumnStretch(20,2)
        self.l.setColumnStretch(0,2)
        self.l.setContentsMargins(0,0,0,0)
        self.l.setSpacing(0)
    
    def updateThreadDisplay(self):
        threadCount = len(self.threads)+1
        self.cpuCoreDisplay.setText(f"{threadCount}/{self.idealThreadCount} cores |")

    def enableMaskOperations(self):
        self.contourButton.setEnabled(True)
        self.labelSelect.setEnabled(True)
        self.trackButton.setEnabled(True)
        self.trackObjButton.setEnabled(True)
        self.trackButton.setStyleSheet(self.styleUnpressed)
        self.trackObjButton.setStyleSheet(self.styleUnpressed)
        self.cellNumButton.setEnabled(True)
        self.maskOnCheck.setEnabled(True)
        self.maskOnCheck.setChecked(True)
        self.probOnCheck.setEnabled(self.label().has_probability)
        self.lineageButton.setEnabled(True)
        self.lineageButton.setStyleSheet(self.styleUnpressed)
    
    def enableImageOperations(self):
        self.brushSelect.setEnabled(True)
        self.brushTypeSelect.setEnabled(True)
        #self.saturationSlider.setEnabled(True)
        #self.autoSaturationButton.setEnabled(True)
        #self.autoSaturationButton.setStyleSheet(self.styleUnpressed)
        self.modelButton.setEnabled(True)
        self.modelButton.setStyleSheet(self.styleUnpressed)
        self.artiButton.setEnabled(True)
        self.artiButton.setStyleSheet(self.styleUnpressed)
        self.channelSelect.setEnabled(True)
        self.checkInterpolation()
    
    def toggleDrawing(self,b):
        if b:
            idx = 0
        else:
            idx = -1
        self.brushTypeSelect.setCurrentIndex(idx)
        self.brushTypeSelect.setEnabled(b)
        self.brushSelect.setEnabled(b)
        self.brushTypeSelect.setEnabled(b)

    def updateModelNames(self):
        self.getModelNames()
        self.modelChoose.clear()
        self.modelChoose.addItems(sorted(self.modelNames))


    def channelSelectEdit(self, text):
        curr_text = self.channelSelect.currentText()
        idx = self.channelSelect.currentIndex()
        if not self.emptying:
            if isinstance(self.experiment().channels[idx], InterpolatedChannel) and InterpolatedChannel.text_id not in text:
                if not self.newInterpolation:
                    self.showError(f"Removal of the {InterpolatedChannel.text_id} text id from the name of this channel will disable certain features upon reloading")
            if self.experiment().new_channel_name(idx, text):
                self.channelSelect.setItemText(idx, str(text))
        else:
            self.channelSelect.setItemText(idx, "")


    def channelSelectIndexChange(self,index):
        if not self.emptying:
            if self.imZ != index:
                self.imZ = index

            self.imChanged = True
            self.updateDisplay()
    
    def channelSelectRemote(self):
        index = self.imZ
        self.channelSelect.setCurrentIndex(index)

    def labelSelectEdit(self, text):
        if self.emptying:
            return
        idx = self.labelSelect.currentIndex()
        text = str(text)
        if self.experiment().new_label_name(idx, text):
            self.labelSelect.setItemText(idx, str(text))

    def labelSelectIndexChange(self,index):
        if not self.emptying:
            if self.maskZ != index:
                self.maskZ = index
                self.maskChanged = True
                self.updateDisplay()
    
    def labelSelectRemote(self):
        index = self.maskZ
        self.labelSelect.setCurrentIndex(index)

    def toggleMotherDaughters(self):
        if self.showLineages:
            self.showLineageButton.setChecked(False)
            self.deselectAllCells()
    
        self.showMotherDaughters = self.showMotherDaughtersButton.checkState()
    
        if not self.showMotherDaughters:
            self.deselectAllCells()
        else:
            toSelect = self.selectedCells.copy()
            self.deselectAllCells()
            for cell in toSelect:
                self.selectCellFromNum(cell)

    def toggleLineages(self):
        if self.showMotherDaughters:
            self.showMotherDaughtersButton.setChecked(False)

        self.showLineages = self.showLineageButton.checkState()
        self.deselectAllCells()


    def removeSelectedMothers(self):
        self.deselectAllCells()
    
    def deselectAllCells(self):
        for cell in self.selectedCells.copy():
            self.deselectCell(cell)
        self.drawMask()
    
    def trackObjButtonClick(self):
        if self.label().max_t() == 0:
            self.showError("Error: More than one frame must be present to track")
            return 
        
        labelsToTrack =  self.labelSelect.currentText()
        dlg  = GeneralParamDialog({}, [], f"Track {labelsToTrack}", self, labelSelects=["Cytoplasm Label"])

        if dlg.exec():
            self.deactivateButton(self.trackObjButton)
            data = dlg.getData()


            cellIdx = self.labelSelect.findText(data["Cytoplasm Label"])
            cells = self.experiment().get_label("labels", idx = cellIdx)
            obj = self.getCurrMaskSet()

            task = partial(track_to_cell, obj, cells)
            worker = TrackWorker(task, cells, self.maskZ, self.experiment_index, obj)            
            self.runLongTask(worker,self.trackFinished, self.trackObjButton)

        else:
            return
    
    def checkProbability(self):
        self.probOnCheck.setEnabled(self.label().has_probability)

    
    def trackFinished(self, z, exp_idx, tracked):
        if isinstance(tracked, np.ndarray):
            contours = self.getCurrContourSet()
            newContours = tracked.copy()
            newContours[np.logical_not(contours>0)] = 0
            self.experiments[exp_idx].labels[z].set_data(tracked, newContours)
            self.updateCellData(idx = z, exp_idx = exp_idx)
            self.experiment_index = exp_idx
            self.maskZ = z
            self.drawMask()
            self.showTimedPopup(f"{self.labelSelect.currentText()} has been tracked")
        

    def trackButtonClick(self):
        if self.label().max_t() == 0:
            self.showError("Error: More than one frame must be present to track")
            return

        idxToTrack = self.maskZ
        cells = self.getCurrMaskSet()
        task = trackYeasts
        worker = TrackWorker(task, cells, idxToTrack, self.experiment_index)     
    
        self.runLongTask(worker,self.trackFinished, self.trackButton)

        
    def updateCellData(self, idx = None, exp_idx = None):
        if not self.maskLoaded:
            return
  
        if idx:
            if exp_idx is not None:
                exp = self.experiments[exp_idx]
            else:
                exp = self.experiment()

            mask_obj = exp.labels[idx]
            masks = exp.get_label("labels", idx = idx)
        else:
            exp = self.experiment()
            mask_obj = self.label()
            masks = self.getCurrMaskSet()
            idx = self.maskZ

    
        viable_channel_objs = exp.get_viable_im_objs(mask_obj)
        viableIms = [obj.ims for obj in viable_channel_objs]
        viableImNames = [obj.name for obj in viable_channel_objs]

        if mask_obj.has_cell_data():
            mask_obj.celldata.update_cell_data(masks, channels = viableIms, channel_names = viableImNames)
        else:
            mask_obj.celldata = TimeSeriesData(idx, masks, channels = viableIms, channel_names=viableImNames)

        mask_obj.save()
        self.checkDataAvailibility()  

        if self.pWindow:
            self.pWindow.setData()
            self.pWindow.updatePlots()
    
    def updateCellTable(self):
        if not self.pWindow:
            return
        # self.pWindow.table.model.setData(self.getCellDataAbbrev())
        # self.pWindow.
    

    def hasCellData(self, i = None):
        if type(i) is not int:
            i = self.maskZ
        return self.experiment().labels[i].has_cell_data()
    
    def hasLineageData(self, i = None):
        if not i:
            i = self.maskZ
        return self.experiment().labels[i].has_lineage_data()
    
    def getLabelsWithPopulationData(self):
        return [population.celldata for i,population in enumerate(self.experiment().labels) if self.hasCellData(i = i)]
    
    def getCellData(self):
        if not self.maskLoaded or not self.hasCellData():
            return None
        else:
            return self.label().celldata
    
    def getTimeSeriesDataName(self, tsObj):
        index = tsObj.i
        return self.labelSelect.items()[index]

    def getCellDataAbbrev(self):
        if not self.maskLoaded:
            return pd.DataFrame.from_dict(self.cellDataFormat)
        if not self.hasCellData():
            return pd.DataFrame.from_dict(self.cellDataFormat)
        elif self.maskLoaded and self.hasCellData():
            return self.label().celldata.get_cell_info()
        
    def getLineages(self):
        dlg = GeneralParamDialog({}, [], "Lineage Construction", self, labelSelects=["bud necks", "cells"])
        if dlg.exec_():
            data = dlg.getData()
            neckI = self.labelSelect.findText(data["bud necks"])
            cellI = self.labelSelect.findText(data["cells"])
            necks = self.experiment().get_label("labels", idx = neckI)
            cells = self.experiment().get_label("labels", idx = cellI)
        else:
            return
  
        if isinstance(self.experiment().labels[cellI].celldata, LineageData):
            self.experiment().labels[cellI].celldata.add_lineages(cells, necks)
        else:
            if isinstance(self.experiment().labels[cellI].celldata, TimeSeriesData):
                life = self.experiment().labels[cellI].celldata.life_data
                cell_data = self.experiment().labels[cellI].celldata.cell_data
            else:
                life, cell_data = None, None

            self.experiment().labels[cellI].celldata = LineageData(cellI, cells, buds = necks, cell_data=cell_data, life_data=life)

        self.experiment().labels[cellI].save()
        self.checkDataAvailibility()
        self.showTreeButton.setCheckState(True)
    

    
    def getMatingLineages(self, cellI, matingI):
        mating = self.experiment().get_label("labels", idx = matingI)
        cells = self.experiment().get_label("labels", idx = cellI)

        if isinstance(self.experiment().labels[cellI].celldata, LineageData):
            self.experiment().labels[cellI].celldata.add_mating(cells, mating)
        else:
            if isinstance(self.experiment().labels[cellI].celldata, TimeSeriesData):
                life = self.experiment().labels[cellI].celldata.life_data
                cell_data = self.experiment().labels[cellI].celldata.cell_data
            else:
                life, cell_data = None, None

            self.experiment().labels[cellI].celldata = LineageData(cellI, cells, mating = mating, cell_data=cell_data, life_data=life)

        self.experiment().labels[cellI].save()
        self.checkDataAvailibility()
    
    def saveData(self):
        pass
    
    def getCellDataLabelProps(self):
        label_props = LABEL_PROPS.copy()
        label_props.remove("label")
        return sorted(label_props)

    def getCellDataImProperties(self, i = None):
        if not i:
            i = self.maskZ

        extra_props = [prop.__name__ for prop in EXTRA_IM_PROPS]
        if not self.maskLoaded or not self.hasCellData(i = i):
            return sorted(IM_PROPS + extra_props)
        else:
            im_props = []
            columns  = self.label().celldata.columns
            for column in columns:
                for prop in IM_PROPS + extra_props:
                    if prop in column:
                        im_props.append(column)
            return sorted(im_props)

    def showError(self, message = ""):
        self.errorDialog = QtWidgets.QErrorMessage()
        self.errorDialog.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        self.errorDialog.setFixedWidth(500)
        self.errorDialog.showMessage(message)

    def showTimedPopup(self, text, time = 30):
        new_text =f"<b>{text}</b>"
        popup = TimedPopup(new_text, time)
        popup.exec_()



    def closeThread(self, thread):
        index = self.threads.index(thread)
        thread = self.threads[index]
        thread.quit()
        thread.wait()
        del self.threads[index]
        del self.workers[index]
        
        self.updateThreadDisplay()

    def toggleContours(self):
        self.contourOn = self.contourButton.isChecked()
        self.contourButton.clearFocus()

        if self.contourOn:
            self.drawContours()
        else:
            self.drawMask()

    def toggleCellNums(self):
        if self.cellNumButton.isChecked():
            self.addCellNumsToCurrIm()
        else:
            self.clearCellNums()
        self.cellNumButton.clearFocus()
    
    def clearCellNums(self):
        self.pg_mask.image = self.maskColors[self.currMask]  
        self.pg_mask.updateImage()

    
    def addCellNumsToCurrIm(self):
        vals, xtemp, ytemp = self.getCellCenters(self.currMask)
        image = self.pg_mask.image
        for i in range(0,len(xtemp)):
            image = cv2.putText(image, str(vals[i]), (xtemp[i], ytemp[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255,255,255,255), 1)
        self.pg_mask.image = image
        self.pg_mask.updateImage()
    

    def toggleLineageTreeWindow(self):
        self.lineageWindowOn = self.showTreeButton.isChecked()

        if self.lineageWindowOn:
            self.showTree()
        elif self.lineageWindow is not None:
            self.lineageWindow.close()
            self.lineageWindow = None
    
    def showTree(self):
        data = self.label().celldata.get_cell_info()
        data = data.drop(columns = ["confidence"])
        data =  data.fillna(-1)
        data = data.to_numpy()
        self.lineageWindow = plot.LineageTreeWindow(self, data, selected_cells=self.selectedCells)

        
    def buildPlotWindow(self):
        win = PlotWindowCustomize(self)

        if win.exec_():
            self.toPlot = win.getData()
        else:
            self.plotButton.setCheckState(False)
            return
        
        self.pWindow = plot.PlotWindow(self, self.toPlot)
        self.pWindow.show()

        
    def togglePlotWindow(self):
        self.plotWindowOn = self.plotButton.isChecked()

        if self.plotWindowOn:
            if self.hasCellData():
                self.buildPlotWindow()
            else:
                self.showError("No Timeseries Data Available - Track Cell First")   
        elif self.pWindow is not None:
            self.pWindow.close()
            self.pWindow = None

    def brushSizeChoose(self):
        self.brush_size = self.brushSelect.value()
        self.pg_mask.setDrawKernel(kernel_size=self.brush_size)
    
    def brushTypeChoose(self):
        self.drawType = self.brushTypeSelect.currentText()
        if self.drawType != "":
            self.pg_mask.turnOnDraw()
        else:
            self.pg_mask.turnOffDraw()
    
    def changeBrushTypeComboBox(self, brushName):
        index = self.brushTypeSelect.findText(brushName, QtCore.Qt.MatchFixedString)
        if index>=0:
            self.brushTypeSelect.setCurrentIndex(index)
    
    def addCellContour(self, contourMask, cellMask, cellnum):
        contour = get_cell_contour(cellMask, cellnum)
        contourMask[contourMask==cellnum] = 0
        contourMask[contour==1] = cellnum
        return contourMask

    def addRecentDrawing(self):
        newStroke, newColor = self.strokes[-1]
        #self.maskData.channels[self.maskZ][0, self.tIndex,:,:][newStroke]=newColor
        contourMask = self.experiment().get_label("contours", idx = self.maskZ, t = self.tIndex)
        cellMask = self.experiment().get_label("labels", idx = self.maskZ, t = self.tIndex)
        self.label().npzdata["contours"][self.tIndex,:,:] = (self.addCellContour(contourMask, cellMask, newColor))
        self.label().save()

    def updateDataDisplay(self, x = None,y = None, val = None):
        #dataString = f"Session Id: {self.sessionId}"
        dataString = ""
        if x is not None and y is not None and val is not None and x>=0 and y>=0 and x<self.currIm.shape[1] and y<self.currIm.shape[0]:
            self.labelX, self.labelY = x,y
            if self.probOn:
                val = self.experiment().get_label("probability", idx = self.maskZ, t = self.tIndex)[y,x]/255
            dataString += f"x={str(x)}, y={str(y)}, value={str(val)}"

        if self.imLoaded:
            dataString += f'\n{self.channel().get_files(self.tIndex)}'

            dataString += f" | IMAGE: {self.channel().get_string(self.tIndex)}, T: {self.tIndex}/{self.channel().max_t()}"
            
            if self.maskLoaded:
                dataString += f"  |  MASK: {self.label().get_string(self.tIndex)}, T: {self.tIndex}/{self.label().max_t()}"

            #dataString += f"  |  TIME: {self.tIndex}/{self.maxT}"

            if self.maskLoaded:
                dataString += f"  |  REGIONS: {self.numObjs}"
        self.dataDisplay.setText(dataString)
    
    def experiment(self):
        return self.experiments[self.experiment_index]
    
    def channel(self):
        return self.experiment().channels[self.imZ]

    def label(self):
        return self.experiment().labels[self.maskZ]
    
    def deleteCell(self, cellNum):
        stroke = self.currMask == cellNum
        self.strokes.append((stroke,cellNum))
        self.currMask[stroke] = 0
        self.label().npzdata["contours"][self.tIndex,:,:][stroke]=0
        self.drawMask()
        self.label().save()

    def getMaskCmap(self):
        cmap = plt.get_cmap("tab20",20) # returns Ncolors from the jet colormap
        colors = []
        i = 0
        for i in range(0, 20):

            temp = np.array(list(cmap(i)))
            temp *=255.0
            colors.append(tuple(temp))
        
        # rng = np.random.default_rng()
        colors = colors*100
        colors = np.array(colors, dtype = np.uint8)
        # colors =  rng.permutation(colors)
        colors[0,-1] = 0
        #exit()
        colors[:,-1] = colors[:,-1]//2
        return colors
    
    def getFloatCmap(self):
        cmap = plt.get_cmap("gist_rainbow", 256)
        colors = []
        i = 0
        for i in range(0,256):
            temp = np.array(list(cmap(i)))
            temp *=255.0
            colors.append(tuple(temp))
        colors = colors*100
        colors = np.array(colors, dtype = np.uint8)
        colors[:,-1] = colors[:,-1]//2
        colors[0,-1] = 0
        return colors
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
    
    def dragMoveEvent(self, event):
        if not self.imLoaded:
            m = "Image files"
        else:
            if self.isUpperHalf(event):
                m = "Mask Files"
            else:
                m = "Image Files"
        self.toStatusBar(m)
    
    def get_experiment_names(self):
        return [experiment.name for experiment in self.experiments]
    
    def toStatusBar(self, message, time = None):
        self.statusBar.clearMessage()
        # if time:
        #     self.statusBar.showMessage(message, time)
        # else:
        #     self.statusBar.showMessage(message)
    
    def setChannelSelectName(self, name):
        self.channelSelect.setItemText(self.channelSelect.currentIndex(), self.getNewChannelName(name))
    
    def setLabelSelectName(self, name):
        self.labelSelect.setItemText(self.labelSelect.currentIndex(), self.getNewLabelName(name)) 

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]

        if os.path.isdir(files[0]):
            if get_filename(files[0]) in self.get_experiment_names():
                self.showError(f"{get_filename} already exists as an experiment. Change the directory name.")
                return
            else:
                self.loadExperiment(files[0])
    
    def userSelectExperiment(self):
        dir = QFileDialog.getExistingDirectory(self, "Choose Experiment Directory")
        self.loadExperiment(dir)
    
    def loadExperiment(self,file):
        if file.endswith("/") or file.endswith('\\'):
            file = file[:-1]                         
        dlg = GeneralParamDialog({"num_channels":1}, [int], "", self)
        if dlg.exec():
            num_channels = int(dlg.getData()["num_channels"])
        else:
            return
        self.experiment_index+=1
        new_experiment = Experiment(file, num_channels=num_channels)
        self.experiments.append(new_experiment)
        self.experimentSelect.addItem(new_experiment.name)
        self.experimentSelect.setCurrentIndex(self.experiment_index)
        self.showExperimentMessage(new_experiment)
        self.checkInterpolation()
    
    def showExperimentMessage(self, newExp):
        channels = ", ".join(newExp.get_channel_names())
        message = f"NEW EXPERIMENT: {newExp.name} | CHANNELS: {channels}"
        if newExp.has_labels():
            labels = ",".join(newExp.get_label_names())
            message+=f" | LABELS: {labels}"
        self.showTimedPopup(message, time = 60)


    def setDataSelects(self):
        self.clearDataSelects()
        self.channelSelect.addItems(self.experiment().get_channel_names())
        if self.experiment().has_labels():
            self.labelSelect.addItems(self.experiment().get_label_names())
        
    
    def experimentChange(self):
        if not self.emptying:
            self.imLoaded = True
            self.setDataSelects()
            self.experiment_index = self.experimentSelect.currentIndex()
            self.tIndex, self.maskZ, self.imZ = 0,0,0
            self.imChanged, self.maskChanged = True, True
            self.enableImageOperations()
            if self.experiment().has_labels():
                self.maskLoaded = True
                self.enableMaskOperations()
            self.updateDisplay()

    def clearDataSelects(self):
        self.labelSelect.clear()
        self.channelSelect.clear()

    def newIms(self, ims = None, files = None, dir = None, name = None, annotations = None):
        if files is not None:
            self.experiment().add_channel(files)
        else:
            if name is None:
                name = self.experiments[self.experiment_index].get_channel_name()
            new_channel = ChannelNoDirectory(ims = ims, dir = self.experiments[self.experiment_index].dir, name = name, annotations=annotations)
            self.experiments[self.experiment_index].add_channel_object(new_channel)
        self.imZ+=1
        self.channelSelect.addItem(self.experiments[self.experiment_index].channels[self.imZ].name)
        self.computeMaxT()
        self.drawIm()
        self.showTimedPopup(f"{name} HAS BEEN ADDED TO CHANNELS OF EXPERIMENT {self.experiments[self.experiment_index].name} ")
    
    def newMasks(self, masks = None, files = None, dir = None, name = None, exp_idx = None):
        if exp_idx is None:
            exp_idx = self.experiment_index
        enable = False
        if not self.experiments[exp_idx].has_labels():
            enable = True
        self.experiments[exp_idx].add_label(files = files, arrays = masks, name = name)

        if exp_idx == self.experiment_index:
            self.labelSelect.addItem(self.experiment().labels[-1].name)
            self.maskZ = len(self.experiment().labels)-1
            self.computeMaxT()
            self.updateDisplay()
            if enable:
                self.enableMaskOperations()
        if name is None:
            name = "new_masks"
        self.showTimedPopup(f"{name} HAS BEEN ADDED TO LABELS OF EXPERIMENT {self.experiments[self.experiment_index].name} ")
        self.checkInterpolation()
        
    def loadMasks(self, masks, exp_idx = None, name = None, contours = None):
        if type(masks) is tuple:
            mask1, mask2 = np.expand_dims(masks[0],0), np.expand_dims(masks[1],0)
            masks = np.concatenate((mask1, mask2), axis = 0)
        self.newMasks(masks, name = name, exp_idx=exp_idx)
    
    def isUpperHalf(self,ev):
        posY= ev.pos().y()
        topDist = abs(self.geometry().top() - posY)
        bottomDist = abs(self.geometry().bottom() - posY)

        if bottomDist <= topDist:
            return False
        else:
            return True

    def keyPressEvent(self, event):
        nextMaskOn  = self.maskOn

        if (event.key() == QtCore.Qt.Key_Delete or event.key() == QtCore.Qt.Key_Backspace) and self.selectedCells:
            for selectedCell in self.selectedCells.copy():
                self.deselectCell(selectedCell)
                self.deleteCell(selectedCell)
        

        if event.key() == QtCore.Qt.Key_Y:
            self.cellToChange+=1
            self.maskColors[self.cellToChange,:] = [255,255,255,255]
            self.maskChanged  = True
        
        if event.key() == QtCore.Qt.Key_B:
            self.changeBrushTypeComboBox("Brush")
        if event.key() == QtCore.Qt.Key_E:
            self.changeBrushTypeComboBox("Eraser")
        if event.key() == QtCore.Qt.Key_N:
            self.brushTypeSelect.setCurrentIndex(-1)
        if event.key() == QtCore.Qt.Key_O:
            self.changeBrushTypeComboBox("Outline")


        if event.key() == QtCore.Qt.Key_Period:
            self.brushSelect.setValue(self.brushSelect.value()+1)
        if event.key() == QtCore.Qt.Key_Comma:
            if self.brushSelect.value()-1>=1:
                self.brushSelect.setValue(self.brushSelect.value()-1)

        if event.key() == QtCore.Qt.Key_Right:
            if event.modifiers() & QtCore.Qt.ControlModifier:
                if self.tIndex+self.bigTStep<self.maxT:
                    self.tIndex+=self.bigTStep
            else:
                if self.tIndex<self.maxT:
                    self.tIndex+=1
                else:
                    self.tIndex = 0

        if event.key() == QtCore.Qt.Key_Left:
            if event.modifiers() & QtCore.Qt.ControlModifier:
                if self.tIndex-self.bigTStep>=0:
                    self.tIndex-=self.bigTStep
            else:
                if self.tIndex>0:
                    self.tIndex-=1
                else:
                    self.tIndex = self.maxT

        if event.key() == QtCore.Qt.Key_Up:
            if event.modifiers() & QtCore.Qt.ControlModifier:
                if self.maskZ<self.experiment().get_num_labels()-1:
                    self.maskZ+=1
                else:
                    self.maskZ = 0
                nextMaskOn = True
            else:
                if self.imZ<self.experiment().get_num_channels()-1:
                    self.imZ+=1
                else:
                    self.imZ = 0

        if event.key() == QtCore.Qt.Key_Down:
            if event.modifiers() & QtCore.Qt.ControlModifier:
                if self.maskZ>0:
                    self.maskZ-=1
                else:
                    self.maskZ = self.experiment().get_num_labels()-1
                nextMaskOn = True
            else:
                if self.imZ>0:
                    self.imZ-=1
                else:
                    self.imZ = self.experiment().get_num_channels()-1


        
        if self.maskOn == False and nextMaskOn == True:
            self.addMask()
        self.updateDisplay()

    def toggleMask(self):
        if not self.emptying:
            if not self.maskOnCheck.isChecked():
                self.hideMask()
            else:
                self.addMask()
    
    def toggleProb(self):
        self.probOn = self.probOnCheck.isChecked()
        self.drawIm()

    def hideMask(self):
        self.maskOn = False
        self.drawMask()

    def addMask(self):
        self.prevMaskOn = False
        self.maskOn = True
        self.drawMask()

    def drawMask(self):
        self.currMask = self.getCurrMask()
        self.numObjs  = count_objects(self.currMask)
        self.currMaskDtype = str(self.currMask.dtype)

        if self.cellNumButton.isChecked():
            self.clearCellNums()
            self.addCellNumsToCurrIm()
        else:
            self.pg_mask.setImage(self.maskColors[self.currMask], autolevels =False, levels = [0,255])
        
        if self.contourOn:
            self.drawContours()
        
        self.maskChanged = False

        # change Label select drop down "remotely"
        self.labelSelectRemote()
        self.updateDataDisplay()

    def changeMaskType(self, id):
        if self.maskType != id:
            self.maskType = id
            self.maskColors = self.cmaps[self.maskType].copy()
            self.probOn = id == 1
            self.cmap = self.cmaps[self.maskType]
            self.drawMask()
    
    def showMotherDaughter(self, mother, daughter, birth):
        self.tIndex = birth 
        self.drawIm()

        self.contourOn = False
        self.drawMask()

        contour = self.getCurrContours().copy()
        self.addSpecificContours(contour, [mother, daughter], [[255,0,0,255], [0,255,0,255]])
    
    def getDaughters(self, cellNum):
        potentialDaughters = self.label().celldata.daughters[cellNum-1]
        daughters = list(np.where(potentialDaughters)[0]+1)
        return daughters
        
    def drawContours(self):
        contour = self.getCurrContours().copy()
        color = np.array(self.goldColor)
        if self.selectedCells:
            for i, cell in enumerate(self.selectedCells):
                color = self.selectedCellContourColors[i]
                self.addSpecificContours(contour, [cell], [color])
        else:
            self.pg_mask.addMask(contour, color)
    
    def addSpecificContours(self, contour, nums, colors):
        i = 0
        for cell,color in zip(nums, colors):
            newcontour = contour.copy()
            newcontour[contour!=cell] = 0
            self.pg_mask.addMask(newcontour, color, update = i == len(nums)-1)
            i+=1

    def getCurrImSet(self):
        if not self.imLoaded:
            return np.zeros((512,512))
        return self.experiments[self.experiment_index].get_channel(idx = self.imZ)
    
    def getCurrMaskSet(self):
        return self.experiments[self.experiment_index].get_label("labels", idx = self.maskZ)
    
    def getCurrIm(self):
        if self.tIndex > self.channel().max_t():
            return np.zeros((512,512))
        if self.probOn:
            im =  self.floatCmap[self.experiment().get_label("probability", idx = self.maskZ, t = self.tIndex)].astype(np.uint8)
            return im
        else:
            return self.experiments[self.experiment_index].get_channel(idx = self.imZ, t = self.tIndex)
    
    def getCurrMask(self):
        if not self.experiment().has_labels():
            return np.zeros((512,512), dtype = np.uint8)
        if self.tIndex > self.label().max_t():
            return np.zeros((512,512), dtype = np.uint8)
        if self.maskOn:
            return self.experiment().get_label("labels", t = self.tIndex, idx = self.maskZ)
        else:
            return np.zeros(self.experiment().shape(), dtype = np.uint8)

    def getCurrContours(self):
        return self.experiment().get_label("contours", idx = self.maskZ, t = self.tIndex)

    def getCurrContourSet(self):
        return self.experiment().get_label("contours", idx = self.maskZ)

    def getCurrFrames(self):
        return (self.getCurrIm(), self.getCurrMask())
    
    def bootUp(self):
        self.pg_im.setImage(self.currIm, autoLevels=False, levels = [0,0])
        self.pg_mask.setImage(self.currMask, autolevels  =False, levels =[0,0], lut = self.maskColors)

    def updateDisplay(self):

        if self.maskChanged:
            self.drawMask()
        
        if self.imChanged:
            self.drawIm()
        
        if self.tChanged:
            if self.win.underMouse() and self.labelX and self.labelY:
                x,y = self.labelX, self.labelY
                val = self.currMask[y,x]
                if int(val)==0 or not self.maskOn:
                    val = self.currIm[y,x]
            else:
                x,y,val = None,None,None
            self.updateDataDisplay(x=x,y=y,val=val)
            self.tChanged = False
    
    def drawIm(self): 
        self.saturation = self.computeSaturation()
        self.currIm = self.getCurrIm()
        self.currImDtype = str(self.currIm.dtype)
        if self.probOn:
            self.pg_im.setImage(self.currIm, autoLevels = False, levels = [0,255])
        else:
            self.pg_im.setImage(self.currIm, autoLevels = False)
            self.pg_im.setLevels(self.saturation)
        #self.saturationSlider.setMaximum(self.imData.channels[self.imZ][:,:,:].max())
        #self.saturationSlider.resetLevels(self.saturation)
        self.channelSelectRemote()
        self.imChanged  = False

        self.updateDataDisplay()
        
    def resetAutoSaturation(self):
        self.saturation = self.computeSaturation()
        self.pg_im.setLevels(self.saturation)
        self.saturationSlider.resetLevels(self.saturation)

    def computeSaturation(self):
        saturation =  self.experiment().channels[self.imZ].get_saturation(self.tIndex)
        return saturation
                                
    def selectCell(self, pos):
        pos = [int(pos.y()), int(pos.x())]
        cellNum = self.currMask[pos[0], pos[1]]
        self.selectCellFromNum(int(cellNum))
    
    def selectCellFromNum(self, cellNum):
        if self.showLineages:
            self.deselectAllCells()
            for num in self.getDaughters(cellNum)+[cellNum]:
                self.selectedCells.append(num)
                self.selectedCellContourColors.append([255,0,0,255])
                self.maskColors[num,:] = np.array(self.goldColor, dtype = np.uint8)

        elif self.showMotherDaughters and self.hasLineageData:
            self.deselectAllCells()
            self.selectedCells.append(cellNum)
            self.selectedCellContourColors.append([255,0,0,255])
            self.maskColors[cellNum,:] = np.array(self.goldColor, dtype = np.uint8)
            motherNum = self.getMotherFromLineageData(cellNum)
            if motherNum:
                if motherNum not in self.selectedCells:
                    self.selectedCells.append(motherNum)
                    self.selectedCellContourColors.append([0,255,0,255])
                    self.maskColors[motherNum,:] = np.array(self.goldColor, dtype = np.uint8)
                else:
                    motherIdx = self.selectedCells.index(motherNum)
                    self.selectedCellContourColors[motherIdx] = [0,255,0,255]
        
        elif cellNum in self.selectedCells:
            self.deselectCell(cellNum)
    
        else:
            self.selectedCells.append(cellNum)
            self.selectedCellContourColors.append([255,0,0,255])
            self.maskColors[cellNum,:] = np.array(self.goldColor, dtype = np.uint8)
        self.drawIm()
        self.drawMask() 
        if self.pWindow and self.pWindow.hasSingleCellPlots():
            self.pWindow.updateSingleCellPlots()
    
    def getMotherFromLineageData(self,cell):
        ld = self.label().celldata.mothers
        mother = ld["mother"].iloc[cell-1]
        if pd.notna(mother):
            return (int(mother))
        else:
            return None

    def deselectCell(self, num):
        self.maskColors[num,:] = self.cmap[num,:].copy()
        del self.selectedCellContourColors[self.selectedCells.index(num)]
        self.selectedCells.remove(num)

    def getCellCenters(self, plotmask):
        """Get approximate locations for cell centers"""
        vals = np.unique(plotmask).astype(int)
        vals = np.delete(vals,np.where(vals==0)) 
        xtemp = []
        ytemp = []
        for k in vals:
            y,x = (plotmask==k).nonzero()
            sample = np.random.choice(len(x), size=30, replace=True)
            meanx = np.mean(x[sample])
            meany = np.mean(y[sample])
            xtemp.append(int(round(meanx)))
            ytemp.append(int(round(meany)))
        return vals, xtemp, ytemp

    def getBlobAttributes(self):
        im, mask = self.getCurrFrames()
        im = np.array([normalize_im(i) for i in im], dtype = np.float32)
        labelMask = binarize_relabel(mask)
        cellVals = np.unique(labelMask)[1:]
        radiusVals = np.zeros_like(cellVals)
        intensityVals = np.zeros_like(cellVals, dtype = np.float32)
        for i, label in enumerate(cellVals):
            cellmask = labelMask == label
            area = np.sum(cellmask.astype(np.uint8))
            radius = math.sqrt(area / math.pi)
            radiusVals[i] = (radius)
            intensityVals[i] = np.percentile(im[cellmask], 0.1)
        return np.min(radiusVals), np.max(radiusVals), np.min(intensityVals)
        
    def doFlou(self):
        dlg = FlouParamDialog(self)

        if dlg.exec():
            self.segButton.setEnabled(False)
            self.segButton.setStyleSheet(self.stylePressed)
            params = dlg.getData()
            
            if params['extract from curr mask']:
                minRadius, maxRadius, thresh = self.getBlobAttributes()
                
            else:
                minSize, maxSize, thresh = params["min area"], params["max area"], params["threshold"]
                minRadius, maxRadius = math.sqrt(minSize / math.pi), math.sqrt(maxSize / math.pi)

            print(minRadius, maxRadius, thresh)

            tStart,tStop = int(params["T Start"]), int(params["T Stop"])
            channelIndex = self.channelSelect.findText(params['channel'])
            ims = self.imData.channels[channelIndex][tStart:tStop+1,:,:]
            if params['label']:
                labelIndex = self.labelSelect.findText(params['label'])
                if params["use contours"]:
                    labels = self.maskData.channels[labelIndex][0,tStart:tStop+1,:,:]
                else:
                    labels = self.maskData.contours[labelIndex][tStart:tStop+1,:,:]
            else:
                labels = None
            name = self.channelSelect.currentText() + "-blob_detection"
            worker = Worker(self, lambda: self.loadMasks(Blob.run(ims, thresh, minRadius, maxRadius, labels), name = name), self.segButton)
            worker.finished.connect(self.handleFinished)

            try:
                self.beginThread(worker)
            except MemoryError:
                error_dialog  = QErrorMessage()
                error_dialog.showMessage(f"Cannot Segment Until Other Processes are Finished")
                self.activateButton(self.segButton)
                return
        else:
            return

    def getModels(self):
        for modelName in self.modelNames:
            if os.path.exists(os.path.join("models",modelName,"model.py")):
                importlib.import_module(self.getPkgString(modelName))
    
    def getPkgString(self, string):
        return f"yeastvision.models.{string}.model"

    def getModelClass(self, modelName):
        
        module = importlib.import_module(self.getPkgString(modelName))
        modelClass = getattr(module, capitalize(modelName))
        return modelClass
    
    def showTW(self):
        if not self.maskLoaded:
            self.showError("Load Labels before training")
            return
        
        weightName = self.modelChoose.currentText() 
        if weightName == "":
            self.showError("Select A Model First")
            return
        
        try:
            self.getTrainData()
        except IndexError:
            return
        

        modelType = self.modelTypes[self.modelNames.index(weightName)]
        weightPath = join(MODEL_DIR, modelType, weightName)

        modelClass = self.getModelClass(modelType)

        weightPath += modelClass.prefix

        if not self.trainModelName or not self.trainModelSuffix:
            d =  datetime.now()
            d =  d.strftime("%Y_%m_%d_%H_%M")
            suffix = f"{d}"
            self.trainModelName = f"{modelType}_{suffix}"
            self.trainModelSuffix = suffix
    
        
        TW = TrainWindow(modelType, weightPath, self)
        if TW.exec_():
            data = TW.getData()
            data["model_type"] = modelType
            self.trainModelName = data["model_name"]
            self.train(data, modelType, weightPath)
            
        else:
            return


    def train(self, data, modelType, weightPath, autorun = True):
        # weights are overwritten if model is to be trained from scratch
        #check_gpu()
        torch.cuda.empty_cache() 
        if data["scratch"]:
            weightPath = None
        # directory to save weights is current image directory stored in reader
        data["dir"] = os.getcwd()
        modelCls = self.getModelClass(modelType)
        #check_gpu()
        # model is initiated with default hyperparams
        self.model = modelCls(modelCls.hyperparams, weightPath)
        self.model.train(self.trainIms, self.trainLabels, data)
        #check_gpu()
        weightPath = join(data["dir"], "models", data["model_name"])+self.model.prefix
        print("Saving new weights to",weightPath)

        if autorun and self.trainTStop < self.maxT:
            im = self.getCurrMaskSet()[self.trainTStop:self.trainTStop+1,:,:]
            newIms, newProbIms = modelCls.run(im,None,weightPath)
            self.tIndex = self.trainTStop
            self.label().insert(newIms, self.tIndex, new_prob_im = newProbIms)
        elif self.trainTStop == self.maxT:
            self.tIndex = self.trainTStop
            print("Not enough data to autorun")
        self.drawMask()

    def hasProbIm(self):
        return self.label().has_probability

    def getTrainData(self):
        ims = self.getCurrImSet()
        labels = self.getCurrMaskSet()
        
        if ims[0].shape[1:]!=labels[0].shape[1:]:
            maskShape = labels[0].shape[0,:,:]
            imShape = ims[0].shape[0,:,:]
            self.showError(f"Masks of shape {maskShape} does not match images of shape {imShape}")
            raise IndexError
        
        imT, maskT = len(ims), len(labels)
        maskT = get_end_of_cells(labels)
        no_cell_frames = get_frames_with_no_cells(labels)

        files = self.channel().get_file_names()
        print(files)
        if len(files) == imT:
            self.trainFiles = files
        else:
            self.trainFiles = [f"index_{i}" for i in range(len(ims))]


        self.trainTStop = min(imT, maskT)

        self.trainIms = [im for i,im in enumerate(ims[:self.trainTStop+1]) if i not in no_cell_frames]
        self.trainLabels = [im for i,im in enumerate(labels[:self.trainTStop+1]) if i not in no_cell_frames]
        self.trainFiles = [file for i, file in enumerate(self.trainFiles[:self.trainTStop+1]) if i not in no_cell_frames]

    
    def getModelNames(self):
        self.modelNames = []
        self.modelTypes = []
        dirs = [dir for dir in glob.glob(join(MODEL_DIR,"*")) if (os.path.isdir(dir) and "__" not in dir)]
        for dir in dirs:
            dirPath, dirName = os.path.split(dir)

            for dirFile in glob.glob(join(dir,"*")):
                dirPath2, dirName2 = os.path.split(dirFile)
                if dirName.lower() in dirName2.lower():
                    if "." not in dirName2 or (dirFile.endswith("h5") or dirFile.endswith("hdf5")):
                        self.modelNames.append(os.path.split(dirFile)[1].split(".")[0])
                        self.modelTypes.append(dirName)
        if len(self.modelNames)==0:
            self.showError("Weights Have Not Been Downloaded. Find the weights at https://drive.google.com/file/d/1J3R4JKILkQNM0Ap-MKxqv61oAxObSjBo/view?usp=drive_link. Then run install-weights in the same directory as the downloaded zip file.")

    def getModelWeights(self, name = "artilife"):
        weights = []
        dirs = [dir for dir in glob.glob(join(MODEL_DIR,"*")) if (os.path.isdir(dir) and "__" not in dir)]
        for dir in dirs:
            dirPath, dirName = os.path.split(dir)

            if name in dirName:
                for dirFile in glob.glob(join(dir,"*")):
                    dirPath2, dirName2 = os.path.split(dirFile)
                    if dirName.lower() in dirName2.lower():
                        if "." not in dirName2 or (dirFile.endswith("h5") or dirFile.endswith("hdf5")):
                            weights.append(os.path.split(dirFile)[1].split(".")[0])
        return weights


    def segment(self, output, modelClass, ims, params, weightPath, modelType, exp_idx):
        #check_gpu()
        tStart, tStop = int(params["T Start"]), int(params["T Stop"])
        imName = params["Channel"]

        if modelClass.__name__ == "ArtilifeFullLifeCycle" :
            counter = 0
            for labelName, outputTup in output.items():
                if type(outputTup[0]) == np.ndarray:
                    counter += 1
                    templates = []
                    for i in [0,1]:
                        template = np.zeros_like(ims, dtype = outputTup[i].dtype)
                        template[tStart:tStop+1] = outputTup[i]
                        templates.append(template)
                    outputTup = (templates[0], templates[1])
                    self.loadMasks(outputTup, exp_idx=exp_idx, name = f"{imName}_{labelName}")
            if params["Time Series"]:
                if params["Mating Cells"]:
                    mating_idx = self.maskZ -1
                    cell_idx = self.maskZ
                    self.getMatingLineages(cell_idx, mating_idx)

                idx = self.maskZ - counter
                self.updateCellData(idx = idx)

        else:
            templates = []
            for i in [0,1]:
                template = np.zeros_like(ims, dtype = output[i].dtype)
                template[tStart:tStop+1] = output[i]
                templates.append(template)
            outputTup = (templates[0], templates[1])
            self.loadMasks(outputTup, exp_idx=exp_idx, name = f"{imName}_{modelType}_{tStart}-{tStop}")
        
        torch.cuda.empty_cache()
        del modelClass
        del output
        #check_gpu()
        
    def evaluate(self):
        if not self.maskLoaded:
            return
        params = {label: False for label in self.labelSelect.items()}
        paramtypes = [bool] * len(params)
        labelSelects = ["validation"]
        evalDlg = GeneralParamDialog(params, paramtypes, "Evaluate Predictions", self, labelSelects = labelSelects)

        if evalDlg.exec():
            data = evalDlg.getData()
            validationName = data[labelSelects[0]]
            masksTrueIndex = self.labelSelect.items().index(validationName)
            masksTrue = self.experiment().labels[masksTrueIndex].npzdata["labels"]

            toValidate = {}
            for labelName in params:
                if data[labelName]:
                    toValidate[labelName] = (self.experiment().labels[self.labelSelect.items().index(labelName)].npzdata["labels"])
            
            ious = [0.25, 0.50, 0.75, 0.90]
            statsDict = self.getStatsDict(masksTrue, toValidate, ious)
            self.evalWindow = plot.EvalWindow(self,statsDict, ious)
            self.evalWindow.show()
    
    def getStatsDict(self, validation, toValidate, ious):

        validation = list(validation.copy())
        statsDict = {"precision": [], "recall": [], "f1": [], "average_precision":[]}
        for toValidateName, labels in tqdm(toValidate.items()):
            ap, tp, fp, fn = average_precision(validation, list(labels))
            precision = np.mean((tp + (tp+fp)), axis = 0)
            recall = np.mean((tp+(tp+fn)), axis = 0)
            f1 = np.mean(tp/(tp+0.5*(fp+fn)), axis = 0)
            ap = np.mean(ap, axis = 0)
            statsDict["precision"].append((toValidateName, precision))
            statsDict["recall"].append((toValidateName, recall))
            statsDict["f1"].append((toValidateName,f1))
            statsDict["average_precision"].append((toValidateName, ap))
        return statsDict        
    
    def getAllItems(self, combo):
        return [combo.itemText(i) for i in range(combo.count())]
    
    def getNewLabelName(self, potentialName):
        allNames = self.getAllItems(self.labelSelect)
        if not (allNames):
            return potentialName
        n = (np.char.find(np.array(allNames), potentialName)+1).sum()
        if n>0:
            return f"{potentialName}-{n}"
        else:
            return potentialName
    
    def getNewChannelName(self, potentialName):
        allNames = self.getAllItems(self.channelSelect)
        if not allNames:
            return potentialName
        n = (np.char.find(np.array(allNames), potentialName)+1).sum()
        if n>0:
            return f"{potentialName}-n"
        else:
            return potentialName
    
    def runLongTask(self, worker, finished, button):
        if len(self.threads)+1>self.idealThreadCount:
            self.showError("Too Many Threads Running At This Time")
            self.activateButton(button)
            return
        
        thread = QThread()

        self.workers.append(worker)
        self.threads.append(thread)

        worker.moveToThread(thread)
        # Step 5: Connect signals and slots
        thread.started.connect(worker.run)
        worker.finished.connect(finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(self.handleFinished)
        thread.start()
        self.deactivateButton(button)
        thread.finished.connect(lambda: self.activateButton(button))
        self.updateThreadDisplay()
    
    def interpolationFinished(self, ims, exp_idx, name, annotations, interpolation):
        experiment = self.experiments[exp_idx]
        print(len(ims))
        new_channel = InterpolatedChannel(interpolation=interpolation, ims = ims, dir = experiment.dir , name = name, annotations=annotations)
        experiment.add_channel_object(new_channel)
        self.imZ+=1
        new_name = self.experiment().channels[self.imZ].name
        self.newInterpolation = True
        self.channelSelect.addItem(new_name)
        self.newInterpolation = False
        self.computeMaxT()
        self.drawIm()
        self.showTimedPopup(f"{name} HAS BEEN ADDED TO CHANNELS OF EXPERIMENT {self.experiments[self.experiment_index].name} ")
        self.checkInterpolation()
        


    def computeModels(self):
        #check_gpu()
        weightName = self.modelChoose.currentText() 
        if weightName == '':
            return
        modelType = self.modelTypes[self.modelNames.index(weightName)]
        weightPath = join(MODEL_DIR, modelType, weightName)

        modelClass = self.getModelClass(modelType)

        weightPath += modelClass.prefix

        self.deactivateButton(self.modelButton)
        dlgCls = ModelParamDialog
        dlg = dlgCls(modelClass.hyperparams, modelClass.types, modelType, self)
        
        if dlg.exec():
            params = dlg.getData()
            channel = params["Channel"]
            channelIndex = self.channelSelect.findText(channel)
            ims = self.experiment().get_channel(idx = channelIndex)
            worker = SegmentWorker(modelClass,ims, params, self.experiment_index, weightPath, modelType)
            self.runLongTask(worker, 
                             self.segment,
                             self.modelButton)
        else:
            self.activateButton(self.modelButton)

    
    def computeArtilifeModel(self):
        modelType = "artilife"
        modelClass = ArtilifeFullLifeCycle
        self.deactivateButton(self.artiButton)
        dlgCls = ArtilifeParamDialog
        dlg = dlgCls(modelClass.hyperparams, modelClass.types, modelType, self)
        
        if dlg.exec():
            params = dlg.getData()
            channel = params["Channel"]
            weightPath = params["artiWeights"]
            channelIndex = self.channelSelect.findText(channel)
            ims = self.experiment().get_channel(idx = channelIndex)
            worker = SegmentWorker(modelClass,ims, params, self.experiment_index, weightPath, modelType)
            self.runLongTask(worker, 
                             self.segment,
                             self.modelButton)
        else:
            self.activateButton(self.modelButton)

    def get_curr_state(self):
        return {"experiment_index":self.experiment_index,
                "imZ":self.imZ,
                "maskZ":self.maskZ,
                "t":self.tIndex}

    def interpolateButtonClicked(self):
        if self.label().max_t() == 0:
            self.showError("Error: More than one frame must be present to interpolate")

        dlg = GeneralParamDialog({"2x":False, "4x": False, "8x": False, "16x": False}, [bool,bool,bool,bool], "choose interpolation level", self)
        if dlg.exec():
            data = dlg.getData()
            exp = None
            for data, isChecked in data.items():
                if isChecked:
                    interpolation = int(data[:-1])
                    exp = math.log(interpolation,2)
                    break
        else:
            return
    
        if exp:
            curr_im = self.channel()
            ims = curr_im.ims
            currName = curr_im.name
 
            worker = InterpolationWorker(ims, f"{currName}_{interpolation}x{InterpolatedChannel.text_id}", curr_im.annotations, self.experiment_index, interpolation, interpolate, exp)
        else:
            return

        self.runLongTask(worker, self.interpolationFinished, self.interpolateButton)
    
    def interpRemoveButtonClicked(self):
        assert isinstance(self.channel(), InterpolatedChannel)
        lengthsMatch = self.channel().max_t() == self.label().max_t()
        if not lengthsMatch:
            self.showError("Error: Length of label must match length of interpolated channel to perform this operation")
            return

        interp_bool = self.channel().interp_annotations
        print(interp_bool)
        masks = self.getCurrMaskSet()
        contours = self.getCurrContourSet()
        newMasks = [mask for i, mask in enumerate(masks) if interp_bool[i]]
        newContours = [mask for i, mask in enumerate(contours) if interp_bool[i]]
        print()
        currMaskName = self.labelSelect.currentText()
        self.loadMasks(newMasks, name = f"{currMaskName}_removeinterpolation", contours=newContours)

    def doAdaptHist(self):
        curr_im = self.channel()
        annotation_name = "adaptive_hist_eq"
        annotations = None
        if isinstance(curr_im, ChannelNoDirectory):
            annotations = copy.deepcopy(curr_im.annotations)
            for ann in annotations:
                ann.append(annotation_name)

        newIms = im_funcs.do_adapt_hist(self.getCurrImSet())
        self.newIms(ims = newIms, dir = self.experiment().dir, name = curr_im.name+f"-{annotation_name}", annotations = annotations )
      
    # def doNormalizeByIm(self):
    #     files = self.imData.files[self.imZ]
    #     currName = self.channelSelect.currentText()
    #     newIms = [normalize_im(im) for im in self.getCurrImSet()]
    #     self.addProcessedIms(newIms, files, currName+"-imNormalized", dtype = np.float32)
    
    # def doNormalizeBySet(self):
    #     files = self.imData.files[self.imZ]
    #     currName = self.channelSelect.currentText()
    #     newIms = normalize_im(self.getCurrImSet())
    #     self.addProcessedIms(newIms, files, currName+"-setNormalized", dtype = np.float32)

    def doGaussian(self):
        dlg = GeneralParamDialog({"sigma": 1}, [float], "Enter Params for Gaussian Blur", self)
        if dlg.exec():
            sigma  = dlg.getData()["sigma"]
        else:
            return
        curr_im = self.channel()
        annotation_name = "gaussian"
        annotations = self.add_annotations(curr_im, annotation_name)
        newIms = im_funcs.do_gaussian(curr_im.ims, sigma)
        self.newIms(ims = newIms, dir = self.experiment().dir, name = curr_im.name+f"-{annotation_name}", annotations = annotations)

    def doMedian(self):
        dlg = GeneralParamDialog({"kernel_size": 3}, [int], "Size of(Symmetric) Kernel for Median Blur", self)
        if dlg.exec():
            kernel_size  = dlg.getData()["kernel_size"]
        else:
            return
        curr_im = self.channel()
        annotation_name = "median"
        annotations = self.add_annotations(curr_im, annotation_name)
        newIms = im_funcs.do_median(curr_im.ims, kernel_size)
        self.newIms(ims = newIms, dir = self.experiment().dir, name = curr_im.name+f"-{annotation_name}", annotations = annotations )

    def add_annotations(self, curr_im, annotation_name):
        if isinstance(curr_im, ChannelNoDirectory):
            annotations = (copy.deepcopy(curr_im.annotations)).tolist()
            for ann in annotations:
                ann.append(annotation_name)
        else:
            annotations = [[annotation_name] for i in range(len(curr_im.ims))]
        return annotations
    
    def doRescale(self):
        curr_im = self.channel()
        ims  = curr_im.ims
        newIms = self.rescaleFromUser(ims)
        newSize = str(newIms[0].shape)
        annotation_name = f"rescaled_to_{newSize}"
        annotations = self.add_annotations(curr_im, annotation_name)
        self.newIms(ims = newIms, dir = self.experiment().dir, name = curr_im.name+f"-{annotation_name}", annotations = annotations )

    def rescaleFromUser(self, ims):
        row, col = ims[0].shape
        factor =  {"0.25x": 0.25, "0.5x": 0.5, "2x": 2, "4x": 4}
        params = {"0.25x": False, "0.5x": False, "2x": False, "4x": False,
                            "new row": row, "new col": col}
        types  = [bool, bool, bool, bool,None, None]
        dlg = GeneralParamDialog(params, types,
                            "Parameters for rescale",
                            self)
        if dlg.exec_():
            data  = dlg.getData()
            for param, t in zip(params, types):
                if t is bool:
                    if data[param]:
                        return rescaleByFactor(factor[param],ims)
            return rescaleBySize((int(data["new row"]), int(data["new col"])), ims)
        else:
            return
    
    def doZNorm(self):
        curr_im = self.channel()
        annotation_name = "z-normalized"
        annotations = self.add_annotations(curr_im, annotation_name)
        newIms = im_funcs.z_normalize_images(self.getCurrImSet())
        self.newIms(ims = newIms, dir = self.experiment().dir, name = curr_im.name+f"-{annotation_name}", annotations = annotations )
    
    def deactivateButton(self, button):
        button.setEnabled(False)
        button.setStyleSheet(self.stylePressed)
        
    def activateButton(self, button):
        button.setEnabled(True)
        button.setStyleSheet(self.styleUnpressed)
    
    def getCurrImDir(self):
        return self.experiment().channels[self.imZ].dir
    
    def saveIms(self):
        if not self.imLoaded:
            self.showError("No Images Loaded")
            return
        defaultFileName = join(self.getCurrImDir(), self.channelSelect.currentText().replace(" ", ""))
        path, _ = QFileDialog.getSaveFileName(self, 
                            "save images to a directory", 
                            defaultFileName)
        if path:
            write_images_to_dir(path, self.getCurrImSet())
    
    def saveMasks(self):
        if not self.maskLoaded:
            self.showError("No masks to save")
        defaultFileName = join(self.getCurrImDir(), self.labelSelect.currentText().replace(" ", ""))
        path, _ = QFileDialog.getSaveFileName(self, 
                            "save masks to a directory", 
                            defaultFileName)
        if path:
            write_images_to_dir(path, self.getCurrMaskSet())
    
    def saveFigure(self):
        if (not self.imLoaded):
            self.showError("No Images Loaded")
            return

        defaultFileName = join(self.getCurrImDir(), self.labelSelect.currentText().replace(" ", "") + "-figure")
        path, _ = QFileDialog.getSaveFileName(self, 
                            "save figures to directory", 
                            defaultFileName)
        if path:
            write_images_to_dir(path, self.createFigure())
    
    def createMaskOverlay(self):
        if (not self.imLoaded):
            self.showError("No Images Loaded")
            return

        defaultFileName = join(self.getCurrImDir(), "label_overlay_figure.png")
        path, _ = QFileDialog.getSaveFileName(self, 
                            "save figures to directory", 
                            defaultFileName)
        if path:
            imsave(path, self.overlayMasks())
    
    def overlayMasks(self):
        
        def use_label(name, data):
            return data[name]["contours"] or data[name]["labels"]

        def get_labels_to_use(data):
            new_data = copy.deepcopy(data)
            for label in new_data.keys():
                if not use_label(label,data):
                    del data[label]
            return sort_by_nested_key(data, "order")
        
        def sort_by_nested_key(nested_dict, sort_key):
            # Sorting the outer dictionary based on a specific key of the inner dictionaries
            sorted_dict = sorted(nested_dict.items(), key=lambda x: x[1][sort_key])
            return OrderedDict(sorted_dict)


        dlg = FigureDialog(self)
        if dlg.exec():
            data = get_labels_to_use(dlg.get_data())
            im = self.getCurrIm()
            to_overlay = []
            is_contours = []
            t =  self.tIndex
            for label, values in data.items():
                if values["contours"]:
                    to_overlay.append(self.experiment().get_label("contours", name=label, t=t))
                    is_contours.append(True)
                if values["labels"]:
                    to_overlay.append(self.experiment().get_label("labels", name=label, t=t))
                    is_contours.append(False)
            return overlay_masks_on_image(im, to_overlay, is_contours, alpha = 50)



            
        


    def zNormalization(self):
        pass 

    def createFigure(self):
        ims = self.getCurrImSet()
        colored_masks = []
        masks = []
        currT = self.tIndex
        for i in range(self.channel().max_t()+1):
            self.tIndex = i
            self.updateDisplay()
            colored_masks.append(self.pg_mask.image[:,:,:])
            masks.append(self.currMask)
        figures = np.array([overlay_images(ims[i], masks[i], colored_masks[i]) for i in range(len(masks))])
        self.tIndex = currT
        self.updateDisplay()
        return figures
    
    def saveCellData(self):
        if not self.maskLoaded:
            self.showError("No Label Data Loaded")
            return
        
        labelName = self.labelSelect.items()[self.maskZ]
        labelFileName = labelName.replace(" ", "")
        if self.hasCellData():
            data = exportCellData(self.label().celldata.cell_data, self.label().celldata.get_cell_info())
            defaultFileName = join(self.experiment().dir, labelFileName + "_celldata.csv")
            path, _ = QFileDialog.getSaveFileName(self, 
                            "save data as csv", 
                            defaultFileName)
            data.to_csv(path)
        else:
            self.showError(f"No Cell Data for {labelName}")
    
    def saveLineageData(self):
        if not self.maskLoaded:
            self.showError("No Label Data Loaded")
            return
        
        labelName = self.labelSelect.items()[self.maskZ]
        labelFileName = labelName.replace(" ", "")
        if self.hasLineageData():
            motherdata, daughterdata = self.label().celldata.mothers, self.label().celldata.daughters
            fname1= join(self.experiment().dir, labelFileName + "_mother_daughters.csv")
            fname2 = join(self.experiment().dir, labelFileName + "_daughter_array.csv.csv")
            path, _ = QFileDialog.getSaveFileName(self, 
                            "save data as csv", 
                            fname1)
            if path:
                np.savetxt(fname2, daughterdata, delimiter=",")
                motherdata.to_csv(fname1)
        else:
            self.showError(f"No Cell Data for {labelName}")
    
    
    def saveHeatMaps(self):
        if not self.maskLoaded:
            self.showError("No Label Data Loaded")
            return
        
        labelName = self.labelSelect.items()[self.maskZ]
        labelFileName = labelName.replace(" ", "")
        if self.hasCellData():
            data = getHeatMaps(self.label().celldata.cell_data)
            defaultFileName = join(self.experiment().dir, labelFileName + "_heatmaps.tif")
            path, _ = QFileDialog.getSaveFileName(self, 
                            "save heatmaps as stack tif", 
                            defaultFileName)
            imsave(path, data) 
        else:
            self.showError(f"No Cell Data for {labelName}")

    def loadMasksAndImages(self):
        dir = QFileDialog.getExistingDirectory(self, "Choose Directory with Masks and Images Together")
        fileSample = os.listdir(dir)
        if not fileSample:
            self.showError("Directory Contains No Images")
            return
        ending = "."+fileSample[0].split('.')[-1]
        if ending not in ImageData.fileEndings:
            self.showError("Directory Contains No Images")
            return
        files = sorted(glob.glob(join(dir,"*"+ending)))
        masks = sorted(glob.glob(join(dir,"*masks"+ending)))
        ims = [file for file in files if file not in masks]
        self.imData.loadFileList(ims)
        self.maskData.loadFileList(masks)
                       

    def loadByFileName(self):
        dir = QFileDialog.getExistingDirectory(self, "Choose Directory with Multiple Channels")
        fileSample = os.listdir(dir)
        if not fileSample:
            self.showError("Directory Contains No Images")
            return
        ending = "."+fileSample[0].split('.')[-1]
        if ending not in ImageData.fileEndings:
            self.showError("Directory Contains No Images")
            return
        
        files = sorted(glob.glob(join(dir,"*"+ending)))

        dlg = DirReaderDialog(dir, [name for name in fileSample if name.endswith(ending)])

        if dlg.exec_():
            fileIds = dlg.getData()

            for fileId in fileIds:
                contains = False
                for file in files:
                    if fileId in file:
                        contains = True
                        break
                if not contains:
                    self.showError(f"File Ending {fileId} does not exist in any path \nwithin {dir}")
            self.imData.loadMultiChannel(fileIds, files)

    def closeEvent(self, event):
        for thread in self.threads:
            self.closeThread(thread)
        if self.pWindow:
            self.pWindow.close()
            self.pWindow = None
        if self.lineageWindow:
            self.lineageWindow.close()
            self.lineageWindow = None
    
    def getCurrImDir(self):
        return self.experiment().channels[self.imZ].dir
    
    def checkGPUs(self):
        self.checkTfGPU()
        self.checkTorchGPU()

    def checkTorchGPU(self):
        self.torch = check_torch_gpu()

    def checkTfGPU(self):
        self.tf = check_tf_gpu()
    
    def checkDataAvailibility(self):
        hasLineages = self.hasLineageData()
        self.hasLineageBox.setChecked(hasLineages)
        self.hasCellDataBox.setChecked(self.hasCellData())
        self.plotButton.setEnabled(self.hasCellData())
        self.showMotherDaughtersButton.setEnabled(hasLineages)
        self.showLineageButton.setEnabled(hasLineages)
        self.showTreeButton.setEnabled(hasLineages)
    
    def checkInterpolation(self):
        hasMasks = self.experiment().has_labels()
        currIsInterp = isinstance(self.channel(), InterpolatedChannel)

        if not currIsInterp or not hasMasks:
            self.disableInterpRemove()
        else:
            self.enableInterpRemove()
        
        if currIsInterp:
            self.disableInterp()
        else:
            self.enableInterp()

    def disableInterp(self):
        self.interpolateButton.setEnabled(False)
        self.interpolateButton.setStyleSheet(self.styleInactive)
    
    def enableInterp(self):
        self.interpolateButton.setEnabled(True)
        self.interpolateButton.setStyleSheet(self.styleUnpressed)

    
    def disableInterpRemove(self):
        self.interpRemoveButton.setEnabled(False)
        self.interpRemoveButton.setStyleSheet(self.styleInactive)
    
    def enableInterpRemove(self):
        self.interpRemoveButton.setEnabled(True)
        self.interpRemoveButton.setStyleSheet(self.styleUnpressed)



@profile
def main():
    print("running main")
    app = QApplication([])
    im_path = "test_phase.tif"
    mask_path = "test_mask.tif"
    cdc_path = "test_cdc.tif"
    mainThread = app.instance().thread()
    window = MainWindow(mainThread, im_path, mask_path)
    window.show()
    toc = process_time()
    app.exec_()        

if __name__ == "__main__":
    main()




        