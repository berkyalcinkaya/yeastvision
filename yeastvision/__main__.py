from yeastvision.data.ims import Experiment, ChannelNoDirectory, InterpolatedChannel
from yeastvision.parts.canvas import ImageDraw, ViewBoxNoRightDrag
from yeastvision.parts.guiparts import *
from yeastvision.parts.workers import FiestWorker, SegmentWorker, TrackWorker, InterpolationWorker
from yeastvision.parts.dialogs import *
from yeastvision.track.fiest.full_lifecycle_utils import track_general_masks
from yeastvision.track.track import track_to_cell, track_proliferating
from yeastvision.track.data import LineageData, TimeSeriesData
from yeastvision.track.cell import LABEL_PROPS, IM_PROPS, EXTRA_IM_PROPS, average_axis_lengths, exportCellData, getHeatMaps, getPotentialHeatMapNames
import yeastvision.plot.plot as plot
from yeastvision.utils import *
import yeastvision.ims.im_funcs as im_funcs
from yeastvision.ims.interpolate import interpolate_intervals, rife_weights_loaded, deinterpolate, RIFE_WEIGHTS_PATH, RIFE_WEIGHTS_NAME
import yeastvision.parts.menu as menu
from yeastvision.models.utils import MODEL_DIR, getBuiltInModelTypes, getModelLoadedStatus, produce_weight_path, getModelsByType
from yeastvision.install import TEST_MOVIE_DIR, TEST_MOVIE_URL, install_test_ims, install_weight, install_rife
from yeastvision.disk.reader import ImageData
from yeastvision.parts.fiest_wizard import FiestWizard
from yeastvision.parts.fiest_full_lifecycle_wizard import FiestFullLifeCycleWizard
from yeastvision.track.fiest.track import fiest_basic, fiest_basic_with_lineage, fiest_full_lifecycle
import os
import torch
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QApplication, QSplitter, QGroupBox, QPushButton, QMessageBox,
                             QStatusBar, QFileDialog, QSpinBox, QLabel, QWidget, QComboBox, 
                             QSizePolicy, QGridLayout, QProgressBar, QShortcut, QVBoxLayout)
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QKeySequence
from QSwitchControl import SwitchControl
import pyqtgraph as pg
import matplotlib.pyplot as plt
import cv2
import glob
from os.path import join
import importlib
from datetime import datetime
import pandas as pd
import math
from skimage.io import imsave, imread
from cellpose.metrics import average_precision
from tqdm import tqdm
import copy
from collections import OrderedDict
import argparse
import shutil
from skimage.measure import regionprops_table
import warnings 
import functools

os.environ['QT_LOGGING_RULES'] = '*.warning=false'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache() 
warnings.filterwarnings("ignore") #TODO: remove this for deployment

global logger
logger, _ = logger_setup()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, dir = None, dir_num_channels = None, dir_ims_only=False):
        super(MainWindow, self).__init__()
        pg.setConfigOption('imageAxisOrder', 'row-major')
        self.setWindowTitle("yeastvision")
        self.setGeometry(100, 100, 1200, 1000)
        self.setAcceptDrops(True)

        self.idealThreadCount = 2
        self.sessionId = self.getCurrTimeStr()
        
        self.INTERP_MODEL_NAME = "RIFE"
        self.PROB_ID = 1
        self.FLOW_ID = 2
        
        self.goldColor = [255,215,0,255]

        self.setStyleSheet("QMainWindow {background-color: rgb(30,31,32);}")
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
        self.labelstyle = """QLabel{
                            color: white
                            } 
                         QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }"""
        self.statusbarstyle = ("color: white;" "background-color : black")
        self.boldfont = QtGui.QFont("Arial", 15, QtGui.QFont.Bold)
        self.medfont = QtGui.QFont("Arial", 13)
        self.smallfont = QtGui.QFont("Arial", 11)
        self.headings = ('color: rgb(200,10,10);')
        self.dropdowns = ("color: white;"
                        "background-color: rgb(40,40,40);"
                        "selection-color: white;"
                        "selection-background-color: rgb(50,100,50);")
        self.checkstyle = "color: rgb(190,190,190);"
        
        self.firstMaskLoad = True
        self.measure_window = None
        self.multi_label_window = None
        
        self.getModelNames()
        self.importModelClasses()
        self.WEIGHTS_LOADED_STATUS = self.get_weights_loaded_status()

        # Setting up the central widget and main layout
        self.cwidget = QtWidgets.QWidget(self)
        self.mainLayout = QtWidgets.QGridLayout(self.cwidget)
        self.setCentralWidget(self.cwidget)

        # Create a QSplitter to separate left controls and main window
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.mainLayout.addWidget(self.splitter, 1, 0, 1, 1)

        # Left panel widget for controls
        self.leftWidget = QtWidgets.QWidget()
        self.leftLayout = QtWidgets.QGridLayout(self.leftWidget)
        self.splitter.addWidget(self.leftWidget)

        # Right panel widget for the image and data selectors
        self.rightWidget = QtWidgets.QWidget()
        self.rightLayout = QtWidgets.QVBoxLayout(self.rightWidget)
        self.splitter.addWidget(self.rightWidget)

        # Create a horizontal layout for data selectors and place it in the right panel
        self.dataSelectorLayout = QtWidgets.QHBoxLayout()
        self.build_data_selectors()
        self.add_spacer_to_data_selectors()

        # Add the data selectors layout to the right layout (above the image)
        self.rightLayout.addLayout(self.dataSelectorLayout)
        
        self.build_data_display()

        # Main viewer (GraphicsLayoutWidget) for the image display
        self.win = pg.GraphicsLayoutWidget()
        self.rightLayout.addWidget(self.win)

        # Make sure the viewer takes up most of the screen
        self.splitter.setStretchFactor(0, 1)  # Buttons panel stretch
        self.splitter.setStretchFactor(1, 10)  # Main viewer stretch
        self.make_viewbox()
        
        self.hist_off = True
        self.colors = [(0,0,0,255), (255,255,255,255)]

        self.labelX, self.labelY = None, None
        self.currMaskDtype = "uint8"
        self.currImDtype = "uint8"

        self.threads = []
        self.workers = []
        
        self.setEmptyDisplay(initial=True)
        self.build_status_bar()
        self.build_widgets()
        self.add_spacer_to_left()
        
        self.updateDataDisplay()
        
        self.drawType = ""
        self.cellToChange = 0

        self.annotationList = []

        self.addRegionStrokes = []
        self.currPointSet = []

        menu.menubar(self)

        self.cellDataFormat = {"Cell":[],
                                "Birth":[],
                                "Death":[]}
        self.cellLineageDataFormat = {"Mother":[],
                                    "Daughters": []}
        
        self.plotTypes = ["single", "population", "heatmap"]
        self.populationPlotTypes = ["population", "heatmap"]
            
        self.lineageWindow = None
        self.plotPropsBeenChecked = False
        self.toPlot = None
        self.emptying = False

        self.tsPlotWindow = None 
        self.sfPlotWindow = None
        self.tsPlotWindowOn = False
        self.sfPlotWindowOn = False

        self.overrideNpyPath = None

        self.trainModelName = None
        self.trainModelSuffix = None

        self.experiments = []
        self.experiment_index = -1

        self.newInterpolation = False

        self.numObjs = 0
        self.deleting = False
        self.resetting_data = False

        self.win.show()

        if len(self.modelNames)==0:
            self.showError("No Models Loaded. Go to Models>Load Model Weights")

        if dir is not None:
            if dir_num_channels is None:
                dir_num_channels = 1
            logger.info(f"Loading Experiment Directory from {dir} with {dir_num_channels} channels")
            self.loadExperiment(dir, num_channels=dir_num_channels, ims_only=dir_ims_only)

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
    
    def return_focus_to_main_window(self):
        # Set focus back to the main window's central widget
        return
        #self.centralWidget().setFocus()
    
    def get_channel_obj_from_name(self, channel_name):
        for channel in self.experiment().channels:
            if channel.name == channel_name:
                return channel
        raise ValueError(f"{channel_name} does not exist in channelSelect")
    
    def get_label_obj_from_name(self, label_name):
        for label in self.experiment().labels:
            if label.name == label_name:
                return label
        raise ValueError(f"{label_name} does not exist in channelSelect")
    
    def flow_on(self):
        return self.imageTypeSelect.currentIndex() == self.FLOW_ID
    
    def prob_on(self):
        return self.imageTypeSelect.currentIndex() == self.PROB_ID

    def prob_or_flow_on(self):
        return self.flow_on() or self.prob_on()
    
    def get_models_loaded_status(self):
        models_loaded = {model: getModelLoadedStatus(model) for model in getBuiltInModelTypes()}
        for custom_model in self.find_custom_models():
            models_loaded[custom_model] = True
        return models_loaded
    
    def find_custom_models(self):
        custom_models = []
        for model in self.modelNames:
            for default_model in getBuiltInModelTypes():
                if default_model in model and default_model != model:
                    custom_models.append(model)
        return custom_models

    def get_weights_loaded_status(self):
        models_loaded = self.get_models_loaded_status()
        models_loaded[self.INTERP_MODEL_NAME] = rife_weights_loaded()
        return models_loaded

    def handleFinished(self, error = False):
        self.closeThread(self.threads[-1])
        self.updateThreadDisplay()
    
    def clearCurrMask(self):
        self.currMask[self.currMask>0] = 0
        contours  = self.getCurrContours()
        contours[:,:] = 0 
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
            self.enableImageOperations()
        
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
        self.colors = [(0,0,0,255), (255, 255,255,255)]
        
        self.pg_im.clear()
        self.pg_im.setLookupTable(None) 
        
        self.histogram.setLevels(self.saturation[0], self.saturation[1])
            #self.setHistGradient(self.saturation)
        #self.saturationSlider.setMaximum(self.imData.channels[self.imZ][:,:,:].max())
        #self.saturationSlider.resetLevels(self.saturation)
        self.pg_im.setImage(self.currIm, autoLevels=False, levels = [0,255])
        
        self.setHistGradient([0,1])
        
        #self.setHistGradient()

        if not initial:
            self.channelSelect.clear()
            self.brushTypeSelect.setEnabled(False)
            self.brushTypeSelect.setCurrentIndex(-1)
            self.disableImageOperations()

    def setEmptyMasks(self, initial = False):
        self.firstMaskLoad = True
        self.maskLoaded = False
        self.selectedCells = []
        self.selectedCellContourColors = []
        self.cmap = self.getMaskCmap()
        
        self.probCmap = self.getProbCmap()
        #self.pg_im.setLookupTable(self.convertMatplotlibCmapToPg(self.probCmap))
        
        self.flowCmap = self.getFlowCmap()
        self.maskColors = self.cmap.copy()
        self.maskChanged = False
        self.prevMaskOn = True
        self.maskOn = True
        self.contourOn = False
        self.maskZ = -1
        self.currMask = np.zeros((512,512), dtype = np.uint8)
        self.pg_mask.setImage(self.currMask, autolevels  =False, levels =[0,0], lut = self.maskColors)

        if not initial:
            self.labelSelect.clear()
            self.maskOnCheck.setChecked(False)
            self.imageTypeSelect.setCurrentIndex(0)
            self.contourButton.setChecked(False)
            self.cellNumButton.setChecked(False)
            self.disableMaskOperations()

    def getCurrTimeStr(self):
        now = datetime.now()
        return now.strftime("%d-%m-%Y|%H-%M")
    
    def newData(self):
        self.computeMaxT()
        self.updateDisplay()
        #self.saveData()
        
    def addBlankMasks(self):
        if not self.imLoaded:
            self.showError("Load Images First")
            return
        blankMasks = np.zeros(self.getCurrImShapeExcludeRGB(), dtype = np.uint16)
        self.loadMasks(blankMasks, name = "blank")

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

        self.win.addItem(self.view, row = 0, col = 0, rowspan = 20, colspan = 17)
        self.view.setMenuEnabled(False)
        self.view.setMouseEnabled(x=True, y=True)

        self.pg_im = pg.ImageItem()
        self.pg_mask = ImageDraw(viewbox = self.view, parent = self)

        self.view.addItem(self.pg_im)
        self.view.addItem(self.pg_mask)
        self.pg_mask.setZValue(10)
        
        self.histogram = pg.HistogramLUTItem(gradientPosition="left")
        self.histogram.setImageItem(self.pg_im)
        self.win.addItem(self.histogram, row=0, col=17, rowspan=20, colspan=3)
        
    def getCellColors(self,im):
        return np.take(self.maskColors, np.unique(im), 0)

    def build_status_bar(self):
        self.statusBar = QStatusBar()
        self.statusBar.setFont(self.medfont)
        self.statusBar.setStyleSheet(self.statusbarstyle)
        self.setStatusBar(self.statusBar)

        self.gpuDisplayTorch = ReadOnlyCheckBox(" gpu |  ")
        self.gpuDisplayTorch.setFont(self.smallfont)
        self.gpuDisplayTorch.setStyleSheet(self.checkstyle)
        self.gpuDisplayTorch.setChecked(False)
        self.checkGPUs()
        if self.torch:
            self.gpuDisplayTorch.setChecked(True)
        self.statusBarLayout = QGridLayout()
        self.statusBarWidget = QWidget()
        self.statusBarWidget.setLayout(self.statusBarLayout)

        self.cpuCoreDisplay = QLabel("")
        self.cpuCoreDisplay.setFont(self.smallfont)
        self.cpuCoreDisplay.setStyleSheet(self.labelstyle)
        self.updateThreadDisplay()

        self.hasLineageBox = ReadOnlyCheckBox("lineage data  |  ")
        self.hasCellDataBox = ReadOnlyCheckBox("cell data")
        for display in [self.hasLineageBox,self.hasCellDataBox]:
            display.setFont(self.smallfont)
            display.setStyleSheet(self.checkstyle)
            display.setChecked(False)
        
        self.progressBar = QProgressBar(self)
        self.progressBar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }

            QProgressBar::chunk {
                background-color: red;
                width: 10px; /* Used to create stripe effect if needed */
            }
        """)
        self.progressBar.setVisible(False)

        self.statusBarLayout.addWidget(self.cpuCoreDisplay, 0, 1, 1,1, alignment=(QtCore.Qt.AlignCenter))
        #self.statusBarLayout.addWidget(self.gpuDisplayTF, 0, 2, 1, 1)
        self.statusBarLayout.addWidget(self.gpuDisplayTorch, 0, 3, 1, 1)
        self.statusBarLayout.addWidget(self.hasLineageBox, 0,4,1,1)
        self.statusBarLayout.addWidget(self.hasCellDataBox,0,5,1,1)
        self.statusBarLayout.addWidget(self.progressBar, 0,6,1,1)
        self.statusBar.addWidget(self.statusBarWidget)
        
    def build_data_selectors(self):
        self.experimentLabel = QLabel("Experiment:")
        self.experimentLabel.setStyleSheet(self.labelstyle)
        self.experimentLabel.setFont(self.smallfont)
        self.dataSelectorLayout.addWidget(self.experimentLabel)

        self.experimentSelect= QComboBox()
        self.experimentSelect.setStyleSheet(self.dropdowns)
        #self.experimentSelect.setFocusPolicy(QtCore.Qt.NoFocus)
        self.experimentSelect.setFont(self.medfont)
        self.experimentSelect.currentIndexChanged.connect(self.experimentChange)
        self.dataSelectorLayout.addWidget(self.experimentSelect)
        
        self.channelSelectLabel = QLabel("Channel: ")
        self.channelSelectLabel.setStyleSheet(self.labelstyle)
        self.channelSelectLabel.setFont(self.smallfont)
        self.dataSelectorLayout.addWidget(self.channelSelectLabel)
        
        self.channelSelect = CustomComboBox(lambda x: self.deleteData(x, channel = True), parent = self, channel = True)
        self.channelSelect.setStyleSheet(self.dropdowns)
        self.channelSelect.setFont(self.medfont)
        self.channelSelect.currentIndexChanged.connect(self.channelSelectIndexChange)
        self.channelSelect.setEditable(True)
        self.channelSelect.editTextChanged.connect(self.channelSelectEdit)
        self.channelSelect.setEnabled(False)
        self.dataSelectorLayout.addWidget(self.channelSelect)
        
        self.dataSelectorLayout.setAlignment(self.channelSelect, QtCore.Qt.AlignLeft)
        self.channelSelect.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.channelSelect.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.channelSelect.setMinimumWidth(200)
        self.channelSelect.setFixedHeight(20)
        #self.channelSelect.setFocusPolicy(QtCore.Qt.NoFocus)
        setattr(self.channelSelect, "items", lambda: [self.channelSelect.itemText(i) for i in range(self.channelSelect.count())])

        self.labelSelectLabel = QLabel("Label: ")
        self.labelSelectLabel.setStyleSheet(self.labelstyle)
        self.labelSelectLabel.setFont(self.smallfont)
        self.dataSelectorLayout.addWidget(self.labelSelectLabel)
        
        self.labelSelect = CustomComboBox(lambda x: self.deleteData(x, label = True), parent = self)
        self.labelSelect.setStyleSheet(self.dropdowns)
        self.labelSelect.setFont(self.medfont)
        self.labelSelect.currentIndexChanged.connect(self.labelSelectIndexChange)
        self.labelSelect.setEditable(True)
        self.labelSelect.editTextChanged.connect(self.labelSelectEdit)
        self.labelSelect.setEnabled(False)
        self.dataSelectorLayout.addWidget(self.labelSelect)
        
        self.dataSelectorLayout.setAlignment(self.labelSelect, QtCore.Qt.AlignLeft)
        setattr(self.labelSelect, "items", lambda: [self.labelSelect.itemText(i) for i in range(self.labelSelect.count())])
        self.labelSelect.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.labelSelect.setMinimumWidth(200)
        self.labelSelect.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        #self.labelSelect.setFocusPolicy(QtCore.Qt.NoFocus)

        self.imageTypeLabel = QLabel("image type:")
        self.imageTypeLabel.setStyleSheet(self.labelstyle)
        self.imageTypeLabel.setFont(self.smallfont)
        self.dataSelectorLayout.addWidget(self.imageTypeLabel)
        
        self.imageTypeSelect = QComboBox()
        self.imageTypeSelect.setStyleSheet(self.dropdowns)
        self.imageTypeSelect.setFont(self.medfont)
        self.imageTypeSelect.setFocusPolicy(QtCore.Qt.NoFocus)
        self.imageTypeSelect.addItems(["image[i]", "probability[p]", "flowsXY[f]"])
        self.imageTypeSelect.setCurrentIndex(0)
        self.imageTypeSelect.setEnabled(True)
        self.imageTypeSelect.setMaximumWidth(100)
        self.dataSelectorLayout.setAlignment(self.imageTypeSelect, QtCore.Qt.AlignLeft)
        self.imageTypeSelect.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.dataSelectorLayout.addWidget(self.imageTypeSelect)
        
        self.imageTypeSelect.setEnabled(True)
        self.set_prob_and_flows_dropdown(False)
        self.imageTypeSelect.currentIndexChanged.connect(self.imageTypeChange)
        
                # Assign shortcuts to specific indices
        self.assignShortcutToIndex("i", 0)  # Ctrl+1 will set the combobox to index 0
        self.assignShortcutToIndex("p", 1)  # Ctrl+2 will set the combobox to index 1
        self.assignShortcutToIndex("f", 2)  # Ctrl+3 will set the combobox to index 2

    def add_spacer_to_data_selectors(self):
        # Add a horizontal stretchable spacer to keep data selectors compact
        spacer = QtWidgets.QSpacerItem(250, 40, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.dataSelectorLayout.addItem(spacer)
    
    def build_data_display(self):
        self.dataDisplay = QLabel("")
        self.dataDisplay.setMinimumWidth(300)
        # self.dataDisplay.setMaximumWidth(300)
        self.dataDisplay.setStyleSheet(self.labelstyle)
        self.dataDisplay.setFont(self.medfont)
        self.dataDisplay.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.rightLayout.addWidget(self.dataDisplay)
    
    def add_horizontal_line(self, row):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setStyleSheet('color: white')
        self.leftLayout.addWidget(line, row, 0, 1, 2)

    def add_labeled_widget(self, label, widget, row):
        self.leftLayout.addWidget(label, row, 0)

        self.leftLayout.addWidget(widget, row, 1)

    def add_two_buttons(self, button1, button2, row):
        self.leftLayout.addWidget(button1, row, 0)
        self.leftLayout.addWidget(button2, row, 1)

    def add_spacer_to_left(self):
        # Add a stretchable spacer at the bottom to prevent widgets from spreading out
        spacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.leftLayout.addItem(spacer)
    
    def build_widgets(self):
        row = 1
        
        fixed_spacer = QtWidgets.QSpacerItem(20, 50, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.leftLayout.addItem(fixed_spacer)
        
        label = QLabel('Drawing:')#[\u2191 \u2193]')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.leftLayout.addWidget(label, row, 0, 1, 2)
        row+=1

        label = QLabel("Brush Type:")
        label.setStyleSheet(self.labelstyle)
        label.setFont(self.medfont)
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
        self.add_labeled_widget(label, self.brushTypeSelect, row)
        row+=1

        label = QLabel("Brush Size")
        label.setStyleSheet(self.labelstyle)
        label.setFont(self.medfont)
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
        self.add_labeled_widget(label, self.brushSelect, row)
        row+=1
        
        self.add_horizontal_line(row)
        row+=1

        label = QLabel('Tracking')#[\u2191 \u2193]')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.leftLayout.addWidget(label, row, 0, 1, 2)
        row += 1

        self.fiestButton = QPushButton('track with FIEST')
        self.fiestButton.setStyleSheet(self.styleInactive)
        self.fiestButton.setFont(self.medfont)
        self.fiestButton.clicked.connect(self.fiestButtonClick)
        self.fiestButton.setEnabled(False)
        self.fiestButton.setToolTip("Frame Interpolation Enhanced Tracking")
        
        self.trackButton = QPushButton('basic track')
        #self.trackButton.setFixedWidth(90)
        #self.trackButton.setFixedHeight(20)
        self.trackButton.setStyleSheet(self.styleInactive)
        self.trackButton.setFont(self.medfont)
        self.trackButton.clicked.connect(self.trackButtonClick)
        self.trackButton.setEnabled(False)
        self.trackButton.setToolTip("Track current cell labels")
        
        self.add_two_buttons(self.fiestButton, self.trackButton, row)
        row+=1
        
        self.lineageButton = QPushButton("construct lineages")
        self.lineageButton.setStyleSheet(self.styleInactive)
        #self.lineageButton.setFixedWidth(90)
        #self.lineageButton.setFixedHeight(18)
        self.lineageButton.setFont(self.medfont)
        self.lineageButton.setToolTip("Use current budNET mask to assign lineages to a cellular label")
        self.lineageButton.setEnabled(False)
        self.showMotherDaughters = False
        self.showLineages = False
        self.lineageButton.clicked.connect(self.getLineages)

        self.trackObjButton = QPushButton('track label to cell')
        #self.trackObjButton.setFixedWidth(90)
        #self.trackObjButton.setFixedHeight(20)
        self.trackObjButton.setFont(self.medfont)
        self.trackObjButton.setStyleSheet(self.styleInactive)
        self.trackObjButton.clicked.connect(self.trackObjButtonClick)
        self.trackObjButton.setEnabled(False)
        self.trackObjButton.setToolTip("Track current non-cytoplasmic label to a cellular label")
        
        self.add_two_buttons(self.lineageButton, self.trackObjButton, row)
        row+=1

        self.interpolateButton = QPushButton("interpolate movie")
        self.interpolateButton.setStyleSheet(self.styleInactive)
        self.interpolateButton.setFont(self.medfont)
        self.interpolateButton.setEnabled(False)
        self.interpolateButton.clicked.connect(self.interpolateButtonClicked)

        self.interpRemoveButton = QPushButton("de-interpolate")
        self.interpRemoveButton.setStyleSheet(self.styleInactive)
        self.interpRemoveButton.setFont(self.medfont)
        self.interpRemoveButton.setEnabled(False)
        self.interpRemoveButton.clicked.connect(self.interpRemoveButtonClicked)
        
        self.add_two_buttons(self.interpolateButton, self.interpRemoveButton, row)
        row+=1
        
        self.add_horizontal_line(row)
        row+=1
    
        label = QLabel('Segmentation')#[\u2191 \u2193]')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.leftLayout.addWidget(label, row, 0, 1, 2)
        row+=1
        
        self.GB = QGroupBox("Unets")
        self.GB.setStyleSheet("QGroupBox { border: 1px solid white; color:white; padding: 10px 0px;}")
        self.GBLayout = QGridLayout()
        self.GB.setLayout(self.GBLayout)
        self.GB.setToolTip("Select Unet(s) to be used for segmenting channel")
    
        self.modelChoose = QComboBox()
        self.modelChoose.addItems(sorted(self.modelNames, key = lambda x: x[0]))
        self.modelChoose.setStyleSheet(self.dropdowns)
        self.modelChoose.setFont(self.medfont)
        self.modelChoose.setFocusPolicy(QtCore.Qt.NoFocus)
        self.modelChoose.setCurrentIndex(-1)
        self.modelChoose.setFixedWidth(180)
        self.GBLayout.addWidget(self.modelChoose, 0,0,1,1)

        self.modelButton = QPushButton(u'run model')
        self.modelButton.clicked.connect(self.computeModels)
        self.GBLayout.addWidget(self.modelButton, 0,1,1,1)
        self.modelButton.setEnabled(False)
        self.modelButton.setStyleSheet(self.styleInactive)
        self.leftLayout.addWidget(self.GB, row, 0, 2, 2, Qt.AlignTop | Qt.AlignHCenter)
        row+=2

        self.add_horizontal_line(row)
        row+=1
        
        label = QLabel('Display')#[\u2191 \u2193]')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.leftLayout.addWidget(label, row, 0, 1, 2)
        row+=1

        # "Show Contours" switch
        contourLabel = QLabel("cell contours")
        contourLabel.setStyleSheet(self.labelstyle)
        contourLabel.setFont(self.medfont)
        self.contourButton = SwitchControl()
        self.contourButton.setStyleSheet(self.checkstyle)
        self.contourButton.stateChanged.connect(self.toggleContours)
        self.contourButton.setEnabled(False)
        self.add_labeled_widget(contourLabel, self.contourButton, row)
        row+=1

        # # "Show Plot Window" switch
        # plotLabel = QLabel("plot window")
        # plotLabel.setStyleSheet(self.labelstyle)
        # plotLabel.setFont(self.medfont)
        # self.l.addWidget(plotLabel, rowspace + 1, 18, 1, 1)
        # self.plotButton = SwitchControl()
        # self.plotButton.setStyleSheet(self.checkstyle)
        # self.plotButton.stateChanged.connect(self.togglePlotWindow)
        # self.l.addWidget(self.plotButton, rowspace + 1, 19, 1, 1)

        # "Mask" switch
        maskLabel = QLabel("masks")
        maskLabel.setStyleSheet(self.labelstyle)
        maskLabel.setFont(self.medfont)
        self.maskOnCheck = SwitchControl()
        self.maskOnCheck.setStyleSheet(self.checkstyle)
        self.maskOnCheck.setEnabled(False)
        self.maskOnCheck.stateChanged.connect(self.toggleMask)
        self.add_labeled_widget(maskLabel, self.maskOnCheck, row)
        row+=1

        # # "Probability" switch
        # probLabel = QLabel("pixel probability")
        # probLabel.setStyleSheet(self.labelstyle)
        # probLabel.setFont(self.medfont)
        # self.l.addWidget(probLabel, rowspace + 1, 18, 1, 1)
        # self.probOnCheck = SwitchControl()
        # self.probOnCheck.setStyleSheet(self.checkstyle)
        # self.probOnCheck.setEnabled(False)
        # self.probOnCheck.stateChanged.connect(self.toggleProb)
        # self.l.addWidget(self.probOnCheck, rowspace + 1, 19, 1, 1)

        # "mother-daughters" switch
        motherDaughtersLabel = QLabel("mother-daughters")
        motherDaughtersLabel.setStyleSheet(self.labelstyle)
        motherDaughtersLabel.setFont(self.medfont)
        self.showMotherDaughtersButton = SwitchControl()
        self.showMotherDaughtersButton.setStyleSheet(self.checkstyle)
        self.showMotherDaughtersButton.setEnabled(False)
        self.showMotherDaughtersButton.stateChanged.connect(self.toggleMotherDaughters)
        self.add_labeled_widget(motherDaughtersLabel, self.showMotherDaughtersButton, row)
        row+=1

        # "cell nums" switch
        cellNumLabel = QLabel("cell numbers")
        cellNumLabel.setStyleSheet(self.labelstyle)
        cellNumLabel.setFont(self.medfont)
        self.cellNumButton = SwitchControl()
        self.cellNumButton.setStyleSheet(self.checkstyle)
        self.cellNumButton.setEnabled(False)
        self.cellNumButton.stateChanged.connect(self.toggleCellNums)
        self.add_labeled_widget(cellNumLabel, self.cellNumButton, row)
        row+=1

        # "lineages" switch
        lineageLabel = QLabel("lineages")
        lineageLabel.setStyleSheet(self.labelstyle)
        lineageLabel.setFont(self.medfont)
        self.showLineageButton = SwitchControl()
        self.showLineageButton.setStyleSheet(self.checkstyle)
        self.showLineageButton.setEnabled(False)
        self.showLineageButton.stateChanged.connect(self.toggleLineages)
        self.add_labeled_widget(lineageLabel, self.showLineageButton, row)
        row+=1

        # "lineage tree" switch
        treeLabel = QLabel("lineage tree")
        treeLabel.setStyleSheet(self.labelstyle)
        treeLabel.setFont(self.medfont)
        self.showTreeButton = SwitchControl()
        self.showTreeButton.setStyleSheet(self.checkstyle)
        self.showTreeButton.setEnabled(False)
        self.showTreeButton.stateChanged.connect(self.toggleLineageTreeWindow)
        self.add_labeled_widget(treeLabel, self.showTreeButton, row)

        # for i in range(self.mainViewRows):
        #     self.l.setRowStretch(i, 10)

        # self.l.setColumnStretch(20,2)
        # self.l.setColumnStretch(0,2)
        # self.l.setContentsMargins(0,0,0,0)
        # self.l.setSpacing(0)


        # self.contourButton = QCheckBox("Show Contours")
        # self.contourButton.setStyleSheet(self.checkstyle)
        # self.contourButton.setFont(self.medfont)
        # self.contourButton.stateChanged.connect(self.toggleContours)
        self.contourButton.setShortcut(QtCore.Qt.Key_C)
        # self.contourButton.setEnabled(False)
        # self.l.addWidget(self.contourButton, rowspace+1, 16,1,2)

        # self.plotButton = QCheckBox("Show Plot Window")
        # self.plotButton.setStyleSheet(self.checkstyle)
        # self.plotButton.setFont(self.medfont)
        # self.plotButton.stateChanged.connect(self.togglePlotWindow)
        #self.plotButton.setShortcut(QtCore.Qt.Key_P)
        # self.l.addWidget(self.plotButton, rowspace+1, 18, 1,2)

        # self.maskOnCheck = QCheckBox("Mask")
        # self.maskOnCheck.setStyleSheet(self.checkstyle)
        # self.maskOnCheck.setFont(self.medfont)
        # self.maskOnCheck.setEnabled(False)
        self.maskOnCheck.setShortcut(QtCore.Qt.Key_Space)
        # self.maskOnCheck.stateChanged.connect(self.toggleMask)
        # self.l.addWidget(self.maskOnCheck, rowspace+2, 16,1,2)

        # self.probOnCheck = QCheckBox("Probability")
        # self.probOnCheck.setStyleSheet(self.checkstyle)
        # self.probOnCheck.setFont(self.medfont)
        # self.probOnCheck.setEnabled(False)
        #self.probOnCheck.setShortcut(QtCore.Qt.Key_F)
        # self.probOnCheck.stateChanged.connect(self.toggleProb)
        # self.l.addWidget(self.probOnCheck, rowspace+2, 18,1,2)

        # self.cellNumButton = QCheckBox("cell nums")
        # self.cellNumButton.setStyleSheet(self.checkstyle)
        # self.cellNumButton.setFont(self.medfont)
        # self.cellNumButton.stateChanged.connect(self.toggleCellNums)
        # self.cellNumButton.setEnabled(False)
        # self.l.addWidget(self.cellNumButton, rowspace+3, 16, 1,2)

        # self.showLineageButton = QCheckBox("lineages")
        # self.showLineageButton.setStyleSheet(self.checkstyle)
        # self.showLineageButton.setFont(self.medfont)
        # self.showLineageButton.stateChanged.connect(self.toggleLineages)
        # self.showLineageButton.setEnabled(False)
        # self.l.addWidget(self.showLineageButton, rowspace+3, 18, 1,2)

        # self.showMotherDaughtersButton = QCheckBox("mother-daughters")
        # self.showMotherDaughtersButton.setStyleSheet(self.checkstyle)
        # self.showMotherDaughtersButton.setFont(self.medfont)
        # self.showMotherDaughtersButton.stateChanged.connect(self.toggleMotherDaughters)
        # self.showMotherDaughtersButton.setEnabled(False)
        # self.l.addWidget(self.showMotherDaughtersButton, rowspace+4, 16, 1,2)

        # self.showTreeButton = QCheckBox("lineage tree")
        # self.showTreeButton.setStyleSheet(self.checkstyle)
        # self.showTreeButton.setFont(self.medfont)
        # self.showTreeButton.stateChanged.connect(self.toggleLineageTreeWindow)
        # self.showTreeButton.setEnabled(False)
        # self.l.addWidget(self.showTreeButton, rowspace+4, 18, 1,2)

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
        # for i in range(self.mainViewRows):
        #     self.l.setRowStretch(i, 10)

        # self.l.setColumnStretch(20,2)
        # self.l.setColumnStretch(0,2)
        # self.l.setContentsMargins(0,0,0,0)
        # self.l.setSpacing(0)

    def set_prob_and_flows_dropdown(self, b):
        self.imageTypeSelect.model().item(1).setEnabled(b)
        self.imageTypeSelect.model().item(2).setEnabled(b)
        
        if not b:
            self.imageTypeSelect.setCurrentIndex(0)
    
    def updateThreadDisplay(self):
        threadCount = len(self.threads)+1
        self.cpuCoreDisplay.setText(f"{threadCount}/{self.idealThreadCount} threads |")

    def enableMaskOperations(self):
        self.brushSelect.setEnabled(True)
        self.brushTypeSelect.setEnabled(True)
        self.contourButton.setEnabled(True)
        self.labelSelect.setEnabled(True)
        self.trackButton.setEnabled(True)
        self.trackObjButton.setEnabled(True)
        self.trackButton.setStyleSheet(self.styleUnpressed)
        self.trackObjButton.setStyleSheet(self.styleUnpressed)
        self.cellNumButton.setEnabled(True)
        self.maskOnCheck.setEnabled(True)
        self.maskOnCheck.setChecked(True)
        self.set_prob_and_flows_dropdown(self.label().has_probability and self.label().has_flows)
        self.lineageButton.setEnabled(True)
        self.lineageButton.setStyleSheet(self.styleUnpressed)
    
        
    def assignShortcutToIndex(self, key_sequence, index):
        """Assign a single key shortcut to select a specific index in the QComboBox."""
        shortcut = QShortcut(QKeySequence(key_sequence), self)
        shortcut.activated.connect(lambda: self.setImTypeSelectIndexIfEnabled(index))
        
    def setImTypeSelectIndexIfEnabled(self, index):
        """Set the combobox index only if the item is enabled."""
        if self.imageTypeSelect.model().item(index).isEnabled():
            self.imageTypeSelect.setCurrentIndex(index)
        else:
            print(f"Index {index} is disabled and cannot be selected.")

    def disableMaskOperations(self):
        self.brushSelect.setEnabled(False)
        self.brushTypeSelect.setEnabled(False)
        self.contourButton.setEnabled(False)
        self.labelSelect.setEnabled(False)
        self.trackButton.setEnabled(False)
        self.trackObjButton.setEnabled(False)
        self.trackButton.setStyleSheet(self.styleInactive)
        self.trackObjButton.setStyleSheet(self.styleInactive)
        self.cellNumButton.setEnabled(False)
        self.maskOnCheck.setEnabled(False)
        self.maskOnCheck.setChecked(False)
        self.set_prob_and_flows_dropdown(False)
        self.lineageButton.setEnabled(False)
        self.lineageButton.setStyleSheet(self.styleInactive)

    def enableImageOperations(self):
        #self.saturationSlider.setEnabled(True)
        #self.autoSaturationButton.setEnabled(True)
        #self.autoSaturationButton.setStyleSheet(self.styleUnpressed)
        self.modelButton.setEnabled(True)
        self.modelButton.setStyleSheet(self.styleUnpressed)
        # self.artiButton.setEnabled(True)
        # self.artiButton.setStyleSheet(self.styleUnpressed)
        self.channelSelect.setEnabled(True)
        self.imageTypeSelect.setEnabled(True)
        self.checkInterpolation()
        
        self.fiestButton.setEnabled(True)
        self.fiestButton.setStyleSheet(self.styleUnpressed)
    
    def disableImageOperations(self):
        self.brushSelect.setEnabled(False)
        self.brushTypeSelect.setEnabled(False)
        #self.saturationSlider.setEnabled(True)
        #self.autoSaturationButton.setEnabled(True)
        #self.autoSaturationButton.setStyleSheet(self.styleUnpressed)
        self.modelButton.setEnabled(False)
        self.modelButton.setStyleSheet(self.styleInactive)
        self.imageTypeSelect.setEnabled(False)
        self.imageTypeSelect.setCurrentIndex(0)
        # self.artiButton.setEnabled(True)
        # self.artiButton.setStyleSheet(self.styleInactive)
        self.channelSelect.setEnabled(False)
        self.fiestButton.setEnabled(False)
        self.fiestButton.setStyleSheet(self.styleInactive)

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
    
    def deleteData(self, idx, channel = False, label = False):
        self.deleting = True
        val = None
        if channel:
            val =  self.channelDelete(idx)
        elif label:
            val = self.labelDelete(idx)
        self.deleting = False
        return val
    
    def onDelete(self, index, new_index, channel = False):
        if channel:
            self.experiment().delete_channel(index)
            if new_index == -1:
                self.setEmptyDisplay(initial = False)
            else:
                self.imZ = new_index
                self.setDataSelects()
                self.imChanged = True
                self.updateDisplay()
        
        else:
            self.experiment().delete_label(index)
            self.setDataSelects()
            if new_index == -1:  
                self.setEmptyMasks()
                self.disableMaskOperations()
            
            self.maskZ = new_index
            self.maskChanged = True
            self.updateDisplay()
        
    def blockComboSignals(self, b):
        self.channelSelect.blockSignals(b)
        self.labelSelect.blockSignals(b)

    def channelDelete(self, index):
        num_channels = self.experiment().num_channels
        channel_to_remove = self.experiment().channels[index]
        message = f'''Are you sure you want to remove channel {channel_to_remove.name}'''
        if num_channels == 1:
            if self.experiment().has_labels():
                message = f'''Removing the last set of images will result 
                            in an empty display with all masks removed as 
                            well. Are you sure you want to remove channel 
                            {channel_to_remove.name}'''
        
        return self.doYesOrNoDialog("Confirmation", message)
        
    def labelDelete(self, index):
        label_to_remove = self.experiment().labels[index]
        message = f'''Are you sure you want to remove label {label_to_remove.name}'''
        return self.doYesOrNoDialog("Confirmation", message)
 
    def doYesOrNoDialog(self, title, message):
        reply = QMessageBox.question(self, "Confirmation", message, QMessageBox.Yes | QMessageBox.No)
        return reply == QMessageBox.Yes

    def channelSelectEdit(self, text):
        if self.resetting_data:
            idx = self.channelSelect.currentIndex()
            self.channelSelect.setItemText(idx, str(text))
            return
        if not self.deleting:
            curr_text = self.channelSelect.currentText()
            idx = self.channelSelect.currentIndex()
            if not self.emptying:
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

        self.return_focus_to_main_window()
        
    def channelSelectRemote(self):
        index = self.imZ
        self.channelSelect.setCurrentIndex(index)

    def labelSelectEdit(self, text):
        if self.emptying or self.deleting:
            return
        idx = self.labelSelect.currentIndex()
        text = str(text)
        if self.resetting_data or self.experiment().new_label_name(idx, text):
            self.labelSelect.setItemText(idx, str(text))

    def labelSelectIndexChange(self,index):
        if not self.emptying:
            if self.maskZ != index:
                self.maskZ = index
                self.maskChanged = True
                self.updateDisplay()
                self.imageTypeSelect.setCurrentIndex(0)
                self.checkProbability()
        self.return_focus_to_main_window()
    
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
        dlg  = GeneralParamDialog({}, [], f"tracking {labelsToTrack}", self, labelSelects=["Cytoplasm Label"])

        if dlg.exec():
            self.deactivateButton(self.trackObjButton)
            data = dlg.getData()


            cellIdx = self.labelSelect.findText(data["Cytoplasm Label"])
            cells = self.experiment().get_label("labels", idx = cellIdx)
            obj = self.getCurrMaskSet()

            task = track_to_cell
            worker = TrackWorker(task, cells, self.maskZ, self.experiment_index, obj)            
            self.runLongTask(worker,self.trackFinished, self.trackObjButton)

        else:
            return
    
    def checkProbability(self):
        self.set_prob_and_flows_dropdown(self.label().has_probability and self.label().has_flows)

    def trackFinished(self, z, exp_idx, tracked):
        if isinstance(tracked, np.ndarray):
            contours = self.getCurrContourSet()
            newContours = tracked.copy()
            newContours[np.logical_not(contours>0)] = 0
            self.experiments[exp_idx].labels[z].set_data(tracked, newContours)
            curr_label = self.labelSelect.currentText()
            try:
                self.updateCellData(idx = z, exp_idx = exp_idx)
            except IndexError:
                self.showError(f'''Unable to produce data for {curr_label}, 
                               likely due to poor label quality or difficulty producing accurate tracks.
                                Correct tracks and/or label accuracy and try again''')
            self.experiment_index = exp_idx
            self.maskZ = z
            self.drawMask()
            self.showTimedPopup(f"{curr_label} has been tracked")
            
    def fiestButtonClick(self):
        choices = ['full_lifecycle (mating/tetrads)', 'asexual_only']
        dlg = ChoiceDialog(choices, "Full lifecycle movie?", parent=self)
        if dlg.exec():
            choice = dlg.get_choice()
            if choice:
                if choice == choices[1]:
                    self.fiestProliferating()
                elif choice == choices[0]:
                    self.fiestFullLifecycle()
                     
    def fiestFullLifecycle(self):
        SEG_MODEL_TYPE = "proSeg"
        MATING_MODEL_TYPE = "matSeg"
        SPORE_MODEL_TYPE = "spoSeg"
        
        seg_model_options = getModelsByType(SEG_MODEL_TYPE)
        mat_model_options = getModelsByType(MATING_MODEL_TYPE)
        spore_model_options = getModelsByType(SPORE_MODEL_TYPE)

        if not rife_weights_loaded():
            self.showError("RIFE model weights are not loaded. Go to models>load model weights in the menu bar")
            return
        if not seg_model_options:
            self.showError(f"No {SEG_MODEL_TYPE} weights detected. Go to models>load model weights in the menu bar")
            return
        if not mat_model_options:
            self.showError(f"No {MATING_MODEL_TYPE} weights detected. Go to models>load model weights in the menu bar")
            return
        if not spore_model_options:
            self.showError(f"No {SPORE_MODEL_TYPE} weights detected. Go to models>load model weights in the menu bar")
            return
        
        exp = self.experiment()
        valid_channels = exp.get_timeseries()
        if not valid_channels:
            self.showError("FIEST requires multi-frame channels. No timeseries detected in {exp.name}")
            return
        wizard = FiestFullLifeCycleWizard(self, valid_channels, seg_model_options, mat_model_options, 
                                          spore_model_options)
        if wizard.exec():
            self.startFiestFLWorker(wizard.getData())
    
    def startFiestFLWorker(self, fiest_data):
        channel = self.experiment().channels[fiest_data["channelIndex"]]
        ims = channel.ims
        intervals = fiest_data["intervals"]
        
        proSegWeightName = fiest_data["proSeg"]["modelWeight"]
        proSegWeightPath = join(MODEL_DIR, "proSeg", proSegWeightName)
        
        spoSegWeightName = fiest_data["spoSeg"]["modelWeight"]
        spoSegWeightPath = join(MODEL_DIR, "spoSeg", spoSegWeightName)
        spore_intervals = [fiest_data["spoSeg"]["t_start"], fiest_data["spoSeg"]["t_stop"]]
        
        matSegWeightName = fiest_data["matSeg"]["modelWeight"]
        matSegWeightPath = join(MODEL_DIR, "matSeg", matSegWeightName)
        mat_intervals = [fiest_data["matSeg"]["t_start"], fiest_data["matSeg"]["t_stop"]]
        
        def worker_func(): return fiest_full_lifecycle(ims,intervals, fiest_data["proSeg"], proSegWeightPath, 
                                                       fiest_data["matSeg"], matSegWeightPath, fiest_data["spoSeg"], 
                                                       spoSegWeightPath, shock_period=None, mat_start=mat_intervals[0],
                                                       mat_stop=mat_intervals[1], spo_start=spore_intervals[0], spo_stop = spore_intervals[1]
                                                       )
        worker = FiestWorker(worker_func, self.experiment_index, channel.id, False)
        self.runLongTask(worker, self.fiestFLFinished, self.fiestButton)
        
    def fiestProliferating(self):
        SEG_MODEL_TYPE = "proSeg"
        BUD_MODEL_TYPE = 'budSeg'

        seg_model_options = getModelsByType(SEG_MODEL_TYPE)
        bud_model_options = getModelsByType(BUD_MODEL_TYPE)

        if not rife_weights_loaded():
            self.showError("RIFE model weights are not loaded. Go to models>load model weights in the menu bar")
            return
        if not seg_model_options:
            self.showError(f"No {SEG_MODEL_TYPE} weights detected. Go to models>load model weights in the menu bar")
            return
        if not bud_model_options:
            self.showError(f"No {BUD_MODEL_TYPE} weights detected. Go to models>load model weights in the menu bar")
            return
        valid_channels = self.experiment().get_timeseries()
        if not valid_channels:
            self.showError("FIEST requires multi-frame channels. No timeseries detected.")
            return
        wizard = FiestWizard(self, valid_channels, seg_model_options, bud_model_options)
        if wizard.exec():
            self.startFiestWorker(wizard.getData())
    
    def startFiestWorker(self, fiest_data):
        channel = self.experiment().channels[fiest_data["channelIndex"]]
        ims = channel.ims
        intervals = fiest_data["intervals"]
        proSegWeightName = fiest_data["proSeg"]["modelWeight"]
        proSegWeightPath = join(MODEL_DIR, "proSeg", proSegWeightName)
        doLineage = fiest_data["doLineage"]
        if doLineage:
            budSegWeightName = fiest_data["budSeg"]["modelWeight"]
            budSegWeightPath = join(MODEL_DIR, "budSeg", budSegWeightName)
            def worker_func(): return fiest_basic_with_lineage(ims, intervals, fiest_data["proSeg"], proSegWeightPath,
                                     fiest_data["budSeg"], budSegWeightPath)     
        else:
            def worker_func(): return fiest_basic(ims, intervals, fiest_data["proSeg"], proSegWeightPath)
        
        worker = FiestWorker(worker_func, self.experiment_index, channel.id, doLineage)
        self.runLongTask(worker, self.fiestProlifFinished, self.fiestButton)        
    
    def fiestFLFinished(self, fiest_output, exp_idx, channel_id, lineage):
        im_name = self.getChannelNameFromId(channel_id, exp_idx=exp_idx)
        proSeg_output_name = f"{im_name}_fiest-general"
        matSeg_output_name = f"{im_name}_fiest-mating"
        spoSeg_output_name = f"{im_name}_fiest-tetrads"
        
        self.loadMasks(fiest_output["cells"], exp_idx=exp_idx, name=proSeg_output_name)
        self.loadMasks(fiest_output["spores"], exp_idx=exp_idx, name=spoSeg_output_name)
        self.loadMasks(fiest_output["mating"], exp_idx=exp_idx, name=matSeg_output_name)
        
    def fiestProlifFinished(self, fiest_output, exp_idx, channel_id, lineage):
        im_name = self.getChannelNameFromId(channel_id, exp_idx=exp_idx)
        proSeg_name = f"{im_name}_fiest"
        if lineage:
            self.loadMasks(fiest_output["cells"], exp_idx=exp_idx, name=proSeg_name)
            new_label_obj = self.experiments[exp_idx].labels[-1]
            self.experiments[exp_idx].labels[-1].celldata = LineageData(new_label_obj.id, new_label_obj.get_labels(),
                                                                        daughters=fiest_output["lineages"][0],
                                                                        mothers=fiest_output["lineages"][1])
            self.experiments[exp_idx].labels[-1].save()
            self.loadMasks(fiest_output["buds"], exp_idx=exp_idx, name=f"{im_name}_buds_fiest")
        else:
            self.loadMasks(fiest_output, exp_idx=exp_idx, name=proSeg_name)

    def saveAllChannels(self, exp_idx=None):
        if exp_idx is None:
            exp_idx = self.experiment_index
        
        for label in self.experiments[exp_idx].labels:
            label.save()
            
        
    def getChannelNameFromId(self, id, exp_idx=None):
        return self.getChannelFromId(id, exp_idx=exp_idx).name
    
    def getMaskNameFromId(self, id, exp_idx=None):
        return self.getMaskFromId(id, exp_idx=exp_idx).name
    
        
    def getChannelFromId(self, id, exp_idx=None):
        if exp_idx is None:
            exp_idx = self.experiment_index
        for channel in self.experiments[exp_idx].channels:
            if channel.id == id:
                return channel
        raise ValueError(f"No channel with id {id} exists")
    
    def getMaskFromId(self, id, exp_idx=None):
        if exp_idx is None:
            exp_idx = self.experiment_index
        for label in self.experiments[exp_idx].labels:
            if label.id == id:
                return label
        raise ValueError(f"No label with id {id} exists")
    
    def trackButtonClick(self):
        if self.label().max_t() == 0:
            self.showError("Error: More than one frame must be present to track")
            return

        idxToTrack = self.maskZ
        cells = self.getCurrMaskSet()
        task = track_general_masks
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
            print("has cell data")
            mask_obj.celldata.update_cell_data(masks, channels = viableIms, channel_names = viableImNames)
        else:
            mask_obj.celldata = TimeSeriesData(mask_obj.id, masks, channels = viableIms, channel_names=viableImNames)

        mask_obj.save()
        self.checkDataAvailibility()  

        if self.tsPlotWindow:
            self.tsPlotWindow.setData()
            self.tsPlotWindow.updatePlots()
    
    def updateCellTable(self):
        if not self.tsPlotWindow:
            return
        # self.pWindow.table.model.setData(self.getCellDataAbbrev())
        # self.pWindow.
    
    def guiIsEmpty(self):
        return not bool(self.experiments)
    

    def hasCellData(self, i = None):
        if self.guiIsEmpty():
            return False
        if not isinstance(i,int):
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
        return self.getMaskNameFromId(tsObj.mask_id)

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
            cell_id = self.experiment().labels[cellI].id
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

            self.experiment().labels[cellI].celldata = LineageData(cell_id, cells, buds = necks, cell_data=cell_data, life_data=life)

        self.experiment().labels[cellI].save()
        self.checkDataAvailibility()
        
        try:
            self.toggleLineageTreeWindow()
        except AttributeError:
            return
    
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
        self.popup = TimedPopup(new_text, time, parent=self)
        self.popup.show()

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
            image = cv2.putText(image, str(vals[i]), (xtemp[i], ytemp[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.23, (255,255,255,255), 1)
        self.pg_mask.image = image
        self.pg_mask.updateImage()
    
    def toggleLineageTreeWindow(self):
        self.lineageWindowOn = self.showTreeButton.isChecked()
        if self.lineageWindowOn:
            self.showTree()
        elif self.lineageWindow is not None:
            self.lineageWindow.close()
            self.lineageWindow = None
    
    def showTree(self, index = None):
        data = self.label().celldata.get_cell_info()
        data = data.drop(columns = ["confidence"])
        data =  data.fillna(-1)
        data = data.to_numpy()
        self.lineageWindow = plot.LineageTreeWindow(self, data, selected_cells=self.selectedCells)

    def measureDiams(self):
        if not self.maskLoaded:
            self.showError("No Masks Loaded. Draw or Segment First")
            return
        avg_maj, avg_minor = average_axis_lengths(self.currMask)
        self.showTimedPopup(f"Avg major axis length (pixels): {avg_maj}\n Avg minor axis length (pixels): {avg_minor}")
    
    def buildMeasureWindow(self):
        if self.measure_window is None:
            self.measure_window = MeasureWindow(self.currIm, self)
            self.measure_window.show()
        else:
            self.showError("Measuring Window already open")
    
    def buildTSPlotWindow(self):
        win = PlotWindowCustomizeTimeSeries(self)

        if win.exec_():
            self.toPlot = win.getData()
        else:
            self.tsPlotWindow = None
            self.tsPlotWindowOn = False
            return
        
        if self.toPlot:
            self.tsPlotWindow = plot.TimeSeriesPlotWindow(self, self.toPlot)
            self.tsPlotWindow.show()
        else:
            self.showError("Please Select Properties to Plot")
    
    def get_single_frame_plot_data(self, to_plot, do_all_frames, time):
        data = {}
        for mask_name in to_plot:
            props = to_plot[mask_name]
            if props:
                if do_all_frames:
                    data[mask_name] = []
                    mask_ims = self.experiment().get_label("labels", name = mask_name, t = None)
                    for mask_im in mask_ims:
                        data[mask_name].append(regionprops_table(mask_im, properties = props))
                else:
                    mask_im = self.experiment().get_label("labels", name = mask_name, t = self.tIndex)

                    data[mask_name] = regionprops_table(mask_im,
                                                   properties = props)
        return data



    def buildSFPlotWindow(self):
        plotOptions = ["area", "eccentricity", "axis_major_length", "axis_minor_length", "perimeter"]
        mask_names  = self.experiment().get_label_names()
        win = PlotWindowCustomizePerFrame(plotOptions, mask_names, parent = self)

        if win.exec_():
            self.toPlot = win.get_data()
            allFrames = win.do_for_all_frames()
            data = self.get_single_frame_plot_data(self.toPlot, allFrames, self.tIndex)
        else:
            self.sfPlotWindow = None
            self.sfPlotWindowOn = False
            return
        
        if data:
            self.sfPlotWindow = plot.SingleFramePlotWindow(self, self.toPlot, data, allFrames = allFrames)
            self.sfPlotWindow.show()
            self.sfPlotWindowOn = True
        else:
            self.showError("Please Select Properties to Plot")
    
    def tsPlotWinClicked(self):
        if not self.tsPlotWindowOn and self.hasCellData():
            self.buildTSPlotWindow()
        else:
            if not self.hasCellData():
                self.showError(f"No Timeseries Data Available for {self.label().name} - Switch Labels or Track/Record Cell Data First")   
            elif self.tsPlotWindowOn:
                self.showError("Time Series Plot Window Already Open. Exit First to Start a New Plot Session")
    
    def sfPlotWinClicked(self):
        hasMasks = self.experiments and self.experiment().has_labels()
        if not self.sfPlotWindowOn and hasMasks:
            self.buildSFPlotWindow()
        else:
            if not hasMasks:
                self.showError("Load Labels to Plot Frame Properties")
            elif self.sfPlotWindowOn:
                self.showError("Single Frame Plot Window Already Open. Exit First to Start a New Plot Session")

    def showInterpolationClicked(self):
        if not self.imLoaded:
            self.showError("No Images Loaded. Load an experiment and run interpolation first")
            return
        if not isinstance(self.channel(), InterpolatedChannel):
            interpolated_channel_names = self.get_interpolated_channels()
            if interpolated_channel_names:
                interpolated_channel_str = ",".join(interpolated_channel_names)
                self.showError(f"{self.channel().name} has not been interpolated. Choose one of {interpolated_channel_str}")
            else:
                self.showError("No interpolated channels are present. Run interpolation first")
        else:
            self.interpPlotWindow = plot.InterpolationHeatmapWindow(self,
                                                                    self.channel().interval_annotations, 
                                                                    self.channel().max_t(),
                                                                    self.channel().interp_intervals)
            self.interpPlotWindow.show()



    def get_interpolated_channels(self):
        return [channel.name for channel in self.experiment().channels
                if isinstance(channel, InterpolatedChannel)]

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
            if self.prob_on():
                val = self.experiment().get_label("probability", idx = self.maskZ, t = self.tIndex)[y,x]/255
            elif self.flow_on(): 
                val = self.experiment().get_label("flows", idx = self.maskZ, t = self.tIndex)[y,x]/255

            dataString += f"x={str(x)}, y={str(y)}, value={str(val)}"

        if self.imLoaded:
            if not self.prob_or_flow_on():
                dataString += f'\n{self.channel().get_files(self.tIndex)}'
                dataString += f" | IMAGE: {self.channel().get_string(self.tIndex)}, T: {self.tIndex}/{self.channel().max_t()}"
            else:
                if self.prob_on():
                    dataString += " | PROBABILITY IMAGE"
                else:
                    dataString += "| PIXEL FLOW MAGNITUDE IMAGE"
            
            
            if self.maskLoaded:
                dataString += f"  |  MASK: {self.label().get_string(self.tIndex)}, T: {self.tIndex}/{self.label().max_t()}"

            #dataString += f"  |  TIME: {self.tIndex}/{self.maxT}"

            if self.maskLoaded:
                dataString += f"  |  REGIONS: {self.numObjs}"

                if self.selectedCells:
                    sCellsString = ",".join([str(cNum) for cNum in self.selectedCells])
                    dataString += f"  |  SELECTED CELLS: {sCellsString}"
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
    
    def convertMatplotlibCmapToPg(self, cmap):
        """
        Convert a matplotlib colormap to a pyqtgraph ColorMap.
        This function generates a pyqtgraph.ColorMap using the given RGBA values.
        """
        pos = np.linspace(0, 1, len(cmap))  # Position of each color stop (between 0 and 1)
        colors = cmap[:, :3] / 255.0  # Normalize the RGB values to 0-1 range (ignore alpha)
        
        # Create and return the pyqtgraph ColorMap
        return pg.ColorMap(pos, colors)
    
    def getProbCmap(self):
        cmap = plt.get_cmap("Reds", 256)
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
    
    def getFlowCmap(self):
        cmap = plt.get_cmap("Blues", 256)
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
                self.showError(f"{get_filename(files[0])} already exists as an experiment. Change the directory name.")
                return
            else:
                self.loadExperiment(files[0])
        elif os.path.isfile(files[0]):
            file_path = files[0]
            file_name = os.path.basename(file_path)
            file_base, file_ext = os.path.splitext(file_name)

            # Check if the file is an image
            try:
                img = imread(file_path)
            except (IOError, ValueError):
                self.showError("The dropped file is not a valid image.")
                return

            # Create a directory named after the image file (without extension) in the current working directory
            current_working_dir = os.getcwd()
            target_dir = os.path.join(current_working_dir, file_base)
            
            if os.path.exists(target_dir):
                self.showError(f"A directory named {file_base} already exists in the current working directory. Please change the file name.")
                return
            else:
                os.makedirs(target_dir)
                target_file_path = os.path.join(target_dir, file_name)
                shutil.copy(file_path, target_file_path)
                self.loadExperiment(target_dir)
        else:
            self.showError("Please drop a directory containing images or a single image file.")

    
    def userSelectExperiment(self):
        dir = QFileDialog.getExistingDirectory(self, "Choose Experiment Directory")
        self.loadExperiment(dir)
    
    def loadExperiment(self,file, num_channels = None, ims_only=False):
        
        self.statusBar.showMessage(f"Loading Experiment from {file}. See terminal for progress")
        
        if file.endswith("/") or file.endswith('\\'):
            file = file[:-1]      
        if num_channels is None:                   
            dlg = GeneralParamDialog({"num_channels":1, "exclude_mask_keyword": False}, [int, bool], "number of channels/keep masks", self)
            if dlg.exec():
                data = dlg.getData()
                num_channels = int(data["num_channels"])
                ims_only = bool(data["exclude_mask_keyword"])
            else:
                return
        self.experiment_index+=1
        

        new_experiment = Experiment(file, num_channels=num_channels, ims_only=ims_only)
        # except:
        #     self.showError(f"Error laoding {file} as an experiment: directories must cannot contain only masks.")
        #     self.experiment_index-=1
        #     return
        
        self.experiments.append(new_experiment)
        self.experimentSelect.addItem(new_experiment.name)
        self.experimentSelect.setCurrentIndex(self.experiment_index)
        self.showExperimentMessage(new_experiment)
        self.checkInterpolation()
        
        self.statusBar.clearMessage()
    
    def showExperimentMessage(self, newExp):
        channels = ", ".join(newExp.get_channel_names())
        message = f"NEW EXPERIMENT: {newExp.name} | CHANNELS: {channels}"
        if newExp.has_labels():
            labels = ",".join(newExp.get_label_names())
            message+=f" | LABELS: {labels}"
        self.showTimedPopup(message, time = 60)


    def setDataSelects(self): 
        self.resetting_data = True
        self.clearDataSelects()
        self.channelSelect.addItems(self.experiment().get_channel_names())
        if self.experiment().has_labels():
            self.labelSelect.addItems(self.experiment().get_label_names())
        self.resetting_data = False 
    
    def blockDataSelect(self, b):
        self.channelSelect.blockSignals(b)
        self.labelSelect.blockSignals(b)


    def experimentChange(self):
        if not self.emptying:
            self.imLoaded = True
            self.experiment_index = self.experimentSelect.currentIndex()
            self.blockDataSelect(True)
            self.setDataSelects()
            self.blockDataSelect(False)
            self.tIndex, self.maskZ, self.imZ = 0,0,0
            self.imChanged, self.maskChanged = True, True
            self.enableImageOperations()
            if self.experiment().has_labels():
                self.maskLoaded = True
                self.enableMaskOperations()
            else:
                self.disableMaskOperations()
            self.updateDisplay()

    def clearDataSelects(self):
        self.labelSelect.clear()
        self.channelSelect.clear()

    def newIms(self, ims = None, files = None, dir = None, name = None, annotations = None):
        name = self.getNewChannelName(name)
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
            mask1, mask2, mask3 = np.expand_dims(masks[0],0), np.expand_dims(masks[1],0), np.expand_dims(masks[2],0)
            masks = np.concatenate((mask1, mask2, mask3), axis = 0)
        self.newMasks(masks, 
                      name = self.getNewLabelName(name), 
                      exp_idx=exp_idx)
    
    def isUpperHalf(self,ev):
        posY= ev.pos().y()
        topDist = abs(self.geometry().top() - posY)
        bottomDist = abs(self.geometry().bottom() - posY)

        if bottomDist <= topDist:
            return False
        else:
            return True

    def keyPressEvent(self, event):
        if (not self.maskLoaded) and (not self.imLoaded):
            return 
        
        nextMaskOn  = self.maskOn
        if (event.key() == QtCore.Qt.Key_Delete or event.key() == QtCore.Qt.Key_Backspace) and self.selectedCells:
            for selectedCell in self.selectedCells.copy():
                self.deselectCell(selectedCell)
                self.deleteCell(selectedCell)
        
        # if event.key() == QtCore.Qt.Key_Y:
        #     self.cellToChange+=1
        #     self.maskColors[self.cellToChange,:] = [255,255,255,255]
        #     self.maskChanged  = True
        
        if event.key() == QtCore.Qt.Key_B:
            self.changeBrushTypeComboBox("Brush")
        if event.key() == QtCore.Qt.Key_E:
            self.changeBrushTypeComboBox("Eraser")
        if event.key() == QtCore.Qt.Key_N:
            self.brushTypeSelect.setCurrentIndex(-1)
        if event.key() == QtCore.Qt.Key_O:
            self.changeBrushTypeComboBox("Outline")
        
        if event.key() == QtCore.Qt.Key_R:
            self.configureHistogram()
            self.drawIm()
            self.drawMask()

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
    
    def imageTypeChange(self):
        if self.prob_or_flow_on():
            self.channelSelect.setEnabled(False)
        else:
            self.channelSelect.setEnabled(True)
        self.configureHistogram()
        if not self.emptying:
            self.drawIm()
    
    def configureHistogram(self):
        if self.prob_or_flow_on():
            self.colors = [(0,0,0,200), (250,10,10,200)]
        else:
            self.colors = [(0,0,0,255), (255,255,255,255)]
    
    def resetImageView(self):
        self.configureHistogram()
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
    
    def getCellLineage(self, cellNum):
        lineage = []
        curr_gen = [cellNum]
        while True:
            next_gen = []
            for cell in curr_gen:
                daughters = self.getDaughters(cell)
                next_gen += daughters
            if next_gen:
                lineage += next_gen
                curr_gen = next_gen
            else:
                break
        return lineage
        
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
    
    def getCurrImShapeExcludeRGB(self):
        if not self.imLoaded:
            return (512,512)
        else:
            len_t = self.experiments[self.experiment_index].get_t()
            one_im_shape = self.experiments[self.experiment_index].shape()
            return (len_t, one_im_shape[0], one_im_shape[1])

    def getCurrMaskSet(self):
        return self.experiments[self.experiment_index].get_label("labels", idx = self.maskZ)
    
    def getAllMaskTypes(self): 
        masks = []
        for mask_type in ["labels", "probability", "flows"]:
            masks.append(self.experiments[self.experiment_index].get_label(mask_type, idx = self.maskZ))
        return masks
    
    def getCurrIm(self):
        if self.tIndex > self.channel().max_t():
            return np.zeros((512,512))
        if self.prob_on():
            im =  self.experiment().get_label("probability", idx = self.maskZ, t = self.tIndex).astype(np.float32)/255
            return im
        elif self.flow_on():
            im = self.experiment().get_label("flows", idx = self.maskZ, t = self.tIndex).astype(np.float32)/255
            return im
        else:
            return self.experiments[self.experiment_index].get_channel(idx = self.imZ, t = self.tIndex)
    
    def getCurrMask(self):
        if not self.experiment().has_labels() or self.tIndex > self.label().max_t():
            return np.zeros(self.experiment().shape()[:2], dtype = np.uint8)
        if self.maskOn:
            return self.experiment().get_label("labels", t = self.tIndex, idx = self.maskZ)
        else:
            return np.zeros(self.experiment().shape()[:2], dtype = np.uint8)

    def getCurrContours(self):
        return self.experiment().get_label("contours", idx = self.maskZ, t = self.tIndex)

    def getCurrContourSet(self):
        return self.experiment().get_label("contours", idx = self.maskZ)

    def getCurrFrames(self):
        return (self.getCurrIm(), self.getCurrMask())
    
    def bootUp(self):
        self.pg_im.setImage(self.currIm, autoLevels=False, levels = [0,0])
        self.pg_mask.setImage(self.currMask, autolevels  =False, levels =[0,0], lut = self.maskColors)
    
    
    def multi_label_display(self):
        if self.multi_label_window is not None:
            self.showError("A multi-label window is already open")
            return
        
        label_params = {label.name: True for label in self.experiment().labels}
        label_types = [bool for _ in range(len(label_params))]
        dlg = GeneralParamDialog(label_params, label_types, "Select Multiple Labels to View at Once", self, channelSelects=["channel"])
        
        if dlg.exec():
            data = dlg.getData()
            channel = self.get_channel_obj_from_name(data["channel"])
            labels = []
            for label_name in label_params.keys():
                if data[label_name]:
                    labels.append(self.get_label_obj_from_name(label_name))
            self.multi_label_window = MultiLabelWindow(channel.id, [label.id for label in labels], 
                                                       self.experiment_index, parent=self)
            self.multi_label_window.update()
            self.multi_label_window.show()
        
    def updateDisplay(self):
        if self.imChanged:
            self.drawIm()

        if self.maskChanged:
            self.drawMask()

        if self.tChanged:
            if self.win.underMouse() and self.labelX and self.labelY:
                x,y = self.labelX, self.labelY
                val = self.currMask[y,x]
                if not self.maskOn:
                    val = self.currIm[y,x]
            else:
                x,y,val = None,None,None

            if self.sfPlotWindowOn and self.sfPlotWindow.allFrames:
                self.sfPlotWindow.update(self.tIndex)

            self.updateDataDisplay(x=x,y=y,val=val)
            self.tChanged = False
            
            if self.multi_label_window:
                self.multi_label_window.update()
    
    def setHistGradient(self, tickLocs):
        gradient_editor = self.histogram.gradient

        # Set the gradient with specific ticks (positions and colors)
        gradient_editor.restoreState({
            'mode': 'rgb',  # You can choose other modes like 'hsv'
            'ticks': [(float(tickLoc), color) for tickLoc, color in zip(tickLocs, self.colors)]
        })
    
    def drawIm(self): 
        self.saturation = self.computeSaturation()
        self.currIm = self.getCurrIm()
        self.currImDtype = str(self.currIm.dtype)
        if self.prob_or_flow_on():
            self.pg_im.setImage(self.currIm, autoLevels = False, levels = [0,1])
            self.histogram.setLevels(0,1)
            
        else:
            self.pg_im.setImage(self.currIm, autoLevels = False)
            self.pg_im.setLevels(self.saturation)
            self.histogram.setLevels(self.saturation[0], self.saturation[1])
            #self.setHistGradient(self.saturation)
        #self.saturationSlider.setMaximum(self.imData.channels[self.imZ][:,:,:].max())
        #self.saturationSlider.resetLevels(self.saturation)
        self.setHistGradient([0,1])
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
            for num in (self.getCellLineage(cellNum)+[cellNum]):
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
        if self.tsPlotWindow and self.tsPlotWindow.hasSingleCellPlots():
            self.tsPlotWindow.updateSingleCellPlots()
    
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
        
    def newModels(self):
        self.modelChoose.clear()
        self.getModelNames()
        self.importModelClasses()
        self.modelChoose.addItems(sorted(self.modelNames, key = lambda x: x[0]))
        self.WEIGHTS_LOADED_STATUS = self.get_weights_loaded_status()
    


    def importModelClass(self, modelName):
        module = importlib.import_module(self.getPkgString(modelName))
        modelClass = getattr(module, capitalize_first_letter(modelName))
        return modelClass

    def importModelClasses(self):
        self.model_classes = {}
        for model in getBuiltInModelTypes():
            self.model_classes[model] = self.importModelClass(model)
    
    def getPkgString(self, string):
        return f"yeastvision.models.{string}.model"

    def getModelClass(self, modelName):
        if modelName not in self.model_classes:
            raise ValueError(f"{modelName} yet imported")
        else:
            return self.model_classes[modelName]
    
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
        weightPath1 = join(data["dir"], "models", data["model_name"])+self.model.prefix # to users working directory
        weightPath2 = join(MODEL_DIR, modelType, data["model_name"])+self.model.prefix # going to correct model dir
        self.model.train(self.trainIms, self.trainLabels, data, weightPath2)
        #check_gpu()

        if autorun and self.trainTStop < self.maxT:
            im = self.getCurrMaskSet()[self.trainTStop:self.trainTStop+1,:,:]
            mask, prob, flows = modelCls.run(im,None,weightPath2)
            self.tIndex = self.trainTStop
            self.label().insert(mask, self.tIndex, new_prob_im=prob, new_flows_im=flows )
        elif self.trainTStop == self.maxT:
            self.tIndex = self.trainTStop
            self.showError("no images left: cannot autorun new model on next image")
        self.drawMask()
        logger.info(f"Weights will be available for the user at {weightPath1}. They are also saved internall at {weightPath2} for use within yeastvision")
        logger.info(f"To remove your custom model from yeastvision, go to models>toggle model weights")
        shutil.copy2(weightPath1, weightPath2)
        
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
        if len(files) == imT:
            self.trainFiles = files
        else:
            self.trainFiles = [f"index_{i}" for i in range(len(ims))]


        self.trainTStop = min(imT, maskT)

        self.trainIms = [im for i,im in enumerate(ims[:self.trainTStop+1]) if i not in no_cell_frames]
        self.trainLabels = [im for i,im in enumerate(labels[:self.trainTStop+1]) if i not in no_cell_frames]
        self.trainFiles = [file for i, file in enumerate(self.trainFiles[:self.trainTStop+1]) if i not in no_cell_frames]
    
    def getUserRequestedModels(self):
        hyperparam = self.WEIGHTS_LOADED_STATUS
        types = [bool for hp in hyperparam]
        dlg = GeneralParamDialog(hyperparam, types, "Select Models to Load", self)
        if dlg.exec():
            return dlg.getData()
        else:
            return None
        
    def getModelNames(self):
        '''Retrieves all weight path file names along with their parent directory (model type)'''
        self.modelNames = []
        self.modelTypes = []
        dirs = [dir for dir in glob.glob(join(MODEL_DIR,"*")) if (os.path.isdir(dir) and "__" not in dir)]
        for dir in dirs:
            dirPath, dirName = os.path.split(dir)
            for dirFile in glob.glob(join(dir,"*")):
                dirPath2, dirName2 = os.path.split(dirFile)
                if (dirName2 != "__pycache__")  and ("." not in dirName2 or dirName2.endswith(".pt")):
                        self.modelNames.append(dirName2.split(".")[0])
                        self.modelTypes.append(dirName)
    
    def modelExists(self, model):
        if model != self.INTERP_MODEL_NAME:
            return getModelLoadedStatus(model)
        else:
            return rife_weights_loaded()
    
    def userLoadModels(self):
        models_to_load = self.getUserRequestedModels()
        if models_to_load is not None:
            self.toggleWeights(models_to_load)
            self.newModels()

    def toggleWeights(self, models_to_load):
        for model_name, status in models_to_load.items():
            is_custom = self.is_custom_model(model_name)
            if is_custom:
                if not status:
                    custom_model_type = self.modelTypes[self.modelNames.index(model_name)]
                    self.delete_custom_model(model_name, custom_model_type)   
            else:
                if (not status) and self.modelExists(model_name):
                    self.delete_model(model_name)
                elif status and not self.modelExists(model_name):
                    self.install_model(model_name)
    
    def is_custom_model(self, model_name):
        return model_name not in getBuiltInModelTypes() and model_name != self.INTERP_MODEL_NAME
    
    def install_model(self, model_name):
        if model_name == self.INTERP_MODEL_NAME:
            logger.info(f"installing {self.INTERP_MODEL_NAME}")
            install_rife()
            logger.info("RIFE INSTALLED")
        else:
            logger.info(f"Installing segmentation model {model_name}")
            install_weight(model_name)
    
    def delete_model(self, model):
        if model != self.INTERP_MODEL_NAME:
            model_path = produce_weight_path(model, model)
        else:
            model_path = RIFE_WEIGHTS_PATH
        logger.info(f"Removinng model {model} located at {model_path}")
        os.remove(model_path)
        assert (not os.path.exists(model_path))
    
    def delete_custom_model(self, custom_model_name, custom_model_type):
        model_path = produce_weight_path(custom_model_type, custom_model_name)
        os.remove(model_path)
            
    def getModelWeights(self, name = "proSeg"):
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


    def segment(self, output, modelClass, ims, params, weightPath, modelType, exp_idx, mask_i):
        #check_gpu()
        tStart, tStop = int(params["T Start"]), int(params["T Stop"])
        imName = params["Channel"]

        do_insert = mask_i is not None 
        if do_insert and not self.check_valid_insert(exp_idx, mask_i, ims.shape):
            self.showError("Masks must have same dimension along x and y to perform insertion. Loading generated masks as seperate mask channel")
            do_insert = False

        templates = []
        for i in [0,1,2]:
            template = np.zeros_like(ims, dtype = output[i].dtype)
            template[tStart:tStop+1] = output[i]
            templates.append(template)
        outputTup = (templates[0], templates[1], templates[2])

        if do_insert:
            self.do_segment_insert(exp_idx, mask_i, outputTup, tStart, tStop)
        else:
            self.loadMasks(outputTup, exp_idx=exp_idx, name =f"{imName}_{modelType}_{tStart}-{tStop}")
        
        torch.cuda.empty_cache()
        del modelClass
        del output

        #check_gpu()
    
    def check_valid_insert(self, exp_idx, mask_idx, ims_shape):
        exp = self.experiments[exp_idx]
        mask_template = exp.get_label("labels", idx = mask_idx)
        return mask_template.shape[-1] == ims_shape[-1] and mask_template.shape[-2] == ims_shape[-2]

    def extend_array_length(self, arr, new_length, dtype = np.uint16):
        old_length  =len(arr)
        r,c = arr.shape[1:]
        zeroes = np.zeros((new_length,r,c), dtype = np.uint16)
        zeroes[:old_length]=arr
        del arr
        return zeroes

    def do_segment_insert(self, exp_idx, mask_idx, data_tup, tStart, tStop):
        masks, probability, flowsXY = data_tup
        contours = np.array([get_mask_contour(mask) for mask in masks])
        exp = self.experiments[exp_idx]
        mask_template = exp.get_label("labels", idx = mask_idx)
        contour_template = exp.get_label("contours", idx = mask_idx)
        if len(mask_template)<len(masks):
            mask_template = self.extend_array_length(mask_template, len(masks), dtype = np.uint16)
            contour_template = self.extend_array_length(contour_template, len(contours), dtype = np.uint16)
        for i in range(tStart,tStop+1):
            mask_template[i] = masks[i]
            contour_template[i] = contours[i]
        del masks
        del contours

        if exp.labels[mask_idx].has_probability and exp.labels[mask_idx].has_flows:
            prob_template = exp.get_label("probability", idx = mask_idx)
            flow_template = prob_template.copy()
            if len(prob_template) < len(probability):
                prob_template = self.extend_array_length(prob_template, len(probability), dtype = np.float32)
            if len(flow_template) < len(flowsXY):
                flow_template = self.extend_array_length(flow_template, len(flowsXY), dtype = np.float32)
            for i in range(tStart,tStop+1):
                prob_template[i] = probability[i]
                flow_template[i] = flowsXY[i]
            del probability
            del flowsXY
        else:
            prob_template = probability
            flow_template = flowsXY
        
        self.experiments[exp_idx].labels[mask_idx].set_data(mask_template, contour_template, prob_template, flow_template)
        name = self.experiments[exp_idx].labels[mask_idx].name
        self.showTimedPopup(f"{name} HAS BEEN UPDATED ")
        self.maskChanged = True
        self.updateDisplay()

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
        allNames = self.getAllItems(self.labelSelect)  # Retrieves all current names from labelSelect
        
        if not allNames or potentialName not in allNames:
            return potentialName  # Return the potential name if there are no existing names or it's not a duplicate

        # The potential name exists, increment a suffix until a unique name is found
        suffix = 1
        new_name = f"{potentialName}-{suffix}"
        while new_name in allNames:
            suffix += 1
            new_name = f"{potentialName}-{suffix}"

        return new_name
    
    def getNewChannelName(self, potentialName):
        allNames = self.getAllItems(self.channelSelect)  # Retrieves all current names from labelSelect
        
        if not allNames or potentialName not in allNames:
            return potentialName  # Return the potential name if there are no existing names or it's not a duplicate

        # The potential name exists, increment a suffix until a unique name is found
        suffix = 1
        new_name = f"{potentialName}-{suffix}"
        while new_name in allNames:
            suffix += 1
            new_name = f"{potentialName}-{suffix}"

        return new_name
    
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
        
    def checkDoInsert(self, params):
        for param in params:
            if "insert into" in param:
                return params[param]

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
        dlg = dlgCls(copy.deepcopy(modelClass.hyperparams), modelClass.types, modelType, self)
        
        if dlg.exec():
            params = dlg.getData()
            do_insert = self.checkDoInsert(params)
        
            if do_insert:
                mask_template_i = self.maskZ
            else:
                mask_template_i = None

            channel = params["Channel"]
            channelIndex = self.channelSelect.findText(channel)
            ims = self.experiment().get_channel(idx = channelIndex)
            worker = SegmentWorker(modelClass,ims, params, self.experiment_index, 
                                   weightPath, modelType, mask_template_i =  mask_template_i)
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
        dlg = InterpolationDialog(parent=self)
        if dlg.exec():
            channel_name, intervals = dlg.get_data()
            if not intervals:
                return
        else:
            return

        channel = self.get_channel_obj_from_name(channel_name)
        ims = channel.ims
        currName = channel.name
        original_len = channel.max_t() + 1
        worker = InterpolationWorker(ims, f"{currName}_x{InterpolatedChannel.text_id}", 
                                     channel.annotations, self.experiment_index, 
                                     interpolate_intervals, intervals, original_len)
        self.runLongTask(worker, self.interpolationFinished, self.interpolateButton)
    
    def interpolationFinished(self, ims, exp_idx, name, annotations, intervals, original_len, new_intervals):
        experiment = self.experiments[exp_idx]
        new_channel = InterpolatedChannel(intervals=intervals, ims = ims, dir = experiment.dir, 
                                          name = self.getNewChannelName(name), annotations=annotations,
                                          original_len=original_len, new_intervals=new_intervals)
        experiment.add_channel_object(new_channel)
        self.experiment_index = exp_idx
        self.imZ = len(self.experiment().channels) - 1
        new_name = new_channel.name
        self.newInterpolation = True
        self.channelSelect.addItem(new_name)
        self.newInterpolation = False
        self.computeMaxT()
        self.drawIm()
        self.showTimedPopup(f"{name} HAS BEEN ADDED TO CHANNELS OF EXPERIMENT {self.experiments[self.experiment_index].name} ")
        self.checkInterpolation()
    
    def interpRemoveButtonClicked(self):
        assert isinstance(self.channel(), InterpolatedChannel)
        lengthsMatch = self.channel().max_t() == self.label().max_t()
        if not lengthsMatch:
            self.showError("Error: Length of label must match length of interpolated channel to perform this operation. Re-select channel and masks from dropdown menu.")
            return
        interp_bool = self.channel().interp_annotations
        masks = self.getAllMaskTypes()
        deinterpolated = [deinterpolate(mask_set, interp_bool) for mask_set in masks]
        currMaskName = self.labelSelect.currentText()
        self.loadMasks(deinterpolated, name = f"{currMaskName}_deinterpolated")

    def doAdaptHist(self):
        curr_im = self.channel()
        annotation_name = "adaptive_hist_eq"
        curr_im = self.channel()
        annotations = self.add_annotations(curr_im, annotation_name)
        newIms = im_funcs.do_adapt_hist(self.getCurrImSet())
        self.newIms(ims = newIms, dir = self.experiment().dir, name = self.getNewChannelName(curr_im.name+f"-{annotation_name}"), 
                    annotations = annotations )

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
        self.newIms(ims = newIms, dir = self.experiment().dir, name = self.getNewChannelName(curr_im.name+f"-{annotation_name}"), 
                    annotations = annotations)

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
        self.newIms(ims = newIms, dir = self.experiment().dir, name = self.getNewChannelName(curr_im.name+f"-{annotation_name}"), 
                    annotations = annotations )

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
        self.newIms(ims = newIms, dir = self.experiment().dir, name = self.getNewChannelName(curr_im.name+f"-{annotation_name}"), 
                    annotations = annotations )

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
        self.newIms(ims = newIms, dir = self.experiment().dir, name = self.getNewChannelName(curr_im.name+f"-{annotation_name}"), 
                    annotations = annotations )
    
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
            if isinstance(self.channel(), InterpolatedChannel):
                write_images_to_dir(path, self.getCurrImSet(), self.channel().interp_annotations, annotation="_interp")
            else:
                write_images_to_dir(path, self.getCurrImSet())
    
    def saveMasks(self):
        if not self.maskLoaded:
            self.showError("No masks to save")
        defaultFileName = join(self.getCurrImDir(), self.labelSelect.currentText().replace(" ", ""))
        path, _ = QFileDialog.getSaveFileName(self, 
                            "save masks to a directory", 
                            defaultFileName)
        if path:
            mask_obj = self.label()
            mask_set = self.getCurrMaskSet()
            intervals = nonzero_intervals(mask_set)
            t1, t2 = get_longest_interval(intervals)

            dlg = IntervalSelectionDialog(intervals, mask_obj.max_t(), 
                                          f"Select Intervals of {mask_obj.name} to save",
                                          parent = self, labels = f"nonzero intervals: {', '.join(intervals)}", 
                                          presetT1=t1, presetT2 = t2)
            if dlg.exec():
                start, end = dlg.get_selected_interval()
                write_images_to_dir(path, mask_set[start:end])
            else:
                return

    def saveFigure(self):
        if (not self.imLoaded):
            self.showError("No Images Loaded")
            return

        defaultFileName = join(self.getCurrImDir(), self.labelSelect.currentText().replace(" ", "") + "-figure")
        path, _ = QFileDialog.getSaveFileName(self, 
                            "save figures to directory", 
                            defaultFileName)
        if path:
            write_images_to_dir(path, self.createFigure(), extension=".jpeg")

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
            celldata = self.label().celldata.cell_data
            potential_heatmaps = getPotentialHeatMapNames(celldata)
            dlg = GeneralParamDialog({name:False for name in potential_heatmaps}, 
                                     [bool]*len(potential_heatmaps), 
                                     "heatmap property selection", self)
            if dlg.exec():
                chosen = [name for name, status in dlg.getData().items() if status]
                heatmaps = getHeatMaps(celldata, chosen)
                base_dir = QFileDialog.getExistingDirectory(self, caption = "save heatmap directory to specified location")
                dir = join(base_dir, f"{labelName}_heatmaps")
                if not os.path.exists(dir):
                    os.mkdir(dir)
                for name, heatmap in heatmaps.items():
                    out_path = join(dir, name+"_heatmap.tif")
                    imsave(out_path, heatmap)
            else:
                return
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
        if self.tsPlotWindow:
            self.tsPlotWindow.close()
            self.tsPlotWindow = None
        if self.lineageWindow:
            self.lineageWindow.close()
            self.lineageWindow = None
    
    def getCurrImDir(self):
        return self.experiment().channels[self.imZ].dir
    
    def checkGPUs(self):
        self.checkTorchGPU()

    def checkTorchGPU(self):
        self.torch = check_torch_gpu()
    
    def checkDataAvailibility(self):
        hasLineages = self.hasLineageData()
        self.hasLineageBox.setChecked(hasLineages)
        self.hasCellDataBox.setChecked(self.hasCellData())
        self.showMotherDaughtersButton.setEnabled(hasLineages)
        self.showLineageButton.setEnabled(hasLineages)
        self.showTreeButton.setEnabled(hasLineages)
    
    def checkInterpolation(self):
        hasMasks = self.experiments and self.experiment().has_labels()
        currIsInterp = hasMasks and isinstance(self.channel(), InterpolatedChannel)
        has_rife_weights = rife_weights_loaded()

        if not currIsInterp or not hasMasks:
            self.disableInterpRemove()
        else:
            self.enableInterpRemove()
        
        if not has_rife_weights:
            self.disableInterp()
            return
        
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

#@profile
def main():

    parser = argparse.ArgumentParser(description="Yeastvision accepts 3 optional command line argumennts. \
        Example usage: python script.py -test \
        or python script.py --dir /path/to/directory --num_channels 10")

    parser.add_argument("--test", action="store_true",
                        help="Use the test directory. No arguments needed. \
                        Example: --test")
    parser.add_argument("--dir", type=str, 
                        help="Specify the directory to use. Must be an existing directory. \
                        Example: --dir /path/to/directory",
                        required=False)
    parser.add_argument("--num_channels", type=int, default=1,
                        help="Specify the number of channels in --dir. Must be an integer (defaults to 1). Only used is --dir is specified.  \
                        Example: --num_channels 10",
                        required=False)
    parser.add_argument("--ims_only", action="store_true", help="Exclude files with the keyword <mask> anywhere in the filepath. Example: --ims_only")
    args = parser.parse_args()
    dir = None
    test_dir = TEST_MOVIE_DIR
    num_channels = 1
    ims_only = args.ims_only

    if args.test:
        dir = test_dir
        if not os.path.exists(TEST_MOVIE_DIR) or glob.glob(join(TEST_MOVIE_DIR, "*.tif")) == []:
            logger.info(f"Installing Test Images from {TEST_MOVIE_URL}")
            install_test_ims()
        num_channels = 2
    elif args.dir is not None and (os.path.exists(args.dir) or os.path.isdir(args.dir)):
        dir = args.dir
        logger.info(f"Loading {args.dir}")
        num_channels = args.num_channels
    elif args.dir is not None and (not os.path.exists(args.dir) or not os.path.isdir(args.dir)):
        logger.info(f"{args.dir} is not a valid directory")

    app = QApplication([])
    app_icon = QtGui.QIcon()
    icon_path = "yeastvision/docs/figs/logo.png"
    app_icon.addFile(icon_path, QtCore.QSize(16, 16))
    app_icon.addFile(icon_path, QtCore.QSize(24, 24))
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))
    app_icon.addFile(icon_path, QtCore.QSize(48, 48))
    app_icon.addFile(icon_path, QtCore.QSize(64, 64))
    app_icon.addFile(icon_path, QtCore.QSize(256, 256))
    app.setWindowIcon(app_icon)
    window = MainWindow(dir = dir, dir_num_channels=num_channels, dir_ims_only=ims_only)
    window.show()
    app.exec_()      

if __name__ == "__main__":
    main()
    #cleanup_npz_files_from_sample_movie(__file__)




        