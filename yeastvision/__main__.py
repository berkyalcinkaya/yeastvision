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

class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class QVLine(QFrame):
    def __init__(self):
        super(QVLine, self).__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)


class Worker(QtCore.QObject):
    '''Handles Multithreading'''
    finished = QtCore.pyqtSignal()
    def __init__(self, parent, process, button = None, thread = None):
        super(Worker,self).__init__()
        self.parent = parent
        self.process = process
        self.button = button
        self.thread = thread
        self.finished.connect(self.handleFinished)
        self.error = False

    def handleFinished(self, error = False):
        if self.button:
            self.parent.activateButton(self.button)
        self.parent.statusBar.clearMessage()
        self.parent.closeThread(self.thread)
        
    def run(self):
        #try:
        self.process()
        self.parent.updateDisplay()
        self.finished.emit()
        # except:
        #     self.error = True
        #     self.finished.emit()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, thread, imPaths = None, labelPaths = None):
        super(MainWindow, self).__init__()
        pg.setConfigOption('imageAxisOrder', 'row-major')
        self.setGeometry(50, 50, 1000, 1000)
        self.setAcceptDrops(True)

        #self.idealThreadCount  = thread.idealThreadCount()//2
        self.idealThreadCount = 2

        self.sessionId = self.getCurrTimeStr()
        
        self.goldColor = [255,215,0,255]

        self.setStyleSheet("QMainWindow {background: 'black';}")
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
        self.setStyleSheet("QMainWindow {background: 'black';}")
        self.cwidget = QWidget(self)
        self.l = QGridLayout()
        self.cwidget.setLayout(self.l)
        self.setCentralWidget(self.cwidget)
        self.l.setSpacing(3)

        self.win = pg.GraphicsLayoutWidget()
        self.mainViewRows = 30
        self.mainViewCols  = 19
        self.l.addWidget(self.win, 1,0, self.mainViewRows, self.mainViewCols)
        self.make_viewbox()

        self.labelX, self.labelY = None, None
        self.currMaskDtype = "uint8"
        self.currImDtype = "uint8"

        self.threads = []
        self.workers = []
        
        self.setEmptyDisplay(initial=True)
        self.build_widgets()
        self.l.setHorizontalSpacing(8)
        self.l.setVerticalSpacing(8)
        
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
        self.plotPropsBeenChecked = False
        self.toPlot = None
        self.emptying = False

        self.overrideNpyPath = None

        self.win.show()
    
    @property
    def maxT(self):
        return self.imData.maxT
    @maxT.setter
    def maxT(self, num):
        self._maxT = num
    
    def computeMaxT(self):
        self.maxT = self.imData.maxT
        self.tIndex =  min(self.tIndex, self.imData.maxT, self.maskData.maxT)

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
        # if self.imData.maxTs[self._imZ] < self.tIndex:
        #     self.tIndex = 0
    
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
                self.checkDataAvailibility()
                # if self.pWindow:
                #     self.
            except IndexError:
                pass
        
    
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
    
    def setEmptyDisplay(self, initial = False):
        self.tIndex = 0
        self.bigTStep = 3
        self.maxT = 0
        self.tChanged = False

        if not initial:
            self.overrideNpyPath = None
            self.sessionId = self.getCurrTimeStr()
            self.updateDataDisplay()
            self.emptying = True
            self.empyting = False


        self.setEmptyIms(initial = initial)
        self.setEmptyMasks(initial = initial)
        self.saveData()
    
    def setEmptyIms(self, initial = False):
        # if self.maskLoaded:
        #     self.errorDialog = QtWidgets.QErrorMessage()
        #     self.errorDialog.showMessage("Images cannot be removed with masks present")
        #     return

        self.imData = ImageData(self)
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
        self.maskType = 0 # masks = 0, floats = 1
        self.cmaps = [self.getMaskCmap(), self.getFloatCmap()]
        self.cmap = self.cmaps[0]
        self.maskColors = self.cmap.copy()
        self.maskData = MaskData(self)
        self.maskChanged = False
        self.prevMaskOn = True
        self.maskOn = True
        self.contourOn = False
        self.probOn = False
        self.maskZ = -1
        self.currMask = np.zeros((512,512), dtype = np.uint8)
        self.pg_mask.setImage(self.currMask, autolevels  =False, levels =[0,0], lut = self.maskColors)
        self.cellData = [] 
        self.lineageData = []
        self.mother_daughters = []

        if not initial:
            self.labelSelect.clear()
            self.maskTypeSelect.uncheckAll()
            self.contourButton.setChecked(False)
            self.cellNumButton.setChecked(False)

        if self.imLoaded:
            self.loadDummyMasks()

    def getCurrTimeStr(self):
        now = datetime.now()
        return now.strftime("%d-%m-%Y|%H-%M")
    
    def newData(self):
        self.computeMaxT()
        self.updateDisplay()
        self.saveData()

    def newIms(self, name = None):
        if not self.imLoaded:
            self.imLoaded = True
            self.enableImageOperations()

        self.imZ+=1
        self.imChanged = True

        if name:
            self.channelSelect.addItem(self.getNewChannelName(name))
        else:
            self.channelSelect.addItem("Channel " + str(self.imData.maxZ))

        if self.firstMaskLoad:
            self.loadDummyMasks()
            self.firstMaskLoad = False
        self.newData()

    
    def newMasks(self, name = None):
        if not self.maskLoaded:
            self.maskLoaded = True
            self.enableMaskOperations()
        self.maskChanged = True

        if name:
            self.labelSelect.addItem(self.getNewLabelName(name))
        else:
            self.labelSelect.addItem("Label " + str(self.maskData.maxZ))
        
        self.maskTypeSelect.checkMaskRemote()

        self.lineageData.append(None)
        self.mother_daughters.append(None)
        self.cellData.append(None)
        self.newData()
    
    def handleDummyMasks(self, add = True):
        if add:
            self.maskData.addZ(self.maskData.channels[self.maskZ][0,:,:,:])
        self.maskData.removeDummys()
        self.maskZ = 0
    
    def addBlankMasks(self):
        if not self.imLoaded:
            self.showError("Load Images First")
            return
        blankMasks = np.zeros_like(self.getCurrImSet())
        self.loadMasks(blankMasks, name = "Blank")

    def loadDummyMasks(self):
        y,x = self.imData.y,self.imData.x
        dummyMasks = np.zeros((self.imData.maxT+1,y,x), dtype = np.uint8)
        self.maskData.addDummyZ(dummyMasks)
        self.maskZ+=1
        self.maskChanged = True
    
    def loadDummyIms(self):
        y,x = self.maskData.y,self.maskData.x
        self.imData.addDummyZ(np.zeros((self.maxT+1,y,x), dtype = np.uint8))
        self.imChanged = True

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
        rowspace = self.mainViewRows
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

        self.statusBarLayout.addWidget(self.cpuCoreDisplay, 0, 0, 1,1 )
        self.statusBarLayout.addWidget(self.gpuDisplayTF, 0, 1, 1, 1)
        self.statusBarLayout.addWidget(self.gpuDisplayTorch, 0, 2, 1, 1)
        self.statusBarLayout.addWidget(self.hasLineageBox, 0,3,1,1)
        self.statusBarLayout.addWidget(self.hasCellDataBox,0,4,1,1)
        self.statusBar.addPermanentWidget(self.statusBarWidget)
        
        self.dataDisplay = QLabel("")
        self.dataDisplay.setMinimumWidth(300)
        # self.dataDisplay.setMaximumWidth(300)
        self.dataDisplay.setStyleSheet(self.labelstyle)
        self.dataDisplay.setFont(self.smallfont)
        self.dataDisplay.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.updateDataDisplay()
        self.l.addWidget(self.dataDisplay, rowspace-1,0,1,20)

        label = QLabel('Drawing:')#[\u2191 \u2193]')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.l.addWidget(label, rowspace,0,1,5)

        label = QLabel("Brush Type:")
        label.setStyleSheet(self.labelstyle)
        label.setFont(self.medfont)
        self.l.addWidget(label,rowspace+1,0,1,2)
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
        self.l.addWidget(self.brushTypeSelect, rowspace+1,2,1,1)

        
        label = QLabel("Brush Size")
        label.setStyleSheet(self.labelstyle)
        label.setFont(self.medfont)
        self.l.addWidget(label, rowspace+2,0,1,2)
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
        self.l.addWidget(self.brushSelect, rowspace+2,2,1,1)

        line = QVLine()
        line.setStyleSheet('color:white;')
        self.l.addWidget(line, rowspace,3,6,1)

        label = QLabel('Tracking')#[\u2191 \u2193]')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.l.addWidget(label, rowspace,4,1,4)

        self.cellNumButton = QCheckBox("cell nums")
        self.cellNumButton.setStyleSheet(self.checkstyle)
        self.cellNumButton.setFont(self.medfont)
        self.cellNumButton.stateChanged.connect(self.toggleCellNums)
        self.cellNumButton.setEnabled(False)
        self.l.addWidget(self.cellNumButton, rowspace+1, 4, 1,2)

        self.showLineageButton = QCheckBox("lineages")
        self.showLineageButton.setStyleSheet(self.checkstyle)
        self.showLineageButton.setFont(self.medfont)
        self.showLineageButton.stateChanged.connect(self.toggleLineages)
        self.showLineageButton.setEnabled(False)
        self.l.addWidget(self.showLineageButton, rowspace+1, 6, 1,2)

        self.trackButton = QPushButton('track cells')
        self.trackButton.setFixedWidth(90)
        self.trackButton.setFixedHeight(20)
        self.trackButton.setStyleSheet(self.styleInactive)
        self.trackButton.setFont(self.medfont)
        self.trackButton.clicked.connect(self.trackButtonClick)
        self.trackButton.setEnabled(False)
        self.trackButton.setToolTip("Track current cell labels")
        self.l.addWidget(self.trackButton, rowspace+2, 4,1,2)

        self.trackObjButton = QPushButton('track to cell')
        self.trackObjButton.setFixedWidth(90)
        self.trackObjButton.setFixedHeight(20)
        self.trackObjButton.setFont(self.medfont)
        self.trackObjButton.setStyleSheet(self.styleInactive)
        self.trackObjButton.clicked.connect(self.trackObjButtonClick)
        self.trackObjButton.setEnabled(False)
        self.trackObjButton.setToolTip("Track current non-cytoplasmic label to a cellular label")
        self.l.addWidget(self.trackObjButton, rowspace+2, 6,1,2)

        self.lineageButton = QPushButton("get lineages")
        self.lineageButton.setStyleSheet(self.styleInactive)
        self.lineageButton.setFixedWidth(90)
        self.lineageButton.setFixedHeight(18)
        self.lineageButton.setFont(self.medfont)
        self.lineageButton.setToolTip("Use current budNET mask to assign lineages to a cellular label")
        self.lineageButton.setEnabled(False)
        self.showMotherDaughters = False
        self.showLineages = False
        self.lineageButton.clicked.connect(self.getLineages)
        self.l.addWidget(self.lineageButton, rowspace+3, 4,1,2, Qt.AlignBottom)

        self.showMotherDaughtersButton = QCheckBox("mother-daughters")
        self.showMotherDaughtersButton.setStyleSheet(self.checkstyle)
        self.showMotherDaughtersButton.setFont(self.medfont)
        self.showMotherDaughtersButton.stateChanged.connect(self.toggleMotherDaughters)
        self.showMotherDaughtersButton.setEnabled(False)
        self.l.addWidget(self.showMotherDaughtersButton, rowspace+3, 6, 1,2)

        line = QVLine()
        line.setStyleSheet('color:white;')
        self.l.addWidget(line, rowspace,8,6,1)
    
        label = QLabel('Segmentation')#[\u2191 \u2193]')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.l.addWidget(label, rowspace,9,1,5)
        
        #----------UNETS-----------
        self.GB = QGroupBox("Unets")
        self.GB.setStyleSheet("QGroupBox { border: 1px solid white; color:white; padding: 10px 0px;}")
        self.GBLayout = QGridLayout()
        self.GB.setLayout(self.GBLayout)
        self.GB.setToolTip("Select Unet(s) to be used for segmenting channel")
    
        self.getModels()
        self.modelChoose = QComboBox()
        self.modelChoose.addItems(sorted(self.modelNames, key = lambda x: x[0]))
            #self.modelChoose.setItemChecked(i, False)
        self.modelChoose.setFixedWidth(180)
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
        self.l.addWidget(self.GB, rowspace+1,9,3,5, Qt.AlignTop | Qt.AlignHCenter)

        #------Flourescence Segmentation -----------pp-p-
        self.segButton = QPushButton(u'blob detection')
        self.segButton.setEnabled(False)
        self.segButton.clicked.connect(self.doFlou)
        self.segButton.setStyleSheet(self.styleInactive)
        self.l.addWidget(self.segButton, rowspace+1+3,9,3,5, Qt.AlignTop | Qt.AlignLeft)

        #-----------------------------------------------

        line = QVLine()
        line.setStyleSheet('color:white;')
        self.l.addWidget(line, rowspace,14,6,1)

        label = QLabel('Display')#[\u2191 \u2193]')
        label.setStyleSheet(self.headings)
        label.setFont(self.boldfont)
        self.l.addWidget(label, rowspace,15,1,5)

        self.contourButton = QCheckBox("Show Contours")
        self.contourButton.setStyleSheet(self.checkstyle)
        self.contourButton.setFont(self.medfont)
        self.contourButton.stateChanged.connect(self.toggleContours)
        self.contourButton.setShortcut(QtCore.Qt.Key_C)
        self.contourButton.setEnabled(False)
        self.l.addWidget(self.contourButton, rowspace+1, 15,1,2)

        self.plotButton = QCheckBox("Show Plot Window")
        self.plotButton.setStyleSheet(self.checkstyle)
        self.plotButton.setFont(self.medfont)
        self.plotButton.stateChanged.connect(self.togglePlotWindow)
        self.plotButton.setShortcut(QtCore.Qt.Key_P)
        self.l.addWidget(self.plotButton, rowspace+1, 17, 1,2)

        self.channelSelect = QComboBox()
        self.channelSelect.setStyleSheet(self.dropdowns)
        self.channelSelect.setFont(self.medfont)
        self.channelSelect.setCurrentText("")
        self.channelSelect.currentIndexChanged.connect(self.channelSelectIndexChange)
        self.channelSelect.setEditable(True)
        self.channelSelect.editTextChanged.connect(self.channelSelectEdit)
        self.channelSelect.setFixedWidth(120)
        self.l.addWidget(self.channelSelect, rowspace+2,15,1,2)
        self.l.setAlignment(self.channelSelect, QtCore.Qt.AlignLeft)
        setattr(self.channelSelect, "items", lambda: [self.channelSelect.itemText(i) for i in range(self.channelSelect.count())])

        self.labelSelect = QComboBox()
        self.labelSelect.setStyleSheet(self.dropdowns)
        self.labelSelect.setFont(self.medfont)
        self.labelSelect.setCurrentText("")
        self.labelSelect.currentIndexChanged.connect(self.labelSelectIndexChange)
        self.labelSelect.setEditable(True)
        self.labelSelect.editTextChanged.connect(self.labelSelectEdit)
        self.labelSelect.setFixedWidth(120)
        self.labelSelect.setEnabled(False)
        self.l.addWidget(self.labelSelect, rowspace+2,17,1,2)
        self.l.setAlignment(self.labelSelect, QtCore.Qt.AlignLeft)
        setattr(self.labelSelect, "items", lambda: [self.labelSelect.itemText(i) for i in range(self.labelSelect.count())])

        self.maskTypeSelect  = MaskTypeButtons(parent=self, row = rowspace+3, col = 15)

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
        self.maskTypeSelect.setEnabled(True)
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
        self.segButton.setEnabled(True)
        self.segButton.setStyleSheet(self.styleUnpressed)
    
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
        self.modelChoose.addItems(sorted(self.modelNames))


    def channelSelectEdit(self, text):
        self.channelSelect.setItemText(self.channelSelect.currentIndex(), str(text))

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
        self.labelSelect.setItemText(self.labelSelect.currentIndex(), str(text))

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
        labelsToTrack =  self.labelSelect.currentText()
        dlg  = GeneralParamDialog({}, [], f"Track {labelsToTrack}", self, labelSelects=["Cytoplasm Label"])

        if dlg.exec():
            self.trackObjButton.setEnabled(False)
            self.trackObjButton.setStyleSheet(self.stylePressed)
            data = dlg.getData()

            cellIdx = self.labelSelect.findText(data["Cytoplasm Label"])
            cells = self.maskData.channels[cellIdx][0, :, :,:]

            worker = Worker(self, lambda: self.trackObj(self.maskZ, cells), self.trackObjButton)

            try:
                self.beginThread(worker)
            except MemoryError:
                error_dialog  = QErrorMessage()
                error_dialog.showMessage(f"Cannot Track Until Other Processes are Finished")
                self.activateButton(self.trackObjButton)
                return
        else:
            return
    
    def trackObj(self, z, cells):
        tracked = track_to_cell(self.maskData.channels[z][0, :, :,:], cells)
        self.maskData.channels[z][0,:,:,:] = tracked
        self.updateCellData(idx = z)


    def trackButtonClick(self):
        self.trackButton.setEnabled(False)
        self.trackButton.setStyleSheet(self.stylePressed)

        idxToTrack = self.maskZ
        worker  = Worker(self, 
                        lambda: self.track(idxToTrack),
                        self.trackButton)
        try:
            self.beginThread(worker)
        except MemoryError:
            error_dialog  = QErrorMessage()
            error_dialog.showMessage(f"Cannot Track Until Other Processes are Finished")
            self.activateButton(self.trackButton)
            return
        self.maskZ = idxToTrack
        self.drawMask()
    
    def track(self,z):
        currMasks = self.getCurrMaskSet()
        newMasks = trackYeasts(currMasks)
        self.maskData.channels[z][0,:,:,:] = newMasks
        self.cellData[z] = None
        self.updateCellData(idx = z)
        
    def updateCellData(self, idx = None):
        if not self.maskLoaded:
            return
        self.toStatusBar("Updating Cell Data...")
        if idx:
            masks  = self.maskData.channels[idx][0,:,:,:]
        else:
            masks = self.getCurrMaskSet()
            idx = self.maskZ
    
        viableIms = []
        viableImNames = []
        for i, imChannel in enumerate(self.imData.channels):
            if imChannel[0].shape == masks[0].shape and imChannel.shape[0] >= masks.shape[0]:
                viableIms.append(imChannel)
                viableImNames.append(self.channelSelect.itemText(i))
        #try:
        cellData = getCellData(masks, viableIms, viableImNames)    
        # except:
        #     self.showError("Error Getting Cell Data: Ensure Labels Have Been Tracked")    
        #     return      
                                                             
        self.cellData[idx] = cellData
        self.statusBar.clearMessage()
        self.saveData()

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
        return (type(self.cellData[i]) is pd.core.frame.DataFrame)
    
    def hasLineageData(self, i = None):
        if not i:
            i = self.maskZ
        return ((type(self.lineageData[self.maskZ]) is pd.core.frame.DataFrame) 
                and type(self.mother_daughters[self.maskZ]) is np.ndarray)
    
    def getLabelsWithPopulationData(self):
        return [population for i,population in enumerate(self.labelSelect.items()) if self.hasCellData(i = i)]
    

    def getCellData(self):
        if not self.maskLoaded or not self.hasCellData():
            return None
        else:
            return self.cellData

    def getCellDataAbbrev(self):
        if not self.maskLoaded:
            return pd.DataFrame.from_dict(self.cellDataFormat)
        
        if not self.hasCellData():
            if self.hasLineageData():
                return self.lineageData[self.maskZ]
            else:
                return pd.DataFrame.from_dict(self.cellDataFormat)

        elif self.maskLoaded and self.hasCellData():
            data = self.cellData[self.maskZ].copy()[0:-1]
            columnsToKeep = ["labels", "birth", "death"]
            for columnName in data.columns:
                if columnName not in columnsToKeep:
                    data = data.drop(columns = columnName)

            if self.hasLineageData():
                lD = self.lineageData[self.maskZ]
                data["mother"], data["confidence"] = lD["mother"], lD["confidence"]
            return data
    
    def getLineages(self):
        dlg = GeneralParamDialog({}, [], "Lineage Construction", self, labelSelects=["bud necks", "cells"])
        if dlg.exec_():
            data = dlg.getData()
            neckI = self.labelSelect.findText(data["bud necks"])
            cellI = self.labelSelect.findText(data["cells"])
            necks = self.maskData.channels[neckI][0,:,:,:]
            cells = self.maskData.channels[cellI][0,:,:,:]
        else:
            return 
        lng = LineageConstruction(cells, necks, backskip=0, forwardskip=5)
        daughters, motherDF = lng.computeLineages()
        self.mother_daughters[cellI] = daughters
        self.lineageData[cellI] = motherDF
        self.saveData()
        self.checkDataAvailibility()
    
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
            columns  = self.cellData[self.maskZ].columns
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

    def beginThread(self, worker):
        if len(self.threads)+1>self.idealThreadCount:
            self.showError("Too Many Threads Running At This Time")
            return
        thread = QtCore.QThread(self) 
        self.threads.append(thread)
        self.workers.append(worker)
        self.workers[-1].thread = thread
        self.workers[-1].moveToThread(thread)
        thread.started.connect(self.workers[-1].run)
        self.threads[-1].start()
        self.updateThreadDisplay()

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
    

    
    def parseProperties(self):
            if not self.toPlot:
                return {}
            else:
                toPlotDict = {}
                for plot in self.plotTypes:
                    toPlotDict[plot] = ["-".join((prop.split("-")[1:])) for prop in self.toPlot if plot in prop.split("-")[0]]
                return toPlotDict
        
        
    def buildPlotWindow(self):
        win = PlotWindowCustomize(self)

        if win.exec_():
            self.toPlot = win.getData()
        else:
            self.plotButton.setCheckState(False)
            return
        
        self.pWindow = plot.PlotWindow(self, self.parseProperties())
        self.pWindow.show()
    
        
    def togglePlotWindow(self):
        self.plotWindowOn = self.plotButton.isChecked()

        if self.plotWindowOn:
            self.buildPlotWindow()   
        else:
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
        contourMask = self.maskData.contours[self.maskZ][self.tIndex,:,:]
        cellMask = self.maskData.channels[self.maskZ][self.maskType, self.tIndex,:,:]
        self.maskData.contours[self.maskZ][self.tIndex,:,:] = self.addCellContour(contourMask, cellMask, newColor)
        
        #self.saveData()

    def updateDataDisplay(self, x = None,y = None, val = None):
        dataString = f"Session Id: {self.sessionId}"

        if self.imLoaded:
            files = self.imData.files[self.imZ]
            if len(files) > 1:
                file = files[self.tIndex]
            else:
                file = files[0]
            dataString += f'  |  {file}'

            if x!=None and y!=None and val!=None:
                self.labelX, self.labelY = x,y
                if self.maskType == 1:
                    val = round(val/255,2)
                dataString += f"\nx={str(x)}, y={str(y)}, value={str(val)}"

            dataString += f"\nIMAGE: {self.currIm.shape} {self.currImDtype}"
            
            if self.maskLoaded:
                dataString += f"  |  MASK: {self.currMask.shape} {self.currImDtype}"

            dataString += f"  |  TIME: {self.tIndex}/{self.maxT}"

            if self.maskLoaded:
                dataString += f"  |  REGIONS: {self.numObjs}"

        self.dataDisplay.setText(dataString)
    
    def deleteCell(self, cellNum):
        stroke = self.currMask == cellNum
        self.strokes.append((stroke,cellNum))
        self.currMask[stroke] = 0
        self.maskData.contours[self.maskZ][self.tIndex,:,:][stroke]=0
        
        self.saveData()

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
    
    def toStatusBar(self, message, time = None):
        self.statusBar.clearMessage()
        if time:
            self.statusBar.showMessage(message, time)
        else:
            self.statusBar.showMessage(message)
    
    def setChannelSelectName(self, name):
        self.channelSelect.setItemText(self.channelSelect.currentIndex(), self.getNewChannelName(name))
    
    def setLabelSelectName(self, name):
        self.labelSelect.setItemText(self.labelSelect.currentIndex(), self.getNewLabelName(name)) 

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if ".pkl" in files[0]:
            self.toStatusBar("Loading Npy... ")
            loadPkl(files[0], self)
            
        elif not self.imLoaded:        
            self.loadImageFiles(files[0])
        else:
            if self.isUpperHalf(event):
                self.loadMaskFiles(files[0])
            else:
                self.loadImageFiles(files[0])
        self.statusBar.clearMessage()
    
    def loadImageFiles(self, file):
        self.toStatusBar("Loading Image...")
        self.imData.load(file)
        self.newIms(name = file.split("/")[-1].split(".")[0])

    def loadMaskFiles(self, file):
        self.toStatusBar("Loading Mask & Extracting Region Contours...")
        self.maskData.load(file)
        self.newMasks(name = (file.split("/")[-1].split(".")[0]).replace("-", "_"))

    def loadImages(self, ims, name = None):
        self.imData.addZ(ims)
        self.newIms(name = name)

    def loadMasks(self, masks, name = None, contours = None):
        self.maskData.addZ(masks, contours = contours)
        self.newMasks(name = name)
    
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
            for selectedCell in self.selectedCells:
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
                if self.maskZ<self.maskData.maxZ:
                    self.maskZ+=1
                else:
                    self.maskZ = 0
                nextMaskOn = True
            else:
                if self.imZ<self.imData.maxZ:
                    self.imZ+=1
                else:
                    self.imZ = 0

        if event.key() == QtCore.Qt.Key_Down:
            if event.modifiers() & QtCore.Qt.ControlModifier:
                if self.maskZ>0:
                    self.maskZ-=1
                else:
                    self.maskZ = self.maskData.maxZ
                nextMaskOn = True
            else:
                if self.imZ>0:
                    self.imZ-=1
                else:
                    self.imZ = self.imData.maxZ

        if event.key() == QtCore.Qt.Key_0 or event.key() == QtCore.Qt.Key_Space:
            nextMaskOn = not self.maskOn
            self.toggleMask()
        
        if self.maskOn == False and nextMaskOn == True:
            self.addMask()
        self.updateDisplay()

    def toggleMask(self):
        if self.maskOn:
            self.hideMask()
        else:
            self.addMask()

    def hideMask(self):
        self.maskOn = False
        self.maskTypeSelect.uncheckAll()
        self.drawMask()

    def addMask(self):
        self.prevMaskOn = False
        self.maskOn = True

        if self.maskType == 0:
            self.maskTypeSelect.checkMaskRemote()
        elif self.maskType == 1:
            self.maskTypeSelect.checkProbRemote()
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
        potentialDaughters = self.mother_daughters[self.maskZ][cellNum-1]
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
        return self.imData.channels[self.imZ]
    
    def getCurrMaskSet(self):
        if not self.maskLoaded:
            return np.zeros((1,512,512))
        return self.maskData.channels[self.maskZ][0,:,:,:]
    
    def getCurrIm(self):
        if self.tIndex > self.imData.maxT:
            return np.zeros((self.maskData.y, self.maskData.x), dtype = np.uint8)
        return self.imData.channels[self.imZ][self.tIndex,:,:]
    
    def getCurrMask(self):
        if self.maskOn:
            maskData = self.maskData.channels
            if self.tIndex>self.maskData.maxT:
                return np.zeros((self.imData.y,self.imData.x), dtype = np.uint8)
            return maskData[self.maskZ][self.maskType, self.tIndex,:,:]
        else:
            return np.zeros_like(self.currMask)

    def getCurrContours(self):
        maskData = self.maskData.contours
        return maskData[self.maskZ][self.tIndex,:,:]

    def getCurrFrames(self):
        return (self.getCurrIm(), self.getCurrMask())
    
    def bootUp(self):
        self.pg_im.setImage(self.currIm, autoLevels=False, levels = [0,0])
        self.pg_mask.setImage(self.currMask, autolevels  =False, levels =[0,0], lut = self.maskColors)

    def updateDisplay(self):
        if self.imChanged:
            self.drawIm()

        if self.maskChanged:
            self.drawMask()
        
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
        return self.imData.saturation[self.imZ][self.tIndex]
                                
    def selectCell(self, pos):
        pos = [int(pos.y()), int(pos.x())]
        cellNum = self.currMask[pos[0], pos[1]]
        self.selectCellFromNum(int(cellNum))
    
    def selectCellFromNum(self, cellNum):
        if self.showLineages:
            self.deselectAllCells()
            for num in self.getDaughters(cellNum)+[cellNum]:
                print(num)
                self.selectedCells.append(num)
                self.selectedCellContourColors.append([255,0,0,255])
                self.maskColors[num,:] = np.array(self.goldColor, dtype = np.uint8)

        elif cellNum in self.selectedCells:
            self.deselectCell(cellNum)
        else:
            self.selectedCells.append(cellNum)
            self.selectedCellContourColors.append([255,0,0,255])
            self.maskColors[cellNum,:] = np.array(self.goldColor, dtype = np.uint8)

            if self.showMotherDaughters and self.hasLineageData:
                motherNum = self.getMotherFromLineageData(cellNum)
                if motherNum:
                    print("selecting mother", motherNum, "for cell", cellNum)
                    if motherNum not in self.selectedCells:
                        self.selectedCells.append(motherNum)
                        self.selectedCellContourColors.append([0,255,0,255])
                        self.maskColors[motherNum,:] = np.array(self.goldColor, dtype = np.uint8)
                    else:
                        motherIdx = self.selectedCells.index(motherNum)
                        self.selectedCellContourColors[motherIdx] = [0,255,0,255]
        self.drawMask() 
        if self.pWindow and self.pWindow.hasPlots and self.labelSelect.currentText() in self.pWindow.singleCellPopulations:
            self.pWindow.updateSingleCellPlots()
    
    def getMotherFromLineageData(self,cell):
        ld = self.lineageData[self.maskZ]
        mother = ld["mother"].iloc[cell-1]
        if pd.notna(mother):
            return (int(mother))
        else:
            return None

    def deselectCell(self, num):
        print("deselecting", num)
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
        
        try:
            self.getTrainData()
        except IndexError:
            return

        weightName = self.modelChoose.currentText() 
        if weightName == "":
            self.showError("Select A Model First")
            return

        modelType = self.modelTypes[self.modelNames.index(weightName)]
        weightPath = join(MODEL_DIR, modelType, weightName)
    
        
        TW = TrainWindow(modelType, weightPath, self)
        if TW.exec_():
            data = TW.getData()
            self.train(data, modelType, weightPath)
            
        else:
            return


    def train(self, data, modelType, weightPath, autorun = True):
        # weights are overwritten if model is to be trained from scratch
        if data["scratch"]:
            weightPath = None
        # directory to save weights is current image directory stored in reader
        data["dir"] = self.imData.dirs[self.imZ]
        modelCls = self.getModelClass(modelType)
        # model is initiated with default hyperparams
        self.model = modelCls(modelCls.hyperparams, weightPath)
        self.model.train(self.trainIms, self.trainLabels, data)

        if autorun:
            self.trainTStop+=1
            im = self.getCurrMaskSet()[self.trainTStop:self.trainTStop+1,:,:]
            newIms, newProbIms = modelCls.run(im,None,weightPath)

            if self.trainTStop>self.maskData.maxT:
                addMethod = self.maskData.addToZ
            else:
                addMethod = self.maskData.insert
            
            if not self.hasProbIm():
                newProbIms = None
            addMethod(newIms, self.trainTStop, self.maskZ, newProbIms=newProbIms)
            self.drawMask()

    def hasProbIm(self):
        return self.maskData.channels[self.maskZ].shape[0]>1

    def getTrainData(self):
        ims = self.getCurrImSet()
        labels = self.getCurrMaskSet()
        
        if ims[0].shape[1:]!=labels[0].shape[1:]:
            maskShape = labels[0].shape[0,:,:]
            imShape = ims[0].shape[0,:,:]
            self.showError(f"Masks of shape {maskShape} does not match images of shape {imShape}")
            raise IndexError

        if self.imData.files[self.imZ]:
            self.trainFiles = [os.path.split(path)[1].split(".")[0] for path in self.imData.files[self.imZ]]
        else:
            self.trainFiles = [f"Index {i}" for i in range(ims.shape[0])]

        imT, maskT = ims.shape[0], labels.shape[0]
        self.trainTStop = min(imT, maskT)
        for i,label in enumerate(labels[:self.trainTStop+1,:,:]):
            if np.all(label==0):
                break
        self.trainTStop = i-1
        if self.trainTStop==-1:
            self.showErrorMessage("No Labels Have Greater than the minimum of 5 ROIs")
            raise IndexError

        self.trainIms = [im for im in ims[:self.trainTStop+1,:,:]]
        self.trainLabels = [im for im in labels[:self.trainTStop+1,:,:]]
        self.trainFiles = self.trainFiles[:self.trainTStop+1]
    
    def getModelNames(self):
        self.modelNames = []
        self.modelTypes = []
        dirs = [dir for dir in glob.glob(join(MODEL_DIR,"*")) if (os.path.isdir(dir) and "__" not in dir)]
        for dir in dirs:
            dirPath, dirName = os.path.split(dir)

            for dirFile in glob.glob(join(dir,"*")):
                dirPath2, dirName2 = os.path.split(dirFile)
                if dirName in dirName2:
                    if "." not in dirName2 or (dirFile.endswith("h5") or dirFile.endswith("hdf5")):
                        self.modelNames.append(os.path.split(dirFile)[1].split(".")[0])
                        self.modelTypes.append(dirName)
        if len(self.modelNames)==0:
            self.showError("Weights Have Not Been Downloaded. Find the weights at https://drive.google.com/file/d/1J3R4JKILkQNM0Ap-MKxqv61oAxObSjBo/view?usp=drive_link. Then run install-weights in the same directory as the downloaded zip file.")


    def segment(self, modelClass, ims, params, weightPath, modelType):
        row, col = ims[0].shape
        newImTemplate = np.zeros((len(ims), row, col))
        outputTup  = (newImTemplate, newImTemplate.copy())
        tStart, tStop = int(params["T Start"]), int(params["T Stop"])
        output = modelClass.run(ims[tStart:tStop+1],params, weightPath)

        imName = params["Channel"]

        if modelType == "artilife":
            counter = 0
            for labelName, outputTup in output.items():
                if type(outputTup[0]) == np.ndarray:
                    counter += 1
                    ims, prob = outputTup
                    for i in [0,1]:
                        outputTup[i][tStart:tStop+1]
                    self.loadMasks(outputTup, name = f"{imName}_{labelName}")
            if params["Time Series"]:
                idx = self.maskZ - counter
                self.updateCellData(idx = idx)
        else:
            for i in [0,1]:
                output[i][tStart:tStop+1]
            self.loadMasks(output, name = f"{imName}_{modelType}")
        
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
            masksTrue = self.maskData.channels[self.labelSelect.items().index(validationName)][0,:,:,:]

            toValidate = {}
            for labelName in params:
                if data[labelName]:
                    toValidate[labelName] = (self.maskData.channels[self.labelSelect.items().index(labelName)][0,:,:,:])
            
            ious = [0.50, 0.75, 0.90]
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

    def computeModels(self):
        weightName = self.modelChoose.currentText() 
        if weightName == '':
            return
        modelType = self.modelTypes[self.modelNames.index(weightName)]
        weightPath = join(MODEL_DIR, modelType, weightName)

        modelClass = self.getModelClass(modelType)

        weightPath += modelClass.prefix

        dlgCls = ArtilifeParamDialog if modelType == "artilife" else ModelParamDialog
        dlg = dlgCls(modelClass.hyperparams, modelClass.types, modelType, self)
        
        if dlg.exec():
            self.modelButton.setEnabled(False)
            self.modelButton.setStyleSheet(self.stylePressed)
            params = dlg.getData()
            channel = params["Channel"]
            channelIndex = self.channelSelect.findText(channel)
            ims = self.imData.channels[channelIndex]
            worker = Worker(self,lambda: self.segment(modelClass,ims, params, weightPath, modelType), self.modelButton)

            try:
                self.beginThread(worker)
            except MemoryError:
                error_dialog  = QErrorMessage()
                error_dialog.showMessage(f"Cannot Segment Until Other Processes are Finished")
                self.activateButton(self.modelButton)
                return
        else:
            return

    def doAdaptHist(self):
        files = self.imData.files[self.imZ]
        currName = self.channelSelect.currentText()
        newIms = im_funcs.do_adapt_hist(self.getCurrImSet())
        self.addProcessedIms(newIms, files, currName+"-adaptHist", dtype = np.float32)
    
    def doNormalizeByIm(self):
        files = self.imData.files[self.imZ]
        currName = self.channelSelect.currentText()
        newIms = [normalize_im(im) for im in self.getCurrImSet()]
        self.addProcessedIms(newIms, files, currName+"-imNormalized", dtype = np.float32)
    
    def doNormalizeBySet(self):
        files = self.imData.files[self.imZ]
        currName = self.channelSelect.currentText()
        newIms = normalize_im(self.getCurrImSet())
        self.addProcessedIms(newIms, files, currName+"-setNormalized", dtype = np.float32)

    def doGaussian(self):
        files = self.imData.files[self.imZ]
        currName = self.channelSelect.currentText()
        newIms = im_funcs.do_gaussian(self.getCurrImSet())
        self.addProcessedIms(newIms, files, currName+"-gaussian")

    def doMedian(self):
        files = self.imData.files[self.imZ]
        currName = self.channelSelect.currentText()
        newIms = im_funcs.do_median(self.getCurrImSet())
        self.addProcessedIms(newIms, files, currName+"-median")
    
    def doDeflicker(self):
        files = self.imData.files[self.imZ]
        currName = self.channelSelect.currentText()
        newIms = im_funcs.deflicker(self.getCurrImSet())
        self.addProcessedIms(newIms, files, currName+"-deflickered")
    
    def doRescale(self):
        files = self.imData.files[self.imZ]
        currName = self.channelSelect.currentText()
        ims  = self.getCurrImSet()
        newIms = self.rescaleFromUser(ims)
        newSize = str(newIms[0].shape)
        self.addProcessedIms(newIms, files, currName+f"-{newSize}")

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

    def addProcessedIms(self, newIms, files, name, dtype = np.uint16):
        head, _ = os.path.split(files[0])
        self.imData.dirs.append(head)
        self.imData.files.append(files)
        self.loadImages(np.array(newIms, dtype = dtype), name = name)

    def runImageAction(self, imFunc):
        if not self.imLoaded:
            return
        self.toStatusBar("Running Preprocess Function...")
        worker = Worker(self, imFunc)
        self.beginThread(worker)
        
    def activateButton(self, button):
        button.setEnabled(True)
        button.setStyleSheet(self.styleUnpressed)
    
    def getCurrImDir(self):
        return self.imData.dirs[self.imZ]
    
    def saveIms(self):
        if not self.imLoaded:
            self.showError("No Images Loaded")
            return
        defaultFileName = join(self.getCurrImDir(), self.channelSelect.currentText().replace(" ", "") + ".tif")
        path, _ = QFileDialog.getSaveFileName(self, 
                            "save image as greyscale stack tif", 
                            defaultFileName)
        if path:
            imsave(path, self.getCurrImSet().astype(np.uint16))
    

    def saveMasks(self):
        if not self.maskLoaded:
            self.showError("No masks to save")
        defaultFileName = join(self.getCurrImDir(), self.labelSelect.currentText().replace(" ", "") + ".tif")
        path, _ = QFileDialog.getSaveFileName(self, 
                            "save mask as greyscale stack tif (uint16)", 
                            defaultFileName)
        if path:
            imsave(path, self.getCurrMaskSet().astype(np.uint16))
    
    def saveFigure(self):
        if (not self.imLoaded):
            self.showError("No Images Loaded")
            return

        defaultFileName = join(self.getCurrImDir(), self.labelSelect.currentText().replace(" ", "") + "-figure.tif")
        path, _ = QFileDialog.getSaveFileName(self, 
                            "save figure as stack tif (uint16)", 
                            defaultFileName)
        if path:
            imsave(path, self.createFigure())

    def createFigure(self):
        ims = self.getCurrImSet()
        ims = np.array([convertGreyToRGBA(im) for im in ims], dtype = np.uint8)
        masks = []
        currT = self.tIndex
        for i in range(self.imData.maxT+1):
            self.tIndex = i
            self.updateDisplay()
            masks.append(self.pg_mask.image[:,:,0:3])
        self.tIndex = currT
        self.updateDisplay()
        masks = np.array(masks, dtype = np.uint8)
        np.putmask(ims, masks>0, masks)
        return ims
    
    def saveCellData(self):
        if not self.maskLoaded:
            self.showError("No Label Data Loaded")
            return
        
        labelName = self.labelSelect.items()[self.maskZ]
        labelFileName = labelName.replace(" ", "")
        if self.hasCellData():
            data = exportCellData(self.cellData[self.maskZ])
            defaultFileName = join(self.imData.dirs[self.imZ], labelFileName + "_celldata.csv")
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
        motherdata, daughterdata = self.lineageData[self.maskZ], self.mother_daughters[self.maskZ]
        if self.hasLineageData():
            fname1= join(self.imData.dirs[self.imZ], labelFileName + "_mother_daughters.csv")
            fname2 = join(self.imData.dirs[self.imZ], labelFileName + "_daughter_array.csv.csv")
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
        if type(self.cellData[self.maskZ]) is pd.core.frame.DataFrame:
            data = getHeatMaps(self.cellData[self.maskZ])
            defaultFileName = join(self.imData.dirs[self.imZ], labelFileName + "_heatmaps.tif")
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
    
    def getCurrImDir(self):
        return self.imData.dirs[self.imZ]
    
    def saveData(self):
        if self.imLoaded:
        
            if self.overrideNpyPath:
                path = self.overrideNpyPath
            else:
                dir = self.imData.dirs[0]
                name = self.sessionId
                path = join(dir,name+".pkl")

            imdata = {"Images": self.imData.channels,
                        "Saturation": self.imData.saturation,
                        "Dirs": self.imData.dirs,
                        "Files": self.imData.files}
            maskdata = {"Masks": self.maskData.channels,
                        "Contours":self.maskData.contours}
            if self.maskData.isDummy:
                maskdata["Masks"] = ["Blank"]
            otherdata = {"Channels": self.channelSelect.items(),
                        "Labels": self.labelSelect.items()}
            celldata = {"Cells": self.cellData,
                        "Lineages": self.lineageData,
                        "Daughters": self.mother_daughters}
            i = 0
            for item in self.cellData:
                if type(item) is pd.core.frame.DataFrame or type(item) is np.ndarray:
                    i+=1
            print("Saving ", i, "valid dataframes")
            data =  imdata | maskdata | otherdata | celldata

            try:
                with open(path, "wb") as f:
                    pickle.dump(data, f)
            except:
                print("ERROR: cannot sive archive to", path)
    
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
        self.showMotherDaughtersButton.setEnabled(hasLineages)
        self.showLineageButton.setEnabled(hasLineages)

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




        