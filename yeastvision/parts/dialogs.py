from PyQt5 import QtGui, QtCore

import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QTreeWidgetItem, QPushButton, QDialog,
                            QDialogButtonBox, QLineEdit, QFormLayout, QCheckBox,  QSpinBox, QLabel, 
                            QWidget, QComboBox, QGridLayout, QHBoxLayout, QSizePolicy, QHeaderView, QVBoxLayout,
                            QScrollArea, QTreeWidget)
from PyQt5.QtCore import Qt
import numpy as np
import os
import datetime
from yeastvision.parts.guiparts import *
from yeastvision.plot.plot import PlotProperty

class DirReaderDialog(QDialog):
    def __init__(self, dir, fnames):
        super().__init__()

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.formLayout = QFormLayout()
        self.formLayout.addRow(QLabel(f"Load Custom Channels\n{dir}"))
        self.setLayout(self.formLayout)


        # Create the QSpinBox
        self.spinBox = QSpinBox(self)
        self.spinBox.setMinimumWidth(100)
        self.spinBox.setMinimum(0) # Set minimum value
        self.spinBox.valueChanged.connect(self.update_line_edits) # Connect signal to slot
        self.formLayout.addRow("number of channels:", self.spinBox)
        self.lineEditLayout = QFormLayout()
        self.formLayout.addRow(self.lineEditLayout)
        self.formLayout.addWidget(self.buttonBox)

        # Initialize list to hold QLineEdits
        self.line_edits = []
    

    def update_line_edits(self, value):
        # Check if the value has increased or decreased
        if value > len(self.line_edits):
            # Value increased, create new QLineEdit(s)
            for i in range(len(self.line_edits), value):
                line_edit = QLineEdit(self)
                line_edit.setToolTip("a sequence of characters only found in this channels filename")
                self.line_edits.append(line_edit)
                self.lineEditLayout.addRow(f"channel {i} identifier", line_edit)
        elif value < len(self.line_edits):
            # Value decreased, remove QLineEdit(s)
            for i in range(len(self.line_edits)-1, value-1, -1):
                self.lineEditLayout.removeRow(self.line_edits[i]) # Remove QLineEdit from layout
                self.line_edits[i].deleteLater() # Delete QLineEdit from memory
                self.line_edits.pop(i) # Remove QLineEdit from list
    
    def getData(self):
        return [line_edit.text() for line_edit in self.line_edits]
    
class GeneralParamDialog(QDialog):
    def __init__(self, hyperparamDict, paramtypes, windowName, parent, labelSelects = None, channelSelects = None):
        
        super().__init__(parent)
        self.parent = parent
        self.params = hyperparamDict
        self.types = paramtypes

        self.dropDowns = {}
        self.checkBoxes = {}
        self.lineEdits = {}

        if self.params:
            self.splitHyperParams()

        self.labelSelects = labelSelects
        self.channelSelects = channelSelects

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.message = QLabel(f"Enter parameters for {windowName}")

        self.formLayout = QFormLayout()
        self.formLayout.addRow(self.message)
        self.populateFormLayout()
        self.formLayout.addWidget(self.buttonBox)
        self.setLayout(self.formLayout)
        self.errorAlrShown = False
    
    def splitHyperParams(self):
        for paramName, paramType in zip(list(self.params.keys()),self.types) :
            if paramType == "drop":
                self.dropDowns[paramName] = ""
            elif paramType is bool:
                self.checkBoxes[paramName] = self.params[paramName]
            elif paramType is None:
                self.lineEdits[paramName] = self.params[paramName]
    
    def populateFormLayout(self):
        self.dropDownData  = {}
        if self.channelSelects:
            for channelSelectName in self.channelSelects:
                self.dropDownData[channelSelectName] = self.addParentComboBox(self.parent.channelSelect, name = channelSelectName)
        if self.labelSelects:
            for labelSelectName in self.labelSelects:
                self.dropDownData[labelSelectName] = self.addParentComboBox(self.parent.labelSelect, name = labelSelectName)
        
        # check boxes
        hLayout = QHBoxLayout()
        self.checkBoxData = {}
        if self.checkBoxes:
            for checkName, defaultVal in self.checkBoxes.items():
                checkBox = QPushButton(checkName)
                checkBox.setCheckable(True)
                hLayout.addWidget(checkBox)
                checkBox.setChecked(defaultVal)
                self.checkBoxData[checkName] = checkBox
            self.formLayout.addRow(hLayout)

        formDict = self.lineEdits
        self.lineEditData = {}
        for labelName,defaultVal in formDict.items():
            lineEdit  = QLineEdit(self)
            lineEdit.setFixedWidth(120)
            lineEdit.setText(str(defaultVal))
            self.lineEditData[labelName] = lineEdit
            self.formLayout.addRow(self.produceLabel(labelName), lineEdit)
    
    def addParentComboBox(self, parentComboBox, name = "Channel", addNA = False):
        channelSelect = QComboBox()
        channelSelect.setFixedWidth(120)
        channelSelect.addItems([parentComboBox.itemText(i) for i in range(parentComboBox.count())])
        channelSelect.setCurrentIndex(parentComboBox.currentIndex())
        self.formLayout.addRow(self.produceLabel(name), channelSelect)
        return channelSelect

    def produceLabel(self, text):
        label = QLabel(text)
        return label
    
    def getData(self):
        dropDownData = {k:v.currentText() for k,v in self.dropDownData.items()}
        lineEditData = {k:float(v.text()) for k,v in self.lineEditData.items()}
        boolData = {k:v.isChecked() for k,v in self.checkBoxData.items()}
        return dropDownData | lineEditData | boolData



class ModelParamDialog(GeneralParamDialog):
    def __init__(self, hyperparamDict, paramtypes, modelName, parent = None):
        super().__init__(hyperparamDict, paramtypes, modelName, parent)

    def populateFormLayout(self):
        self.dropDownData  = {}
        self.dropDownData["Channel"] = self.addParentComboBox(self.parent.channelSelect, name = "Channel to Segment")
        # order is important here as self.dropDownData["T Start"]/["T Stop"]
            # must exist before addTValues and indexChange can be called
        self.addTDropDown()
        self.dropDownData["Channel"].currentIndexChanged.connect(self.channelSelectIndexChanged)
        self.channelSelectIndexChanged()
        
        if self.dropDowns:
            for dropDownName in self.dropDowns.keys():
                self.dropDownData[dropDownName] = self.addParentComboBox(self.parent.channelSelect, name = dropDownName)
        
        # check boxes
        hLayout = QHBoxLayout()
        self.checkBoxData = {}
        if self.checkBoxes:
            for checkName, defaultVal in self.checkBoxes.items():
                checkBox = QPushButton(checkName)
                checkBox.setCheckable(True)
                hLayout.addWidget(checkBox)
                checkBox.setChecked(defaultVal)
                self.checkBoxData[checkName] = checkBox
            self.formLayout.addRow(hLayout)

        formDict = self.lineEdits
        self.lineEditData = {}
        for labelName,defaultVal in formDict.items():
            lineEdit  = QLineEdit(self)
            lineEdit.setFixedWidth(120)
            lineEdit.setText(str(defaultVal))
            self.lineEditData[labelName] = lineEdit
            self.formLayout.addRow(self.produceLabel(labelName), lineEdit)

    def channelSelectIndexChanged(self):
        channelIndex = self.parent.channelSelect.findText(self.dropDownData["Channel"].currentText())
        maxT = self.parent.imData.maxTs[channelIndex]
        self.dropDownData["T Start"].clear()
        self.dropDownData["T Stop"].clear()
        self.addTValues(maxT)

    def addTimeDropDown(self, name):
        timeSelect = QComboBox()
        timeSelect.setFixedWidth(120)
        self.formLayout.addRow(self.produceLabel(name), timeSelect)
    
    def addTDropDown(self):
        hbox = QHBoxLayout()
        timeSelectStart, timeSelectStop = QComboBox(), QComboBox()
        timeSelectStart.setFixedWidth(120)
        timeSelectStart.setToolTip("Start time point")
        timeSelectStop.setFixedWidth(120)
        timeSelectStop.setToolTip("Stop Time Point (Inclusive)")
        hbox.addWidget(timeSelectStart)
        hbox.addWidget(timeSelectStop)
        self.dropDownData["T Start"] = timeSelectStart
        self.dropDownData["T Stop"] = timeSelectStop
        self.formLayout.addRow(QLabel("Time Start/Stop"), hbox)
    
    def addTValues(self, maxT, prefix = "T "):
        validTs = list(range(maxT+1))
        validTs = [str(t) for t in validTs]
        self.dropDownData[f"{prefix}Start"].addItems(validTs)
        self.dropDownData[f"{prefix}Start"].setCurrentIndex(0)
        self.dropDownData[f"{prefix}Stop"].addItems(validTs)
        self.dropDownData[f"{prefix}Stop"].setCurrentIndex(len(validTs)-1)
    
    def showErrorMessage(self):
        if not self.errorAlrShown:
            self.message.setText(self.message.text() + "\nEnsure T Start<T Stop and All Line Entries are Numeric")
            self.errorAlrShown = True

    def accept(self):
        if self.validateLineData() and self.validateTData():
            super().accept()
        else:
            self.showErrorMessage()
    
    def validateTData(self):
        data = self.dropDownData
        valid = int(data["T Start"].currentText())<=int(data["T Stop"].currentText())
        return valid

    def validateLineData(self):
        data = {k:v.text() for k,v in self.lineEditData.items()}
        i = 0
        vals = list(data.values())
        valid = True
        while i<len(vals) and valid:
            valid = self.isNumber(vals[i]) and float(vals[i])>=0.0
            i+=1
        return valid
    
    def isNumber(self, num):
        """ Returns True is string is a number. """
        # must remove periods to ensure decimals result in True
        return num.replace(".","").isdigit()


class FlouParamDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.message = QLabel(f"Parameters for blob detection")

        self.formLayout = QFormLayout()
        self.formLayout.addRow(self.message)
        self.populateFormLayout()
        self.formLayout.addWidget(self.buttonBox)
        self.setLayout(self.formLayout)
        self.errorAlrShown = False
    
    def populateFormLayout(self):
        self.checkBoxData = {}
        useExampleCheck = QPushButton("extract params from current mask")
        useExampleCheck.setEnabled(np.any(self.parent.getCurrMaskSet() > 0))
        useExampleCheck.setCheckable(True)
        useExampleCheck.setChecked(False)
        useExampleCheck.clicked.connect(self.exampleCheckClicked)
        self.checkBoxData["extract from curr mask"] = useExampleCheck
        self.formLayout.addWidget(useExampleCheck)

        self.dropDownData = {}
        self.dropDownData["channel"] = self.addParentComboBox(self.parent.channelSelect, name = "Channel to Segment")
        self.dropDownData["label"] = self.addParentComboBox(self.parent.labelSelect, name = "Filter Masks")
        self.dropDownData["label"].addItem("")
        self.dropDownData["label"].setCurrentIndex(-1)
        self.dropDownData["label"].setToolTip("Anything segmented outside 'Filter Mask' will be set to 0")
        contourCheck = QPushButton("use contours")
        contourCheck.setEnabled(np.any(self.parent.getCurrMaskSet() > 0))
        contourCheck.setCheckable(True)
        if contourCheck.isCheckable:
            contourCheck.setChecked(self.parent.contourOn)
        self.formLayout.addWidget(contourCheck)
        self.checkBoxData["use contours"] = contourCheck

        self.addTDropDown()
        self.dropDownData["channel"].currentIndexChanged.connect(self.dIndexChanged)
        self.dropDownData["label"].currentIndexChanged.connect(self.dIndexChanged)
        self.dIndexChanged()

        self.lineEditData = {}
        for lineEditName, defaultVal in zip(["min area", "max area", "threshold"], [20,50,0.08]):
            lineEdit = QLineEdit(self)
            lineEdit.setFixedWidth(120)
            lineEdit.setText(str(defaultVal))
            self.lineEditData[lineEditName] = lineEdit
            self.formLayout.addRow(self.produceLabel(lineEditName), lineEdit)

    def exampleCheckClicked(self):
        enabled = not self.checkBoxData["extract from curr mask"].isChecked()
        for checkBox in self.lineEditData.values():
            checkBox.setEnabled(enabled)

    def dIndexChanged(self):
        channelIndex = self.parent.channelSelect.findText(self.dropDownData["channel"].currentText())
        labelIndex = self.parent.labelSelect.findText(self.dropDownData["label"].currentText())

        if labelIndex == -1:
            maxT = self.parent.imData.maxTs[channelIndex]
        else:
            maxT = min(self.parent.imData.maxTs[channelIndex], self.parent.maskData.maxTs[labelIndex])
        
        self.dropDownData["T Start"].clear()
        self.dropDownData["T Stop"].clear()
        self.addTValues(maxT)

    def addParentComboBox(self, parentComboBox, name = "Channel", addNA = False):
        channelSelect = QComboBox()
        channelSelect.setFixedWidth(120)
        channelSelect.addItems([parentComboBox.itemText(i) for i in range(parentComboBox.count())])
        channelSelect.setCurrentIndex(parentComboBox.currentIndex())
        self.formLayout.addRow(self.produceLabel(name), channelSelect)
        return channelSelect

    def produceLabel(self, text):
        label = QLabel(text)
        return label
    
    def getData(self):
        dropDownData = {k:v.currentText() for k,v in self.dropDownData.items()}
        lineEditData = {k:float(v.text()) for k,v in self.lineEditData.items()}
        boolData = {k:v.isChecked() for k,v in self.checkBoxData.items()}
        return dropDownData | lineEditData | boolData

    def addTimeDropDown(self, name):
        timeSelect = QComboBox()
        timeSelect.setFixedWidth(120)
        self.formLayout.addRow(self.produceLabel(name), timeSelect)
    
    def addTDropDown(self):
        hbox = QHBoxLayout()
        timeSelectStart, timeSelectStop = QComboBox(), QComboBox()
        timeSelectStart.setFixedWidth(120)
        timeSelectStart.setToolTip("Start time point")
        timeSelectStop.setFixedWidth(120)
        timeSelectStop.setToolTip("Stop Time Point (Inclusive)")
        hbox.addWidget(timeSelectStart)
        hbox.addWidget(timeSelectStop)
        self.dropDownData["T Start"] = timeSelectStart
        self.dropDownData["T Stop"] = timeSelectStop
        self.formLayout.addRow(QLabel("Time Start/Stop"), hbox)
    
    def addTValues(self, maxT, prefix = "T "):
        validTs = list(range(maxT+1))
        validTs = [str(t) for t in validTs]
        self.dropDownData[f"{prefix}Start"].addItems(validTs)
        self.dropDownData[f"{prefix}Start"].setCurrentIndex(0)
        self.dropDownData[f"{prefix}Stop"].addItems(validTs)
        self.dropDownData[f"{prefix}Stop"].setCurrentIndex(len(validTs)-1)

    def showErrorMessage(self):
        if not self.errorAlrShown:
            self.message.setText(self.message.text() + "\nEnsure T Start<T Stop and All Line Entries are Numeric")
            self.errorAlrShown = True

    def accept(self):
        if self.validateLineData() and self.validateTData():
            super().accept()
        else:
            self.showErrorMessage()
    
    def validateTData(self):
        data = self.dropDownData
        valid = int(data["T Start"].currentText())<=int(data["T Stop"].currentText())
        return valid

    def validateLineData(self):
        data = {k:v.text() for k,v in self.lineEditData.items()}
        i = 0
        vals = list(data.values())
        valid = True
        while i<len(vals) and valid:
            valid = self.isNumber(vals[i]) and float(vals[i])>=0.0
            i+=1
        return valid
    
    def isNumber(self, num):
        """ Returns True is string is a number. """
        # must remove periods to ensure decimals result in True
        return num.replace(".","").isdigit()

class ArtilifeParamDialog(ModelParamDialog):
    def __init__(self, params, paramtypes, name, parent):
        super().__init__(params,paramtypes,"artilife", parent)
    
    def populateFormLayout(self):
        self.dropDownData  = {}
        self.dropDownData["Channel"] =  self.addParentComboBox(self.parent.channelSelect, name = "Channel to Segment")
        # order is important here as self.dropDownData["T Start"]/["T Stop"]
            # must exist before addTValues and indexChange can be called
        self.addTDropDown()
        self.dropDownData["Channel"].currentIndexChanged.connect(self.channelSelectIndexChanged)
        if self.dropDowns:
            for dropDownName in self.dropDowns.keys():
                self.dropDownData[dropDownName] = self.addParentComboBox(self.parent.channelSelect, name = dropDownName, addNA = True)


        formDict = self.lineEdits
        self.lineEditData = {}
        for labelName,defaultVal in formDict.items():
            lineEdit  = QLineEdit(self)
            lineEdit.setFixedWidth(120)
            lineEdit.setText(str(defaultVal))
            self.lineEditData[labelName] = lineEdit
            self.formLayout.addRow(self.produceLabel(labelName), lineEdit)
        
        self.addArtilifeTDropDowns(name = "mat")
        self.addArtilifeTDropDowns(name = "spore")
        self.channelSelectIndexChanged()
    
        # check boxes
        hLayout = QHBoxLayout()
        self.checkBoxData = {}
        if self.checkBoxes:
            for checkName, defaultVal in self.checkBoxes.items():
                checkBox = QPushButton(checkName)
                checkBox.setCheckable(True)
                hLayout.addWidget(checkBox)
                checkBox.setChecked(defaultVal)
                self.checkBoxData[checkName] = checkBox
            self.formLayout.addRow(hLayout)
        
        self.connectSporeMatButtons()

    
    def connectSporeMatButtons(self):
        self.checkBoxData["Mating Cells"].toggled.connect(self.toggleMat)
        self.checkBoxData["Sporulating Cells"].toggled.connect(self.toggleSpore)
    
    def toggleMat(self):
        enabled = self.checkBoxData["Mating Cells"].isChecked()
        self.dropDownData["matStart"].setEnabled(enabled)
        self.dropDownData["matStop"].setEnabled(enabled)
        self.dropDownData["matSeg"].setEnabled(enabled)
    def toggleSpore(self):
        enabled = self.checkBoxData["Sporulating Cells"].isChecked()
        self.dropDownData["sporeStart"].setEnabled(enabled)
        self.dropDownData["sporeStop"].setEnabled(enabled)
        self.dropDownData["tetradSeg"].setEnabled(enabled)

    def addArtilifeTDropDowns(self, name):
        hbox = QHBoxLayout()
        timeSelectStart, timeSelectStop = QComboBox(), QComboBox()
        timeSelectStart.setEnabled(False)
        timeSelectStop.setEnabled(False)
        timeSelectStart.setFixedWidth(120)
        timeSelectStart.setToolTip("Start time point")
        timeSelectStop.setFixedWidth(120)
        timeSelectStop.setToolTip("Stop Time Point (Inclusive)")
        hbox.addWidget(timeSelectStart)
        hbox.addWidget(timeSelectStop)
        self.dropDownData[f"{name}Start"] = timeSelectStart
        self.dropDownData[f"{name}Stop"] = timeSelectStop
        longName = "Mating" if name == "mat" else "Sporulating"
        self.formLayout.addRow(QLabel(f"{longName} Start/Stop"), hbox)
        
        # add dropdown to allow for 
        modelType = "matSeg" if name == "mat" else "tetradSeg"
        modeloptions = [model for model in self.parent.modelNames if modelType in model]
        weights = QComboBox()
        weights.setEnabled(False)
        weights.setFixedWidth(120)
        weights.setToolTip(f"Weights for {modelType}")
        weights.addItems(modeloptions)
        weights.setCurrentIndex(0)
        self.dropDownData[modelType] = weights
        if len(modeloptions)>1:
            self.formLayout.addRow(QLabel(f"{modelType} Weights"), weights)


    def channelSelectIndexChanged(self):
        channelIndex = self.parent.channelSelect.findText(self.dropDownData["Channel"].currentText())
        maxT = self.parent.imData.maxTs[channelIndex]
        for prefix in ["T ", "mat", "spore"]:
            self.dropDownData[f"{prefix}Start"].clear()
            self.dropDownData[f"{prefix}Stop"].clear()
            self.addTValues(maxT, prefix = prefix)

class TrainWindow(QDialog):
    def __init__(self, modelType, weightPath, parent):
        super().__init__(parent)
        self.setGeometry(100,100,900,350)
        self.setWindowTitle("Train Settings")
        
        self.parent = parent
        self.modelType = modelType
        self.weightPath = weightPath
        self.modelName = self.getModelName(self.weightPath)

        self.win = QWidget(self)
        self.l0 = QGridLayout()
        self.win.setLayout(self.l0)

        self.trainingParamVals, self.lossName = self.getTrainingParams()

        self.yoff = 0
        qlabel = QLabel('Train Model w/ Images')
        qlabel.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))
        qlabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.l0.addWidget(qlabel, self.yoff,0,1,2)

        # choose initial model
        self.yoff+=1
        self.modelChoose = QComboBox()
        #self.modelChoose.addItems(modelNames)
        self.modelChoose.addItems([self.modelName], ) 
        self.modelChoose.setFixedWidth(150)
        #self.modelChoose.setCurrentIndex(self.parent.modelChoose.currentIndex())
        self.l0.addWidget(self.modelChoose, self.yoff, 1,1,1)
        qlabel = QLabel('initial model: ')
        qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.l0.addWidget(qlabel, self.yoff,0,1,1)

        # add check box for from scratch training
        self.yoff+=1
        qlabel = QLabel(f"From Scratch")
        qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.checkbox = QCheckBox()
        self.l0.addWidget(qlabel, self.yoff,0,1,1)
        self.l0.addWidget(self.checkbox,self.yoff,1,1,1)

        # add loss function
        self.yoff+=1
        lossDisplay = QComboBox()
        lossDisplay.addItem(self.lossName)
        lossDisplay.setFixedWidth(150)
        self.l0.addWidget(lossDisplay, self.yoff,1,1,1)
        qlabel = QLabel("Loss Function: ")
        qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.l0.addWidget(qlabel, self.yoff,0,1,1)

        # # select channels for training
        # imSelect = self.addImSelect("Images", self.parent.channelSelect)
        # labelSelect = self.addImSelect("Labels", self.parent.labelSelect)
        # for cSelect in [imSelect, labelSelect]:
        #     self.yoff+=1
        #     cSelect.setFixedWidth(150)
        #     self.l0.addWidget(cSelect.labelObj, self.yoff, 0,1,1)
        #     self.l0.addWidget(cSelect, self.yoff, 1,1,1)

        self.yoff+=1
        qlabel = QLabel("Model Suffix")
        qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.l0.addWidget(qlabel, self.yoff,0,1,1)
        d =  datetime.datetime.now()
        d =  d.strftime("%Y_%m_%d_%H_%M_%S.%f")
        defaultSuffix = f"{d}"
        self.modelName = QLineEdit()
        self.modelName.setText(str(defaultSuffix))
        self.modelName.setFixedWidth(200)
        self.l0.addWidget(self.modelName, self.yoff, 1,1,1)

        self.addTrainParams()
        
        self.addImNames()

        # click button
        self.yoff+=3
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.l0.addWidget(self.buttonBox, self.yoff, 0, 1,4)
        
    
    def accept(self):
        if self.validateTrainingParamVals():
            super().accept()
        else:
            self.showErrorMessage()
    
    def validateTrainingParamVals(self):
        data = self.edits.values()
        for entry in data:
            if not self.isNumber(entry.text()):
                return False
        return True
    
    def getData(self):
        suffix = str(self.modelName.text())
        return {k:float(v.text()) for k,v in self.edits.items()} | {"scratch": self.checkbox.isChecked()} | {"model_name": f"{self.modelType}_{suffix}"}

    
    def isNumber(self, num):
        """ Returns True is stsring is a number. """
        # must remove periods to ensure decimals result in True
        return num.replace(".","").isdigit()

    def getModelName(self, weights):
        return os.path.split(weights)[-1].split(".")[0]

    def addImSelect(self, name, parentComboBox):
        parentComboBox = self.parent.channelSelect
        channelSelect = QComboBox()
        channelSelect.setFont(self.parent.medfont)
        channelSelect.setFixedWidth(120)
        channelSelect.addItems([parentComboBox.itemText(i) for i in range(parentComboBox.count())])
        channelSelect.setCurrentIndex(parentComboBox.currentIndex())
        channelSelect.labelObj = QLabel(name)
        return channelSelect
    
    def getTrainingParams(self):
        modelClass =  self.parent.getModelClass(self.modelName)
        modelParams = dict(modelClass.trainparams)
        loss = modelClass.loss
        return dict(modelParams), loss
    
    def addTrainParams(self):
        # choose parameters        
        labels = list(self.trainingParamVals.keys())
        defaultVals = list(self.trainingParamVals.values())
        self.edits = {}
        self.yoff += 1
        for i, label in enumerate(labels):
            qlabel = QLabel(label)
            qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.l0.addWidget(qlabel, i+self.yoff,0,1,1)
            newEdit = (QLineEdit())
            newEdit.setText(str(defaultVals[i]))
            newEdit.setFixedWidth(200)
            self.edits[label] = newEdit
            self.l0.addWidget(newEdit, i+self.yoff, 1,1,1)
    
    def addImNames(self):
         # list files in folder
        qlabel = QLabel('ims')
        qlabel.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        self.l0.addWidget(qlabel, 0,4,1,1)
        qlabel = QLabel('# of masks')
        qlabel.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        self.l0.addWidget(qlabel, 0,5,1,1)
        
        for i in range(10):
            if i > len(self.parent.trainFiles) - 1:
                break
            elif i==9 and len(self.parent.trainFiles) > 10:
                label = '...'
                nmasks = '...'
            else:
                label = os.path.split(self.parent.trainFiles[i])[-1]
                nmasks = str(self.parent.trainLabels[i].max())
            qlabel = QLabel(label)
            self.l0.addWidget(qlabel, i+1,4,1,1)
            qlabel = QLabel(nmasks)
            qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.l0.addWidget(qlabel, i+1, 5,1,1)



class PlotWindowCustomize(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setGeometry(100,100,1000,450)
        self.setWindowTitle("Customize Plot Display")
        self.win = QWidget(self)
        self.l = QGridLayout()
        self.win.setLayout(self.l)

        self.morphProps = self.parent.getCellDataLabelProps()
        self.trees = []
        self.boxes = {}
        self.produceParamTreeSingleCell()
        self.produceParamTreeIncludePopulations()

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.l.addWidget(self.buttonBox, 3, 0, 1, 2)

        self.win.show()
    
    def getProps(self, populationObj, propType):
        if propType == "morphology":
            return self.morphProps
        else:
            return [prop for prop in populationObj.properties if prop not in self.morphProps]

    def produceParamTreeSingleCell(self):
        populations = self.parent.getLabelsWithPopulationData()      
        tree = self.produceTree()
        self.trees.append(tree)
        level1 = "single"
        item = QTreeWidgetItem(tree, [level1])
        item.setExpanded(True)

        for level2 in populations:
            setName = self.parent.getTimeSeriesDataName(level2)
            item2 = QTreeWidgetItem(item, [setName])
            item2.setExpanded(True)

            for level3 in ["morphology", "pixel"]:
                item3 = QTreeWidgetItem(item2, [level3])

                for level4 in self.getProps(level2, level3):
                    item4 = QTreeWidgetItem(item3, [level4])
                    box = QCheckBox()
                    tree.setItemWidget(item4, 1, box)
                    self.boxes[PlotProperty(self.parent, "single", setName, level4)] = box
        self.l.addWidget(tree, 0,0,2,1)

    def produceParamTreeIncludePopulations(self):
        populations = self.parent.getLabelsWithPopulationData()      
        for i,level1 in enumerate(self.parent.populationPlotTypes):
            tree = self.produceTree()
            self.trees.append(tree)
            item = QTreeWidgetItem(tree, [level1])
            item.setExpanded(True)

            for level2 in populations:
                setName = self.parent.getTimeSeriesDataName(level2)
                item2 = QTreeWidgetItem(item, [setName])
                item2.setExpanded(True)

                for level2a in level2.population_names:
                    item2a = QTreeWidgetItem(item2, [level2a])

                    for level3 in ["morphology", "pixel"]:
                        item3 = QTreeWidgetItem(item2a, [level3])

                        for level4 in self.getProps(level2, level3):
                            item4 = QTreeWidgetItem(item3, [level4])
                            box = QCheckBox()
                            tree.setItemWidget(item4, 1, box)
                            self.boxes[PlotProperty(self.parent, level1, setName, level4, populationName=level2a)] = box
            self.l.addWidget(tree, 0,i+1,2,1)

    def getData(self):
        checked = [name for name, box in self.boxes.items() if box.isChecked()]
        toPlotDict = {}
        for plot in self.parent.plotTypes:
            toPlotDict[plot] = [prop_obj for prop_obj in checked if plot in prop_obj.plotType]
        return toPlotDict

    
    def produceTree(self):
        tree = QTreeWidget()
        tree.setColumnCount(2)
        tree.setHeaderHidden(True)
        tree.setMinimumHeight(300)
        tree.setMinimumWidth(300)
        tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        return tree





