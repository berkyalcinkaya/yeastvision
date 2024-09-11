from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import (QTreeWidgetItem, QPushButton, QDialog,
                        QDialogButtonBox, QLineEdit, QFormLayout, QCheckBox,  QSpinBox, QDoubleSpinBox, QLabel, 
                            QWidget, QComboBox, QGridLayout, QHBoxLayout, QHeaderView, QVBoxLayout, QMessageBox,
                            QTreeWidget, QButtonGroup, QFrame, QRadioButton)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
import numpy as np
import os
from yeastvision.parts.guiparts import *
from yeastvision.plot.plot import PlotProperty
import math
from yeastvision.data.ims import InterpolatedChannel

class ChoiceDialog(QDialog):
    def __init__(self, choices, title, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle(title)
        
        self.layout = QVBoxLayout()
        self.button_group = QButtonGroup(self)
        
        # Create radio buttons for each choice
        for choice in choices:
            radio_button = QRadioButton(choice)
            self.layout.addWidget(radio_button)
            self.button_group.addButton(radio_button)
        
        # Add OK and Cancel buttons
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        self.layout.addWidget(self.ok_button)
        self.layout.addWidget(self.cancel_button)
        
        self.setLayout(self.layout)
        
    def get_choice(self):
        # Returns the selected choice, or None if no selection is made
        selected_button = self.button_group.checkedButton()
        if selected_button:
            return selected_button.text()
        return None

class ComboBoxDialog(QDialog):
    def __init__(self, choices, label, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setLayout(QVBoxLayout())

        self.label = QLabel(label)
        self.layout().addWidget(self.label)

        self.comboBox = QComboBox()
        self.comboBox.addItems(choices)
        self.layout().addWidget(self.comboBox)

        self.okButton = QPushButton("OK")
        self.okButton.clicked.connect(self.accept)
        self.layout().addWidget(self.okButton)

    def getSelection(self):
        return self.comboBox.currentText()

class SimpleTextDialog(QDialog):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Information")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        # Text label
        text_label = QLabel(text)
        layout.addWidget(text_label)

        # Ok and Cancel buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

class IntervalSelectionDialog(QDialog):
    def __init__(self, intervals, maxT, windowTitle, parent=None, labels = None, presetT1 = 0, presetT2 = None):
        super().__init__(parent)
        self.intervals = intervals
        self.maxT = maxT

        self.setWindowTitle(windowTitle)
        #self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        if labels:
            if isinstance(labels, list):
                for label in labels:
                    layout.addWidget(QLabel(label))
            else:
                layout.addWidget(QLabel(labels))

        # Spin boxes for start and end indices
        spin_layout = QHBoxLayout()
        self.start_spinbox = QSpinBox()
        self.start_spinbox.setRange(0, maxT)
        self.start_spinbox.setValue(presetT1)
        self.start_spinbox.valueChanged.connect(self._update_end_spinbox_range)

        self.end_spinbox = QSpinBox()
        self.end_spinbox.setRange(0, maxT + 1)
        if presetT2 is None:
            presetT2 = presetT1
        self.end_spinbox.setValue(presetT2)
        self.start_spinbox.valueChanged.connect(self._sync_end_spinbox)

        spin_layout.addWidget(QLabel("Start:"))
        spin_layout.addWidget(self.start_spinbox)
        spin_layout.addWidget(QLabel("End:"))
        spin_layout.addWidget(self.end_spinbox)

        layout.addLayout(spin_layout)

        # Ok and Cancel buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _update_end_spinbox_range(self):
        self.end_spinbox.setMinimum(self.start_spinbox.value() + 1)

    def _sync_end_spinbox(self):
        if self.end_spinbox.value() < self.start_spinbox.value():
            self.end_spinbox.setValue(self.start_spinbox.value() + 1)

    def get_selected_interval(self):
        return self.start_spinbox.value(), self.end_spinbox.value()

class InterpolationIntervalWidget(QFrame):
    removed = pyqtSignal(object)
    
    def __init__(self, parent=None, max_frame=10000):
        super().__init__(parent)
        
        self.start_frame_spinbox = QSpinBox()
        self.start_frame_spinbox.setRange(0, max_frame - 1)  # Set realistic frame range
        
        self.end_frame_spinbox = QSpinBox()
        self.end_frame_spinbox.setRange(1, max_frame)  # Set realistic frame range
        
        self.end_frame_spinbox.valueChanged.connect(self.update_start_frame_max)
        
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(["2x", "4x", "8x", "16x"])
        
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_interval)
        
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Start Frame:"))
        layout.addWidget(self.start_frame_spinbox)
        layout.addWidget(QLabel("End Frame:"))
        layout.addWidget(self.end_frame_spinbox)
        layout.addWidget(QLabel("Interpolation:"))
        layout.addWidget(self.interpolation_combo)
        layout.addWidget(self.remove_button)
        
        self.setLayout(layout)
        
    def remove_interval(self):
        self.setParent(None)
        self.removed.emit(self)  # Emit the removed signal

    def set_max_frame(self, max_frame):
        self.start_frame_spinbox.setMaximum(max_frame - 1)
        self.end_frame_spinbox.setMaximum(max_frame)
        self.update_start_frame_max()
        if self.end_frame_spinbox.value() > max_frame:
            self.end_frame_spinbox.setValue(max_frame)

    def update_start_frame_max(self):
        self.start_frame_spinbox.setMaximum(self.end_frame_spinbox.value() - 1)
    
    def get_interval_data(self):
        return {
            "start": self.start_frame_spinbox.value(),
            "stop": self.end_frame_spinbox.value(),
            "interp": int(math.log(int(self.interpolation_combo.currentText().replace("x", "")), 2))
        }

class InterpolationDialog(QDialog):
    def __init__(self, text = None, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Select Interpolation Intervals")
        
        self.layout = QVBoxLayout()

        if text:
            self.layout.addWidget(text)
        
        self.channel_combo = self.add_parent_combobox(self.parent.channelSelect)
        self.channel_combo.currentIndexChanged.connect(self.channel_select_change)
        
        self.layout.addWidget(QLabel("Channel to Interpolate:"))
        self.layout.addWidget(self.channel_combo)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red")
        self.layout.addWidget(self.error_label)
        
        self.add_interval_button = QPushButton("Add Interval")
        self.add_interval_button.clicked.connect(self.add_interval)
        
        self.dialog_buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self
        )
        self.dialog_buttons.accepted.connect(self.check_intervals)
        self.dialog_buttons.rejected.connect(self.reject)
        
        self.layout.addWidget(self.add_interval_button)
        self.layout.addWidget(self.dialog_buttons)
        
        self.setLayout(self.layout)
        
        self.intervals = []
        self.channel_select_change()
    
    def remove_all_intervals(self):
        for interval_widget in self.intervals:
            interval_widget.setParent(None)
        self.intervals.clear()

    def disable_and_clear_interval_add(self):
        self.remove_all_intervals()
        self.add_interval_button.setDisabled(True)
    
    def enable_interval_add(self):
        self.add_interval_button.setDisabled(False)

    def add_interval(self):
        max_frame = self.get_max_frame()
        interval_widget = InterpolationIntervalWidget(max_frame=max_frame)
        interval_widget.removed.connect(self.remove_interval)  # Connect the signal
        self.intervals.append(interval_widget)
        self.layout.insertWidget(self.layout.count() - 3, interval_widget)

    def remove_interval(self, interval_widget):
        if interval_widget in self.intervals:
            self.intervals.remove(interval_widget)
            interval_widget.deleteLater()

    def channel_select_change(self):
        self.channel_obj = self.parent.get_channel_obj_from_name(self.channel_combo.currentText())
        if isinstance(self.channel_obj, InterpolatedChannel):
            self.disable_and_clear_interval_add()
            self.showMessage(f"{self.channel_obj.name} has already been interpolated")
        else:
            self.clearMessages()
            self.enable_interval_add()
            self.update_max_frame()

    def update_max_frame(self):
        max_frame = self.get_max_frame()
        if max_frame == 0:
            self.disable_and_clear_interval_add()
            self.showMessage("Current Channel has only one image.")
        else:
            self.add_interval_button.setDisabled(False)
            self.clearMessages()
            for interval_widget in self.intervals:
                interval_widget.set_max_frame(max_frame)

    def add_parent_combobox(self, parentComboBox):
        channelSelect = QComboBox()
        channelSelect.setFixedWidth(200)
        channelSelect.addItems([parentComboBox.itemText(i) for i in range(parentComboBox.count())])
        channelSelect.setCurrentIndex(parentComboBox.currentIndex())
        return channelSelect

    def get_max_frame(self):
        return self.channel_obj.max_t()

    def check_intervals(self):
        new_intervals = []
        for interval_widget in self.intervals:
            start_frame = interval_widget.start_frame_spinbox.value()
            end_frame = interval_widget.end_frame_spinbox.value()
            interp = interval_widget.interpolation_combo.currentText()
            max_frame = self.get_max_frame()

            if start_frame < 0 or start_frame >= max_frame:
                self.showMessage(f"Start frame {start_frame} is out of range.")
                return
            if end_frame <= start_frame or end_frame > max_frame:
                self.showMessage(f"End frame {end_frame} is out of range or not greater than start frame.")
                return

            new_intervals.append((start_frame, end_frame, interp))

        # Check for overlapping intervals
        new_intervals.sort()  # Sort intervals by start frame
        for i in range(len(new_intervals) - 1):
            if new_intervals[i][1] > new_intervals[i + 1][0]:
                self.showMessage("Intervals overlap.")
                return

        self.accept()
        self.clearMessages()
    
    def get_data(self):
        channel = self.channel_combo.currentText()
        intervals = [interval_widget.get_interval_data() for interval_widget in self.intervals]
        intervals = sorted(intervals, key=lambda x: x['start'])
        return channel, intervals

    def showMessage(self, message):
        self.error_label.setText(message)

    def clearMessages(self):
        self.showMessage("") 

class TimedPopup(QDialog):
    def __init__(self, text, seconds, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Popup')
        label = QLabel(text)
        layout = QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.accept)
        self.timer.start(seconds*100)  # 10 seconds timeout

    def closeEvent(self, event):
        self.timer.stop()
        if self.parent():
            self.parent().popup = None
        super().closeEvent(event)

class FigureDialog(QDialog):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.layout = QGridLayout()
        self.row = 0

        self.label_widgets = {}
        self.label_names = self.parent.experiment().get_label_names()
        self.make_top_row()
        for i, label_name in enumerate(self.label_names):
            self.add_row(label_name, i)

    
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout.addWidget(self.buttonBox, self.row, 0)
        
        self.setLayout(self.layout)

    def make_top_row(self):
        for i, header in enumerate(["label name", "contour", "full label", "order"]):
            self.layout.addWidget(QLabel(header), self.row, i,1,1)
        self.row+=1
    
    def add_row(self, label_name, i):
        self.label_widgets[label_name] = {}
        self.layout.addWidget(QLabel(label_name), self.row, 0,1,1)

        contour_check = QCheckBox()
        self.label_widgets[label_name]["contours"] = contour_check
        self.layout.addWidget(contour_check, self.row,1,1,1)

        label_check = QCheckBox()
        self.label_widgets[label_name]["labels"] = label_check
        self.layout.addWidget(label_check, self.row,2,1,1)

        order_spin = QSpinBox()
        order_spin.setMaximum(len(self.label_names)-1)
        order_spin.setValue(i)
        self.label_widgets[label_name]["order"] = order_spin
        self.layout.addWidget(order_spin, self.row, 3,1,1)

        self.row+=1
    
    def get_data(self):
        data = self.label_widgets.copy()
        for name in data.keys():
            data[name]["contours"] = data[name]["contours"].isChecked()
            data[name]["labels"] = data[name]["labels"].isChecked()
            data[name]["order"] = data[name]["order"].value()
        return data


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
        self.spinBoxes = {}
        self.dSpinBoxes = {}

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
            elif paramType is int:
                self.spinBoxes[paramName] = self.params[paramName]
            elif paramType is float:
                self.dSpinBoxes[paramName] = self.params[paramName]
    
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
        
        self.spinBoxData = {}
        for boxName, defaultVal in self.spinBoxes.items():
            spinBox = QSpinBox(self)
            spinBox.setValue(int(defaultVal))
            self.spinBoxData[boxName] = spinBox
            self.formLayout.addRow(self.produceLabel(boxName), spinBox)
        for boxName, defaultVal in self.dSpinBoxes.items():
            spinBox = QDoubleSpinBox(self)
            spinBox.setValue(float(defaultVal))
            spinBox.setMinimum(-1.0)
            spinBox.setMaximum(1000.0)
            spinBox.setSingleStep(0.1)
            self.spinBoxData[boxName] = spinBox
            self.formLayout.addRow(self.produceLabel(boxName), spinBox)

    
    def addParentComboBox(self, parentComboBox, name = "Channel", addNA = False):
        channelSelect = QComboBox()
        channelSelect.setFixedWidth(200)
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
        spinData = {k:v.value() for k,v in self.spinBoxData.items()}
        return dropDownData | lineEditData | boolData | spinData

class ModelParamDialog(GeneralParamDialog):
    def __init__(self, hyperparamDict, paramtypes, modelName, parent = None):
        if parent.experiment().has_labels() and modelName != "artilife":
            curr_label = parent.label().name
            hyperparamDict[f"insert into {curr_label}"] = False
            paramtypes.append(bool)
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
    
        self.spinBoxData = {}
        for boxName, defaultVal in self.spinBoxes.items():
            spinBox = QSpinBox(self)
            spinBox.setValue(int(defaultVal))
            self.spinBoxData[boxName] = spinBox
            self.formLayout.addRow(self.produceLabel(boxName), spinBox)
        for boxName, defaultVal in self.dSpinBoxes.items():
            spinBox = QDoubleSpinBox(self)
            spinBox.setMinimum(-1.0)
            spinBox.setSingleStep(0.1)
            spinBox.setMaximum(1000.0)
            spinBox.setValue(float(defaultVal))
            self.spinBoxData[boxName] = spinBox
            self.formLayout.addRow(self.produceLabel(boxName), spinBox)

    def channelSelectIndexChanged(self):
        channelIndex = self.parent.channelSelect.findText(self.dropDownData["Channel"].currentText())
        maxT = self.parent.experiment().channels[channelIndex].max_t()
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
        self.dropDownData[f"{prefix}Start"].setCurrentIndex(self.parent.tIndex)
        self.dropDownData[f"{prefix}Stop"].addItems(validTs)
        self.dropDownData[f"{prefix}Stop"].setCurrentIndex(self.parent.tIndex)
    
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
        channelSelect.setFixedWidth(200)
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
        
        self.addArtilifeWeightDropDown()

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
    
        self.spinBoxData = {}
        for boxName, defaultVal in self.spinBoxes.items():
            spinBox = QSpinBox(self)
            spinBox.setValue(int(defaultVal))
            self.spinBoxData[boxName] = spinBox
            self.formLayout.addRow(self.produceLabel(boxName), spinBox)
        for boxName, defaultVal in self.dSpinBoxes.items():
            spinBox = QDoubleSpinBox(self)
            spinBox.setMaximum(1000.0)
            spinBox.setValue(float(defaultVal))
            self.spinBoxData[boxName] = spinBox
            self.formLayout.addRow(self.produceLabel(boxName), spinBox)
        
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

    def addArtilifeWeightDropDown(self):
        artilifeWeights = self.parent.getModelWeights(name = "artilife")
        artiWeights = QComboBox()
        artiWeights.addItems(artilifeWeights)
        self.dropDownData["artiWeights"] = artiWeights
        artiWeights.setFixedWidth(200)
        artiWeights.setEnabled(True)
        artiWeights.setCurrentIndex(0)
        self.formLayout.addRow(QLabel("artilife weights"), artiWeights)

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
        self.formLayout.addRow(QLabel(f"{modelType} Weights"), weights)


    def channelSelectIndexChanged(self):
        channelIndex = self.parent.channelSelect.findText(self.dropDownData["Channel"].currentText())
        maxT = self.parent.experiment().channels[channelIndex].max_t()
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
        lossDisplay.addItem(str(self.lossName))
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
        qlabel = QLabel("New Model Suffix")
        qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.l0.addWidget(qlabel, self.yoff,0,1,1)
        self.suffixName = QLineEdit()
        self.suffixName.setToolTip("Initial model name will automatically be added as a prefix to New Model Suffix.")
        self.suffixName.setText(self.parent.trainModelSuffix)
        self.suffixName.setFixedWidth(200)
        self.l0.addWidget(self.suffixName, self.yoff, 1,1,1)

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
        suffix = str(self.suffixName.text())
        return {k:float(v.text()) for k,v in self.edits.items()} | {"scratch": self.checkbox.isChecked()} | {"model_name": f"{self.modelName}_{suffix}"}

    
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
        modelClass =  self.parent.getModelClass(self.modelType)
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
        qlabel = QLabel('ROIs')
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



class PlotWindowCustomizeTimeSeries(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setGeometry(100,100,1000,450)
        self.setWindowTitle("Choose Properties to Plot Versus Time")
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


class PlotWindowCustomizePerFrame(QDialog):
    def __init__(self, properties, masks, max_selected=6, parent=None):
        super(PlotWindowCustomizePerFrame, self).__init__(parent)
        self.properties = properties
        self.masks = masks
        self.max_selected = max_selected
        self.selected_properties = {mask: [] for mask in masks}
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout()  # Main layout for the dialog
        
        self.layout = QHBoxLayout()  # Horizontal layout for the checkboxes
        self.checkboxes = {mask: [] for mask in self.masks}
        for mask in self.masks:
            mask_layout = QVBoxLayout()
            mask_label = QLabel(mask)
            mask_layout.addWidget(mask_label)
            for prop in self.properties:
                checkbox = QCheckBox(prop)
                checkbox.stateChanged.connect(self.on_checkbox_state_changed)
                mask_layout.addWidget(checkbox)
                self.checkboxes[mask].append(checkbox)
            self.layout.addLayout(mask_layout)
        
        main_layout.addLayout(self.layout)

        # Add mutually exclusive checkboxes for frames
        self.frame_checkboxes = QHBoxLayout()
        self.calculate_all_frames_checkbox = QCheckBox("Calculate for all frames")
        self.calculate_current_frame_checkbox = QCheckBox("Calculate for current frame")
        self.calculate_current_frame_checkbox.setChecked(True)
        
        self.frame_button_group = QButtonGroup()
        self.frame_button_group.addButton(self.calculate_all_frames_checkbox)
        self.frame_button_group.addButton(self.calculate_current_frame_checkbox)
        self.frame_button_group.setExclusive(True)
        
        self.frame_checkboxes.addWidget(self.calculate_all_frames_checkbox)
        self.frame_checkboxes.addWidget(self.calculate_current_frame_checkbox)
        
        main_layout.addLayout(self.frame_checkboxes)
        
        # Add OK button
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        self.button_box.accepted.connect(self.accept)
        main_layout.addWidget(self.button_box)
        
        self.setLayout(main_layout)
        self.setWindowTitle("Select Properties")
        
    def on_checkbox_state_changed(self, state):
        total_selected = sum(len(props) for props in self.selected_properties.values())

        sender = self.sender()
        
        if state == Qt.Checked and total_selected >= self.max_selected:
            QMessageBox.warning(self, "Limit Exceeded", f"A maximum of {self.max_selected} properties can be selected.")
            sender.blockSignals(True)
            sender.setChecked(False)
            sender.blockSignals(False)
            return
        
        for mask, checkboxes in self.checkboxes.items():
            self.selected_properties[mask] = [cb.text() for cb in checkboxes if cb.isChecked()]
    
    def get_data(self):
        return self.selected_properties
    
    def get_unique_properties(self):
        unique_properties = set()
        for props in self.selected_properties.values():
            unique_properties.update(props)
        return list(unique_properties)
    
    def do_for_all_frames(self):
        return self.calculate_all_frames_checkbox.isChecked()


