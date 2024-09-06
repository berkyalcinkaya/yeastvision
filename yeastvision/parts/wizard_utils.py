from PyQt5.QtWidgets import QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QWizard, QWizardPage, QSpinBox, QComboBox, QCheckBox
from PyQt5.QtCore import pyqtSignal
from yeastvision.parts.dialogs import InterpolationIntervalWidget
from yeastvision.parts.guiparts import NumericLineEdit


class SimpleTextPage(QWizardPage):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setTitle("Information Page")
        
        layout = QVBoxLayout()
        text_label = QLabel(text)
        layout.addWidget(text_label)
        self.setLayout(layout)


class InterpolationDialog(QWizardPage):
    completeChanged = pyqtSignal()
    def __init__(self, channel_options, text = None, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setTitle("Select Interpolation Intervals")
        
        self.layout = QVBoxLayout()

        if text:
            self.layout.addWidget(text)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red")
        self.layout.addWidget(self.error_label)
        
        self.add_interval_button = QPushButton("Add Interval")
        self.add_interval_button.clicked.connect(self.add_interval)
        
        self.layout.addWidget(self.add_interval_button)
        self.setLayout(self.layout)
        
        self.intervals = []
        self.channels = channel_options
        self.channel_obj = None
    
    def initializePage(self):
        self.remove_all_intervals()
        channel_index = self.field("channelIndex")
        self.channel_obj = self.channels[channel_index]
        self.update_max_frame()

    def isComplete(self):
        # Page is complete if a valid channel is selected (index >= 0)
        return self.check_intervals()

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
        self.layout.insertWidget(self.layout.count() - 1, interval_widget)
        self.completeChanged.emit()

    def remove_interval(self, interval_widget):
        if interval_widget in self.intervals:
            self.intervals.remove(interval_widget)
            interval_widget.deleteLater()
            self.completeChanged.emit()

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
                return False
            if end_frame <= start_frame or end_frame > max_frame:
                self.showMessage(f"End frame {end_frame} is out of range or not greater than start frame.")
                return False

            new_intervals.append((start_frame, end_frame, interp))

        # Check for overlapping intervals
        new_intervals.sort()  # Sort intervals by start frame
        for i in range(len(new_intervals) - 1):
            if new_intervals[i][1] > new_intervals[i + 1][0]:
                self.showMessage("Intervals overlap.")
                return False

        self.clearMessages()
        return True
    
    def getData(self):
        intervals = [interval_widget.get_interval_data() for interval_widget in self.intervals]
        intervals = sorted(intervals, key=lambda x: x['start'])
        return {"intervals": intervals}

    def showMessage(self, message):
        self.error_label.setText(message)

    def clearMessages(self):
        self.showMessage("") 


class ParameterInputPage(QWizardPage):
    def __init__(self, params_dict, model_weights, modelName, channels, parent=None, includeTime=True):
        super().__init__(parent)
        self.setTitle(f"Set {modelName} Parameters")
        self.modelName = modelName
        self.params_dict = params_dict
        self.includeTime=includeTime

        # Layout for the parameters
        self.param_layout = QVBoxLayout()

        # Create QLineEdits for each parameter
        self.line_edits = {}
        self.check_boxes = {}
        for key, value in params_dict.items():
            h_layout = QHBoxLayout()
            label = QLabel(f"{key}:")
            if isinstance(value, bool):
                checkbox = QCheckBox(self)
                checkbox.setCheckState(value)
                self.check_boxes[key] = checkbox
                h_layout.addWidget(label)
                h_layout.addWidget(checkbox)
                self.param_layout.addLayout(h_layout)
            else:
                line_edit = NumericLineEdit(str(value))
                self.line_edits[key] = line_edit
                h_layout.addWidget(label)
                h_layout.addWidget(line_edit)
                self.param_layout.addLayout(h_layout)

        if self.includeTime:
            # Create spin boxes for t_start and t_stop
            self.t_start_spinbox = QSpinBox()
            self.t_stop_spinbox = QSpinBox()

            self.t_start_spinbox.setRange(0, 1000)  # Arbitrary large max, will be set correctly in initializePage
            self.t_stop_spinbox.setRange(0, 1000)

            # Create layout for the spin boxes
            spin_layout = QHBoxLayout()
            spin_layout.addWidget(QLabel("t_start:"))
            spin_layout.addWidget(self.t_start_spinbox)
            spin_layout.addWidget(QLabel("t_stop:"))
            spin_layout.addWidget(self.t_stop_spinbox)

        # Create combo box for model weights
        self.model_weights_combo = QComboBox()
        self.model_weights_combo.addItems(model_weights)
        self.model_weights_combo.setCurrentIndex(0)
        model_weights_layout = QHBoxLayout()
        model_weights_layout.addWidget(QLabel("Choose model weights:"))
        model_weights_layout.addWidget(self.model_weights_combo)

        # Main layout
        layout = QVBoxLayout()
        layout.addLayout(self.param_layout)
        if self.includeTime:
            layout.addLayout(spin_layout)
        layout.addLayout(model_weights_layout)
        self.setLayout(layout)

        # Connect signals to ensure proper min/max behavior
        if self.includeTime:
            self.t_start_spinbox.valueChanged.connect(self.update_t_stop_min)
            self.t_stop_spinbox.valueChanged.connect(self.update_t_start_max)

        self.channels = channels
        self.channel_obj = None

    def update_t_stop_min(self, value):
        self.t_stop_spinbox.setMinimum(value)

    def update_t_start_max(self, value):
        self.t_start_spinbox.setMaximum(value)

    def initializePage(self):
        # Retrieve the channel index
        channel_index = self.field("channelIndex")
        self.channel_obj = self.channels[channel_index]
        max_t = self.channel_obj.max_t()+1
        
        if self.includeTime:
            self.t_start_spinbox.setMaximum(max_t)
            self.t_stop_spinbox.setMaximum(max_t)
            self.t_stop_spinbox.setValue(max_t)
        
    def getData(self):
        data = {
            "modelName": self.modelName,
            "modelWeight": self.model_weights_combo.currentText(),
        }
        
        if self.includeTime:
            data.update({"t_start": int(self.t_start_spinbox.value()), 
                         "t_stop": int(self.t_stop_spinbox.value())}
            )
        for key in self.line_edits:
            data[key] = float(self.line_edits[key].text())
        for key in self.check_boxes:
            data[key] = bool(self.check_boxes[key].isChecked())
        return data

