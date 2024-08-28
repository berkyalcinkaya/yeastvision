from PyQt5.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QWizard, QWizardPage, QComboBox, QCheckBox
from PyQt5.QtCore import pyqtSignal
from yeastvision.models.proSeg.model import ProSeg
from yeastvision.models.budSeg.model import BudSeg
from yeastvision.parts.wizard_utils import InterpolationDialog, SimpleTextPage, ParameterInputPage

fiest_instructs = '''Frame Interpolation Enhanced Single-cell Tracking (FIEST),(FIEST):\n
Step 1: Interpolate images over specified intervals to enhance resolution\n
Step 2: Segment interpolation-enhanced images with proSeg\n
Step 3: Track resulting masks\n
Step 4 (optional): compute lineages\n
Step 5: De-interpolate masks, leaving only masks corresponding to original images\n'''

class DropDownCheckBoxPage(QWizardPage):
    completeChanged = pyqtSignal()
    def __init__(self, viable_channels, parent=None):
        super().__init__(parent)
        self.setTitle("Choose Channel And Lineages")

        # Create the label and combo box
        label = QLabel("Channel:")
        self.channel_select = QComboBox()
        self.channel_select.addItems([ts.name for ts in viable_channels])

        # Create the checkbox
        self.checkbox = QCheckBox("do_lineage")

        # Create a horizontal layout for the label and combo box
        h_layout = QHBoxLayout()
        h_layout.addWidget(label)
        h_layout.addWidget(self.channel_select)

        # Create the main layout and add the horizontal layout and checkbox to it
        layout = QVBoxLayout()
        layout.addLayout(h_layout)
        layout.addWidget(self.checkbox)
        self.setLayout(layout)

        # Register the combo box index field with automatic updates
        self.registerField("channelIndex*", self.channel_select, "currentIndex", self.channel_select.currentIndexChanged)
        self.registerField("doLineage*", self.checkbox)

        # Connect signals to check for completeness
        self.channel_select.currentIndexChanged.connect(self.completeChanged.emit)
        self.checkbox.stateChanged.connect(self.completeChanged.emit)

    def isComplete(self):
        # Page is complete if a valid channel is selected (index >= 0)
        return self.channel_select.currentIndex() >= 0

    def getData(self):
        return {
            "channelIndex": self.field("channelIndex"),
            "doLineage": self.field("doLineage")
        }

class ParameterInputPageFiestAsexual(ParameterInputPage):
    def __init__(self, params_dict, model_weights, modelName, channels, parent=None, includeTime=True):
        super().__init__(params_dict, model_weights, modelName, channels, parent=parent, includeTime=includeTime)
    
    def nextId(self):
        if self.field("doLineage"):
            return 5  # ID of the additional DisplaySelectionPage
        else:
            return -1  # Complete the wizard
    
class AdditionalParameterInputPage(ParameterInputPageFiestAsexual):
    def __init__(self, params_dict, model_weights, modelName, channels, parent=None, includeTime=True):
        super().__init__(params_dict, model_weights, modelName, channels, parent=parent, includeTime=includeTime)
    
    def nextId(self):
        return -1
    
class FiestWizard(QWizard):
    def __init__(self, parent, channels, proSeg_weights, budSeg_weights):
        super().__init__(parent)
        
        self.setWindowTitle("Frame Interpolation Enhanced Segmentation Tracking")

        self.setPage(1, SimpleTextPage(fiest_instructs))
        self.setPage(2, DropDownCheckBoxPage(channels, parent=self))
        self.setPage(3, InterpolationDialog(channels, parent=self))
        self.setPage(4, ParameterInputPage(ProSeg.hyperparams, proSeg_weights, "proSeg", channels, parent=self, 
                                           includeTime=False))
        self.setPage(5, AdditionalParameterInputPage(BudSeg.hyperparams, budSeg_weights, "budSeg", channels, parent=self, 
                                                     includeTime=False))

        self.non_model_param_ids = [2,3]
        self.cyto_seg_id = 4
        self.bud_seg_id = 5
    
    def getData(self):
        data = {}
        for page_id in self.non_model_param_ids:
            page = self.page(page_id)
            page_data = page.getData()
            data.update(page_data)
        proSeg_page = self.page(self.cyto_seg_id)
        data["proSeg"] = proSeg_page.getData()
        if self.field("doLineage"):
            bud_page = self.page(self.bud_seg_id)
            data["budSeg"] = bud_page.getData()
        return data