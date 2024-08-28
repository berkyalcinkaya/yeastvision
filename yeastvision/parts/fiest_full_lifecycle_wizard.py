from PyQt5.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QWizard, QWizardPage, QComboBox
from PyQt5.QtCore import pyqtSignal
from yeastvision.parts.wizard_utils import SimpleTextPage, InterpolationDialog, ParameterInputPage
from yeastvision.models.proSeg.model import ProSeg
from yeastvision.models.matSeg.model import MatSeg
from yeastvision.models.spoSeg.model import SpoSeg

fiest_instructs = '''Frame Interpolation Enhanced Single-cell Tracking (FIEST), for the Full Yeast Lifecycle:\n
Step 1: Interpolate images over specified intervals to enhance resolution\n
Step 2: Segment (interpolated) move with matSeg, spoSeg, and tetradSeg\n
Step 3: Track tetrads, mating cells, and proliferating cells independently\n
Step 4: Correction Steps\n
Step 5: De-interpolate masks, leaving only masks corresponding to original images
'''

class DropDownCheckBoxPage(QWizardPage):
    completeChanged = pyqtSignal()
    def __init__(self, viable_channels, parent=None):
        super().__init__(parent)
        self.setTitle("Choose Channel")

        # Create the label and combo box
        label = QLabel("Channel:")
        self.channel_select = QComboBox()
        self.channel_select.addItems([ts.name for ts in viable_channels])

        # Create a horizontal layout for the label and combo box
        h_layout = QHBoxLayout()
        h_layout.addWidget(label)
        h_layout.addWidget(self.channel_select)

        # Create the main layout and add the horizontal layout and checkbox to it
        layout = QVBoxLayout()
        layout.addLayout(h_layout)
        self.setLayout(layout)

        # Register the combo box index field with automatic updates
        self.registerField("channelIndex*", self.channel_select, "currentIndex", self.channel_select.currentIndexChanged)

        # Connect signals to check for completeness
        self.channel_select.currentIndexChanged.connect(self.completeChanged.emit)

    def isComplete(self):
        # Page is complete if a valid channel is selected (index >= 0)
        return self.channel_select.currentIndex() >= 0

    def getData(self):
        return {
            "channelIndex": self.field("channelIndex"),
        }

class FiestFullLifeCycleWizard(QWizard):
    def __init__(self, parent, channels, proSeg_weights, matSeg_weights, spoSeg_weights):
        super().__init__(parent)
        self.setWindowTitle("Frame Interpolation Enhanced Segmentation Tracking: Full Lifecycle")
        
        self.setPage(1, SimpleTextPage(fiest_instructs))
        self.setPage(2, DropDownCheckBoxPage(channels, parent=self))
        self.setPage(3, InterpolationDialog(channels, parent=self))
        self.setPage(4, ParameterInputPage(ProSeg.hyperparams, proSeg_weights, "proSeg", channels, parent=self, includeTime=False))
        self.setPage(5, ParameterInputPage(MatSeg.hyperparams, matSeg_weights, "matSeg", channels, parent=self))
        self.setPage(6, ParameterInputPage(SpoSeg.hyperparams, spoSeg_weights, "spoSeg", channels, parent=self))

        self.non_model_param_ids = [2,3]
        self.cyto_seg_id = 4
        self.mat_seg_id = 5
        self.spo_seg_id = 6

    def getData(self):
        data = {}
        for page_id in self.non_model_param_ids:
            page = self.page(page_id)
            page_data = page.getData()
            data.update(page_data)
        proSeg_page = self.page(self.cyto_seg_id)
        data["proSeg"] = proSeg_page.getData()
        matSeg_page = self.page(self.mat_seg_id)
        data["matSeg"] = matSeg_page.getData()
        spoSeg_page = self.page(self.spo_seg_id)
        data["spoSeg"] = spoSeg_page.getData()
        return data