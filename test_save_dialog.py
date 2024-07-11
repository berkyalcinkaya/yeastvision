import sys
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QSpinBox, QPushButton, QHBoxLayout

class IntervalSelectionDialog(QDialog):
    def __init__(self, intervals, maxT, windowTitle, parent=None, labels = None, presetT1 = 0, presetT2 = None):
        super().__init__(parent)
        self.intervals = intervals
        self.maxT = maxT

        self.setWindowTitle(windowTitle)
        self.setGeometry(100, 100, 400, 200)

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
        self.end_spinbox.setMinimum(self.start_spinbox.value())

    def _sync_end_spinbox(self):
        if self.end_spinbox.value() < self.start_spinbox.value():
            self.end_spinbox.setValue(self.start_spinbox.value() + 1)

    def get_selected_interval(self):
        return self.start_spinbox.value(), self.end_spinbox.value()

# Example usage
if __name__ == "__main__":
    app = QApplication(sys.argv)
    intervals = ["0-5", "10-15", "20-25"]
    maxT = 25
    dialog = IntervalSelectionDialog(intervals, maxT)
    if dialog.exec_() == QDialog.Accepted:
        start, end = dialog.get_selected_interval()
        print(f"Selected interval: {start}-{end}")
    sys.exit(app.exec_())
