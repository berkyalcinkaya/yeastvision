from PyQt5 import QtCore
from PyQt5.QtGui import QPainter, QIcon, QPixmap, QIntValidator
from PyQt5.QtWidgets import (QApplication,QSlider, QStyle, QStyleOptionSlider, QPushButton,QCheckBox, QComboBox, QFrame,
                            QStyledItemDelegate, QListView, QLineEdit, QMainWindow, QWidget, QGridLayout, QLabel)
from PyQt5.QtCore import Qt, QRect
import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget

class MultiLabelWindow(QWidget):
    def __init__(self, channel_id, mask_ids, exp_idx=None, parent=None):
        super().__init__()
        self.exp_idx = exp_idx
        self.mask_ids = mask_ids
        self.channel_id = channel_id
        self.num_masks = len(mask_ids)
        self.image_panels = []
        self.mask_panels = []  # Separate panels for masks
        self.parent = parent
        
        self.exp = self.parent.experiments[self.exp_idx]
        self.channel = self.parent.getChannelFromId(self.channel_id, exp_idx=self.exp_idx)
        
        # Set up the layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Determine layout based on num_masks
        if self.num_masks <= 3:
            # Single row with num_masks columns
            for i in range(self.num_masks):
                self._create_panel(i, 0, i)
        else:
            # Multiple rows, two columns, account for odd numbers
            for i in range(self.num_masks):
                row = i // 2
                col = i % 2
                self._create_panel(i, row, col)

    def _create_panel(self, i, row, col):
        image_panel = pg.ImageItem()
        mask_panel = pg.ImageItem()  # Separate mask panel
        self.image_panels.append(image_panel)
        self.mask_panels.append(mask_panel)

        # Create the title label
        mask_obj = self.parent.getMaskFromId(self.mask_ids[i], exp_idx=self.exp_idx)
        title = QLabel(f"<b>{mask_obj.name}</b>")
        title.setAlignment(Qt.AlignCenter)

        view = pg.GraphicsLayoutWidget()
        vb = view.addViewBox()

        # Add the image panel first
        vb.addItem(image_panel)
        # Add the mask panel on top of the image
        vb.addItem(mask_panel)

        # Add the title above the view
        self.layout.addWidget(title, row * 2, col)  # Titles occupy separate rows
        # Add the view containing the image and mask panels
        self.layout.addWidget(view, row * 2 + 1, col)

    def get_channel_im(self):
        if self.parent.tIndex > self.channel.max_t():
            return np.zeros((512, 512))
        else:
            return self.exp.get_channel(id=self.channel_id, t=self.parent.tIndex)
    
    def get_mask_ims(self):
        masks = []
        for mask_id in self.mask_ids:
            if not self.exp.has_labels():
                mask = np.zeros(self.exp.shape()[:2], dtype=np.uint8)
            mask_obj = self.parent.getMaskFromId(mask_id, exp_idx=self.exp_idx)
            if self.parent.tIndex > mask_obj.max_t():
                mask = np.zeros(self.exp.shape()[:2], dtype=np.uint8)
            
            mask = self.exp.get_label("labels", id=mask_id, t=self.parent.tIndex)
            #print(mask_obj.name, mask.shape, np.unique(mask))
            masks.append(mask)
        return masks

    def update(self):
        # Display the image in all panels, and the masks in their respective panels
        image = self.get_channel_im()
        mask_list = self.get_mask_ims()
        for i in range(self.num_masks):
            if i < len(mask_list):
                mask = mask_list[i]
            else:
                # Fill with zeros if not enough masks are provided
                mask = np.zeros(image.shape[:2] + (4,), dtype=np.uint8)  # Assume RGBA mask

            mask_colored = self.parent.cmap[mask]  # Assuming you are using a colormap for the mask

            # Update the image panel with the original image
            self.image_panels[i].setImage(image, autoLevels=True)
            # Update the mask panel with the RGBA mask
            self.mask_panels[i].setImage(mask_colored, autoLevels=False, levels=[0, 255])

    def closeEvent(self, event):
        if self.parent:
            self.parent.multi_label_window = None  # Clear the reference to this window in the parent
        event.accept()  # Accept the event to close the window

    def keyPressEvent(self, event):
        # Forward the key press event to the parent (MainWindow)
        if self.parent:
            self.parent.keyPressEvent(event)  # Delegate the event to the parent
        else:
            super().keyPressEvent(event)

class MeasureWindow(QtWidgets.QWidget):
    def __init__(self, img_data, parent=None):
        super().__init__()
        
        self.parent=parent
        
        # Set up the auxiliary window layout
        self.setWindowTitle("Measure Window")
        self.resize(800, 600)
        
        # Create the layout for this window
        self.layout = QtWidgets.QVBoxLayout(self)
        
        # Create a QLabel to display the length of the line
        self.length_label = QtWidgets.QLabel("Length: 0 px", self)
        self.length_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.layout.addWidget(self.length_label)
        
        # Create a pyqtgraph widget to display the image
        self.pg_widget = GraphicsLayoutWidget()
        self.layout.addWidget(self.pg_widget)
        
        # Add a ViewBox to display the image
        self.view_box = self.pg_widget.addViewBox()
        self.view_box.setAspectLocked(True)  # Keep the aspect ratio of the image
        
        # Create ImageItem and display the image
        self.image_item = pg.ImageItem(img_data)
        self.view_box.addItem(self.image_item)
        
        # Create a Line ROI for measuring distances
        self.line_roi = pg.LineSegmentROI([[100, 100], [200, 200]], pen='r')
        self.view_box.addItem(self.line_roi)
        
        # Connect the Line ROI's change event to update the displayed length
        self.line_roi.sigRegionChangeFinished.connect(self.update_length)
        
        # Initial update of the line length
        self.update_length()
    
    def update_length(self):
        """Update the length of the line and display it on the QLabel."""
        p1, p2 = self.line_roi.getSceneHandlePositions()
        p1 = p1[1]
        p2 = p2[1]
        
        # Calculate the length of the line in pixels
        length = np.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)
        
        # Update the QLabel with the line length
        self.length_label.setText(f"Length: {length:.2f} px")
    
    def closeEvent(self, event):
        if self.parent:
            self.parent.measure_window = None  # Clear the reference to this window in the parent
        event.accept()  # Accept the event to close the window

class NumericLineEdit(QLineEdit):
    def __init__(self, default_value="", parent=None):
        super().__init__(default_value, parent)
        self.setValidator(QIntValidator())

class RemoveItemDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super(RemoveItemDelegate, self).__init__(parent)

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        
        close_icon =  self.parent().style().standardIcon(QStyle.SP_DockWidgetCloseButton)
        #close_icon = self.get_close_icon()
        rect = self.get_close_button_rect(option.rect)
        close_icon.paint(painter, rect)

    def get_close_icon(self):
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.red)
        painter.drawEllipse(0, 0, 16, 16)
        painter.setPen(Qt.white)
        painter.drawLine(4, 4, 12, 12)
        painter.drawLine(4, 12, 12, 4)
        painter.end()
        close_icon = QIcon(pixmap)
        return close_icon


    def get_close_button_rect(self, item_rect):
        icon_size = 16
        return QRect(item_rect.right() - icon_size - 5, item_rect.center().y() - icon_size // 2, icon_size, icon_size)

class CustomListView(QListView):
    def __init__(self, combo, delete_method, parent=None, channel = False):
        super(CustomListView, self).__init__(parent)
        self._combo = combo
        self._delete_method = delete_method
        self.parent = parent
        self.channel = channel

    def mousePressEvent(self, event):
        index_under_mouse = self.indexAt(event.pos())
        delegate = self._combo.itemDelegate()
        if isinstance(delegate, RemoveItemDelegate) and delegate.get_close_button_rect(self.visualRect(index_under_mouse)).contains(event.pos()):
            index = index_under_mouse.row()
            if self._delete_method(index):
                current_index = self._combo.currentIndex()
                self.parent.blockComboSignals(True)
                self._combo.removeItem(index_under_mouse.row())
                new_index = self.update_index_after_removal(current_index)
                self.parent.onDelete(index, new_index, channel = self.channel)
                self.parent.blockComboSignals(False)
        else:
            super().mousePressEvent(event)

    def update_index_after_removal(self, removed_index):
        num_items = self._combo.count()
        if num_items == 0:
            return  -1 # Optionally, disable the combo box or inform the user
        new_index = removed_index if removed_index < num_items else num_items - 1
        self._combo.setCurrentIndex(new_index)
        self._combo.setEditText(self._combo.currentText())

        return new_index


class CustomComboBox(QComboBox):
    def __init__(self, delete_method, parent=None, channel = False):
        super(CustomComboBox, self).__init__(parent)
        self.setView(CustomListView(self, delete_method, parent = parent, channel = channel))
        self.setItemDelegate(RemoveItemDelegate(self))

    def addNewItem(self, name):
        self.addItem(name)




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

class ReadOnlyCheckBox(QCheckBox):
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def mousePressEvent(self, event):
        event.ignore()

class MaskTypeButtons():
    def __init__(self, parent=None, row=0, col=0):
        super(MaskTypeButtons, self).__init__()
        self.parent = parent
        self.buttonStrings = ["Mask", "Probability"] # Mask id = 0, Probability = 2
        self.buttons = []

        self.maskId = 0
        self.probId = 1

        for colAdd in range(len(self.buttonStrings)):
            button = QCheckBox(self.buttonStrings[colAdd])
            button.setStyleSheet(self.parent.checkstyle)
            button.setFont(self.parent.medfont)

            if colAdd == self.maskId:
                button.stateChanged.connect(self.maskButtonPress)
            else:
                button.stateChanged.connect(self.probButtonPress)

            self.parent.l.addWidget(button,row, col+2*colAdd, 1,2)
            button.setCheckState(False)
            button.setEnabled(False)
            self.buttons.append(button)
        
        self.maskButton = self.buttons[self.maskId]
        self.probButton = self.buttons[self.probId]
    
    def maskButtonPress(self):
        if self.maskButton.isChecked():

            if self.parent.maskOn:
                self.toggleButton(self.probButton, False)
                self.parent.changeMaskType(self.maskId)
            else:
                self.parent.maskType = self.maskId
                self.parent.addMask()
        
        elif not self.maskButton.isChecked():
            self.parent.hideMask()
    
    def checkMaskRemote(self):
        if not self.maskButton.isChecked():
            self.maskButton.setChecked(True)
            self.parent.maskType = self.maskId
    
    def checkProbRemote(self):
        if not self.probButton.isChecked():
            self.probButton.setChecked(True)
    
    def probButtonPress(self):

        if self.parent.maskData.channels[self.parent.maskZ].shape[0]>1:

            if not self.probButton.isChecked():
                self.parent.hideMask()
            
            elif self.probButton.isChecked():
                if self.parent.maskOn:  
                    self.toggleButton(self.maskButton, False)
                    self.parent.changeMaskType(self.probId)
                else:
                    self.parent.maskType = self.probId
        
        else:
            self.probButton.setCheckState(False)

    def setEnabled(self, b):
        for button in self.buttons:
            button.setEnabled(b)
    
    def uncheckAll(self):
        for button in self.buttons:
            button.setCheckState(False)
    
    def toggleButton(self, button, boolean):
        button.blockSignals(True)
        button.setCheckState(boolean)
        button.blockSignals(False)      

class RangeSlider(QSlider):
    """ A slider for ranges.
        This class provides a dual-slider for ranges, where there is a defined
        maximum and minimum, as is a normal slider, but instead of having a
        single slider value, there are 2 slider values.
        This class emits the same signals as the QSlider base class, with the
        exception of valueChanged
        Found this slider here: https://www.mail-archive.com/pyqt@riverbankcomputing.com/msg22889.html
        and modified it
    """
    def __init__(self, parent=None, *args):
        super(RangeSlider, self).__init__(*args)

        self._low = self.minimum()
        self._high = self.maximum()

        self.setTickInterval(1)

        self.pressed_control = QStyle.SC_None
        self.hover_control = QStyle.SC_None
        self.click_offset = 0

        self.setOrientation(QtCore.Qt.Horizontal)
        self.setTickPosition(QSlider.TicksRight)
        self.setStyleSheet(\
                "QSlider::handle:horizontal {\
                background-color: white;\
                border: 1px solid white;\
                border-radius: 2px;\
                border-color: white;\
                height: 8px;\
                width: 6px;\
                margin: 0px 2; \
                }")

        # 0 for the low, 1 for the high, -1 for both
        self.active_slider = 0
        self.parent = parent

    def level_change(self):
        if self.parent is not None:
            self.parent.saturation = [self._low, self._high]
            self.parent.pg_im.setLevels(self.parent.saturation)

    def low(self):
        return self._low

    def setLow(self, low):
        self._low = low
        self.update()

    def high(self):
        return self._high

    def setHigh(self, high):
        self._high = high
        self.update()
    
    def resetLevels(self, saturationList):
        self.setLow(saturationList[0])
        self.setHigh(saturationList[1])

    def paintEvent(self, event):
        # based on http://qt.gitorious.org/qt/qt/blobs/master/src/gui/widgets/qslider.cpp
        painter = QPainter(self)
        style = QApplication.style()

        for i, value in enumerate([self._low, self._high]):
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)

            # Only draw the groove for the first slider so it doesn't get drawn
            # on top of the existing ones every time
            if i == 0:
                opt.subControls = QStyle.SC_SliderHandle#QStyle.SC_SliderGroove | QStyle.SC_SliderHandle
            else:
                opt.subControls = QStyle.SC_SliderHandle

            if self.tickPosition() != self.NoTicks:
                opt.subControls |= QStyle.SC_SliderTickmarks

            if self.pressed_control:
                opt.activeSubControls = self.pressed_control
                opt.state |= QStyle.State_Sunken
            else:
                opt.activeSubControls = self.hover_control

            opt.sliderPosition = int(value)
            opt.sliderValue = int(value)
            style.drawComplexControl(QStyle.CC_Slider, opt, painter, self)

    def mousePressEvent(self, event):
        event.accept()

        style = QApplication.style()
        button = event.button()
        # In a normal slider control, when the user clicks on a point in the
        # slider's total range, but not on the slider part of the control the
        # control would jump the slider value to where the user clicked.
        # For this control, clicks which are not direct hits will slide both
        # slider parts
        if button:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)

            self.active_slider = -1

            for i, value in enumerate([self._low, self._high]):
                opt.sliderPosition = value
                hit = style.hitTestComplexControl(style.CC_Slider, opt, event.pos(), self)
                if hit == style.SC_SliderHandle:
                    self.active_slider = i
                    self.pressed_control = hit

                    self.triggerAction(self.SliderMove)
                    self.setRepeatAction(self.SliderNoAction)
                    self.setSliderDown(True)

                    break

            if self.active_slider < 0:
                self.pressed_control = QStyle.SC_SliderHandle
                self.click_offset = self.__pixelPosToRangeValue(self.__pick(event.pos()))
                self.triggerAction(self.SliderMove)
                self.setRepeatAction(self.SliderNoAction)
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if self.pressed_control != QStyle.SC_SliderHandle:
            event.ignore()
            return

        event.accept()
        new_pos = self.__pixelPosToRangeValue(self.__pick(event.pos()))
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        if self.active_slider < 0:
            offset = new_pos - self.click_offset
            self._high += offset
            self._low += offset
            if self._low < self.minimum():
                diff = self.minimum() - self._low
                self._low += diff
                self._high += diff
            if self._high > self.maximum():
                diff = self.maximum() - self._high
                self._low += diff
                self._high += diff
        elif self.active_slider == 0:
            if new_pos >= self._high:
                new_pos = self._high - 1
            self._low = new_pos
        else:
            if new_pos <= self._low:
                new_pos = self._low + 1
            self._high = new_pos

        self.click_offset = new_pos
        self.update()

    def mouseReleaseEvent(self, event):
        self.level_change()

    def __pick(self, pt):
        if self.orientation() == QtCore.Qt.Horizontal:
            return pt.x()
        else:
            return pt.y()


    def __pixelPosToRangeValue(self, pos):
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        style = QApplication.style()

        gr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderGroove, self)
        sr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderHandle, self)

        if self.orientation() == QtCore.Qt.Horizontal:
            slider_length = sr.width()
            slider_min = gr.x()
            slider_max = gr.right() - slider_length + 1
        else:
            slider_length = sr.height()
            slider_min = gr.y()
            slider_max = gr.bottom() - slider_length + 1

        return style.sliderValueFromPosition(self.minimum(), self.maximum(),
                                             pos-slider_min, slider_max-slider_min,
                                             opt.upsideDown)

class ModelButton(QPushButton):
    def __init__(self, parent, model_name, text):
        super().__init__()
        self.setEnabled(True)
        self.setCheckable(True)
        self.setStyleSheet(parent.styleCheckable)
        self.setText(model_name)
        self.setToolTip(text)
        self.setFont(parent.smallfont)
        #self.clicked.connect(lambda: self.press(parent))
        self.model_name = model_name
        
    def press(self, parent):
        for i in range(len(parent.StyleButtons)):
            parent.StyleButtons[i].setStyleSheet(parent.styleUnpressed)
        self.setStyleSheet(parent.stylePressed)
        parent.compute_model(self.model_name)


class CheckableComboBox(QComboBox):
    def __init__(self):
        super().__init__()
        self._changed = False

        self.view().pressed.connect(self.handleItemPressed)

    def setItemChecked(self, index, checked=False):
        item = self.model().item(index, self.modelColumn()) # QStandardItem object

        if checked:
            item.setCheckState(Qt.Checked)
        else:
            item.setCheckState(Qt.Unchecked)

    def handleItemPressed(self, index):
        item = self.model().itemFromIndex(index)

        if item.checkState() == Qt.Checked:
            item.setCheckState(Qt.Unchecked)
        else:
            item.setCheckState(Qt.Checked)
        self._changed = True


    def hidePopup(self):
        if not self._changed:
            super().hidePopup()
        self._changed = False

    def itemChecked(self, index):
        item = self.model().item(index, self.modelColumn())
        return item.checkState() == Qt.Checked

    def getCheckedItems(self):
        checkedItems = [self.itemText(i) for i in range(self.count()) if self.itemChecked(i)]
        return checkedItems






    



