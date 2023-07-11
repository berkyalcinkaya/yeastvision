from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import (QApplication,QSlider, QStyle, QStyleOptionSlider, QPushButton,QCheckBox, QComboBox, QFrame)
from PyQt5.QtCore import Qt


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






    



