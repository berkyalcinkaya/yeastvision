from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from pyqtgraph import Point
import numpy as np
from skimage.morphology import disk, dilation, erosion
from skimage.draw import line_aa
from PIL import Image
from PIL import ImageDraw as ID
import math



class ImageDraw(pg.ImageItem):
    sigImageChanged = QtCore.pyqtSignal()

    def __init__(self, image = None, viewbox = None, parent = None, lut = None, **kargs):
        super(ImageDraw, self).__init__()
        self.levels = np.array([0,255])
        self.lut = lut
        self.axisOrder = "row-major"
        self.removable = False

        self.colorNum = None

        self.parent = parent
        self.setDrawKernel(kernel_size = self.parent.brush_size)
        self.parent.strokes = []
        self.parent.in_addregion_stroke = False
        self.brushColorSelected = False
        self.parent.currStroke = []
        
        self.strokeAppended = True
    
    def turnOnDraw(self):
        self.setDefaultColorNum()
    
    def turnOffDraw(self):
        self.colorNum = None
        self.brushColorSelected = False
    
    def setDefaultColorNum(self):
        if self.parent.drawType == "Eraser":
            self.colorNum = 0
        else:
            self.colorNum = self.parent.currMask.max() + 1

    def setDrawKernel(self, kernel_size=3):
        bs = kernel_size
        kernel = np.ones((bs,bs), np.uint8)
        self.drawKernel = kernel
        self.drawKernelCenter = [int(np.floor(kernel.shape[0]/2)),
                                 int(np.floor(kernel.shape[1]/2))]
    
    def mouseClickEvent(self, ev):
        if not self.parent.imLoaded or not self.parent.maskLoaded:
            if self.parent.imLoaded and not self.parent.maskLoaded: #and (self.parent.drawType == "Brush" or self.parent.drawType == "Eraser" or self.parent.drawType == "Outline"):
                self.parent.showError("No Mask Loaded -- Drawing Disabled. Either Segment Current Images or Go to `Edit>Add Blank Mask` in the menu bar to begin drawing")
            return
        
        try:
            cellNum = self.parent.currMask[int(ev.pos().y()), int(ev.pos().x())]
        except IndexError:
            return
        
        if ev.button() == QtCore.Qt.LeftButton and self.parent.maskOn:
            if cellNum!=0:
                self.parent.selectCell(ev.pos())
        
        # mask must be on for drawing
        elif ev.button() == QtCore.Qt.RightButton and not self.parent.prob_or_flow_on() and self.parent.maskOn:
            
            if self.parent.drawType != "":
                
                # different brush modes
                if self.parent.drawType == "Brush":
                    self.colorNum = self.parent.selectedCells[0] if self.parent.selectedCells else self.parent.currMask.max()+1
                    self.drawAt(ev.pos())
                
                elif self.parent.drawType == "Outline":
                    if not self.parent.in_addregion_stroke:
                        self.colorNum = self.parent.selectedCells[0] if self.parent.selectedCells and self.parent.selectedCells[0] != 0 else self.parent.currMask.max()+1
                        ev.accept()
                        self.create_start(ev.pos())
                        self.parent.in_addregion_stroke = True
                        self.drawAt(ev.pos(), ev)
                        self.strokeAppended = False
                    else:
                        ev.accept()
                        self.end_stroke()
                        self.parent.in_addregion_stroke = False

                elif self.parent.drawType == "Eraser" and not self.brushColorSelected:
                    self.colorNum = 0
                    self.drawAt(ev.pos())
                elif self.brushColorSelected:
                    self.drawAt(ev.pos())
                
                self.brushColorSelected = True

    def mouseDragEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self.parent.view.mouseDragEvent(ev)
            ev.ignore()
            return
        
        
        if not self.parent.imLoaded or not self.parent.maskLoaded:
            if self.parent.imLoaded and not self.parent.maskLoaded: #and (self.parent.drawType == "Brush" or self.parent.drawType == "Eraser"):
                self.parent.showError("No Mask Loaded. Drawing Disabled. Either Segment Current Images or Go to Edit>Add Blank Mask to begin drawing")
            return
        
        # mask must be on for drawing
        if ev.button() ==  QtCore.Qt.RightButton and (self.parent.drawType == "Brush" or self.parent.drawType == "Eraser"):
            ev.accept()
            if self.parent.drawType == "Eraser":
                self.colorNum = 0
            else:
                self.colorNum = self.parent.selectedCells[0] if self.parent.selectedCells else self.colorNum

            if not self.parent.prob_or_flow_on() and self.parent.maskOn:
                if ev.isStart():
                    self.drawAt(ev.pos())
                elif ev.isFinish():
                    return
                else:
                    self.paintBrush(ev.lastPos(),ev.pos())
                return
    
    def hoverEvent(self, ev):
        x,y = None, None
        try:
            x = int(ev.pos().x())
            y = int(ev.pos().y())
            try:
                maskVal  = self.parent.currMask[y,x]
                if maskVal == 0:
                    val = self.parent.currIm[y,x]
                else:
                    val = maskVal
            except IndexError:
                val = None
        except AttributeError:
            val = None
        self.parent.updateDataDisplay(x = x, y = y, val = val)

        if self.parent.in_addregion_stroke:
            self.drawAt(ev.pos())
            if self.is_at_start(ev.pos()):
                self.end_stroke()
                self.parent.in_addregion_stroke = False
        return
    
    def create_start(self, pos):
        self.scatter = pg.ScatterPlotItem([pos.x()], [pos.y()], pxMode=False,
                                        pen=pg.mkPen(color=(255,0,0), width=self.parent.brush_size),
                                        size=max(3*2, self.parent.brush_size*1.8*2), brush=None)
        self.parent.view.addItem(self.scatter)
    
    def is_at_start(self, pos):
        if len(self.parent.currStroke) > 16:
            x0,y0 = self.parent.currStroke[0]
            x1,y1 = pos.x(), pos.y()
            d = (x1-x0)**2 + (y1-y0)**2
            d = math.sqrt(d)
            return d<1
        else:
            return False

    def addSet(self):
        if len(self.parent.currStroke)>3:
            ny, nx = self.parent.label().x(), self.parent.label().y()
            img = Image.new('L', (ny, nx), 0)
            ID.Draw(img).polygon(self.parent.currStroke, outline=1, fill=1)
            polygon = np.array(img).astype(bool)
            self.parent.currMask[polygon] = self.colorNum
            self.setImage(self.parent.maskColors[self.parent.currMask], autolevels  = False)
            #self.parent.label().save()
            self.parent.strokes.append((polygon, self.colorNum))
            self.parent.addRecentDrawing()

            if self.parent.contourOn:
                self.parent.drawContours()
    
    def end_stroke(self):
        self.parent.view.removeItem(self.scatter)
        if not self.strokeAppended:
            # self.parent.addRegionStrokes.append(self.currStroke)
            # self.strokeAppended = True
            # self.currStroke = np.array(self.currStroke)
            # ioutline = self.currStroke[:,3]==1
            # self.parent.currPointSet.extend(list(self.currStroke[ioutline]))
            self.addSet()
            self.parent.currStroke = []
            self.colorNum = 0
            self.parent.in_addregion_stroke = False

    
    def enterEvent(self, ev):
        self.hoverEvent(ev)
    
    def leaveEvent(self, ev):
        self.parent.updateDataDisplay(x = None, y = None, val =None)
        
    def paintBrush(self,pos1, pos2):
        pos1y,pos1x = (int(pos1.y()), int(pos1.x()))
        pos2y,pos2x = (int(pos2.y()), int(pos2.x()))

        selem = disk(self.parent.brush_size-2)

        rr,cc,_ = line_aa(pos2y,pos2x, pos1y, pos1x)

        to_change = np.zeros(self.parent.currMask.shape, dtype = bool)
        to_change[rr,cc] = True
        to_change = dilation(to_change, selem)
        self.parent.currMask[to_change] = self.colorNum

        self.parent.strokes.append((to_change, self.colorNum))
        
        self.setImage(self.parent.maskColors[self.parent.currMask])
        #self.parent.label().save()

        if self.parent.drawType == "Outline":
            # for ky,y in enumerate(np.arange(ty[0], ty[1], 1, int)):
            #     for kx,x in enumerate(np.arange(tx[0], tx[1], 1, int)):
            #         iscent = np.logical_and(kx==kcent[0], ky==kcent[1])
            #         self.currStroke.append([self.parent.maskZ, x, y, iscent])
            self.parent.currStroke.append((int(pos2.x()), int(pos2.y())))
        self.parent.addRecentDrawing()

    
    def drawAt(self, p, ev=None):
        pos = [int(p.y()), int(p.x())]
        dk = self.drawKernel
        kc = self.drawKernelCenter
        sx = [0,dk.shape[0]]
        sy = [0,dk.shape[1]]
        tx = [pos[0] - kc[0], pos[0] - kc[0]+ dk.shape[0]]
        ty = [pos[1] - kc[1], pos[1] - kc[1]+ dk.shape[1]]
        kcent = kc.copy()
        if tx[0]<=0:
            sx[0] = 0
            sx[1] = kc[0] + 1
            tx    = sx
            kcent[0] = 0
        if ty[0]<=0:
            sy[0] = 0
            sy[1] = kc[1] + 1
            ty    = sy
            kcent[1] = 0
        if tx[1] >= self.parent.label().y()-1:
            sx[0] = dk.shape[0] - kc[0] - 1
            sx[1] = dk.shape[0]
            tx[0] = self.parent.label().y() - kc[0] - 1
            tx[1] = self.parent.label().y()
            kcent[0] = tx[1]-tx[0]-1
        if ty[1] >= self.parent.label().x()-1:
            sy[0] = dk.shape[1] - kc[1] - 1
            sy[1] = dk.shape[1]
            ty[0] = self.parent.label().x() - kc[1] - 1
            ty[1] = self.parent.label().x()
            kcent[1] = ty[1]-ty[0]-1


        ts = (slice(tx[0],tx[1]), slice(ty[0],ty[1]))
        #ss = (slice(sx[0],sx[1]), slice(sy[0],sy[1]))
        self.parent.currMask[ts] = self.colorNum

        self.parent.strokes.append((ts, self.colorNum))

        self.setImage(self.parent.maskColors[self.parent.currMask], autolevels  = False)
        #self.parent.label().save()
        if self.parent.drawType == "Brush":
            self.parent.addRecentDrawing()
        if self.parent.drawType == "Outline":
            # for ky,y in enumerate(np.arange(ty[0], ty[1], 1, int)):
            #     for kx,x in enumerate(np.arange(tx[0], tx[1], 1, int)):
            #         iscent = np.logical_and(kx==kcent[0], ky==kcent[1])
            #         self.currStroke.append([self.parent.maskZ, x, y, iscent])
            self.parent.currStroke.append((int(p.x()), int(p.y())))

    
    def addMask(self, mask, color, update = True):
        mask = mask>0
        self.image[mask,:] = color
        if update:
            self.updateImage()

  
class ViewBoxNoRightDrag(pg.ViewBox):
    def __init__(self, parent=None, border=None, lockAspect=False, enableMouse=True, invertY=False, enableMenu=True, name=None, invertX=False):
        pg.ViewBox.__init__(self, None, border, lockAspect, enableMouse,
                            invertY, enableMenu, name, invertX)
        self.parent = parent
        self.axHistoryPointer = -1

    def keyPressEvent(self, ev):
        """
        This routine should capture key presses in the current view box.
        The following events are implemented:
        +/= : moves forward in the zooming stack (if it exists)
        - : moves backward in the zooming stack (if it exists)
        """
        ev.accept()
        if ev.text() == '-':
            self.scaleBy([1.1, 1.1])
        elif ev.text() in ['+', '=']:
            self.scaleBy([0.9, 0.9])
        else:
            ev.ignore()

    def mouseDragEvent(self, ev, axis=None):
        ## if axis is specified, event will only affect that axis.
        if self.parent is None or (self.parent is not None and not self.parent.in_addregion_stroke):
            ev.accept()  ## we accept all buttons

            pos = ev.pos()
            lastPos = ev.lastPos()
            dif = pos - lastPos
            dif = dif * -1

            ## Ignore axes if mouse is disabled
            mouseEnabled = np.array(self.state['mouseEnabled'], dtype=float)
            mask = mouseEnabled.copy()
            if axis is not None:
                mask[1-axis] = 0.0

            ## Scale or translate based on mouse button
            if ev.button() & (QtCore.Qt.LeftButton | QtCore.Qt.MidButton):
                if self.state['mouseMode'] == pg.ViewBox.RectMode:
                    if ev.isFinish():  ## This is the final move in the drag; change the view scale now
                        self.rbScaleBox.hide()
                        ax = QtCore.QRectF(Point(ev.buttonDownPos(ev.button())), Point(pos))
                        ax = self.childGroup.mapRectFromParent(ax)
                        self.showAxRect(ax)
                        self.axHistoryPointer += 1
                        self.axHistory = self.axHistory[:self.axHistoryPointer] + [ax]
                    else:
                        ## update shape of scale box
                        self.updateScaleBox(ev.buttonDownPos(), ev.pos())
                else:
                    tr = dif*mask
                    tr = self.mapToView(tr) - self.mapToView(Point(0,0))
                    x = tr.x() if mask[0] == 1 else None
                    y = tr.y() if mask[1] == 1 else None

                    self._resetTarget()
                    if x is not None or y is not None:
                        self.translateBy(x=x, y=y)
                    self.sigRangeChangedManually.emit(self.state['mouseEnabled'])