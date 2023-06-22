from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QApplication, QRadioButton, QButtonGroup, QSlider, QStyle, QStyleOptionSlider, QMainWindow, QPushButton, QDialog,
                            QDialogButtonBox, QLineEdit, QFormLayout, QMessageBox,
                             QFileDialog, QVBoxLayout, QCheckBox, QFrame, QSpinBox, QLabel, QWidget, QComboBox, QTableWidget, QSizePolicy, QGridLayout, QHBoxLayout)
import pyqtgraph as pg
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from yeastvision.plot.cell_table import TableModel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class CustomizePlot(QWidget):
    def __init__(self):
        pass

class EvalWindow(QWidget):
    def __init__(self, parent, data_dict, IOUs):
        '''
        params
        data_dict - key: name of the plot (precision, recall, F1, accuracy), values - list of tuple for each set of labels being 
        evaluated where tup1 is the name and tup2 is data vs IOU score
        '''
        super(EvalWindow, self).__init__()
        self.setWindowTitle("Evaluation Window")

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.data = data_dict
        print(self.data)

        self.numPlots = len(data_dict)
        self.evalNames = [n for n,_ in list(self.data.values())[0]]
        self.numToEval = len(self.evalNames)
        self.xVals = IOUs
        print(IOUs)

        # Create a table widget to display data
        self.tableLayout = QVBoxLayout()
        self.table_widgets = []

        self.toolbar = NavigationToolbar(self.canvas, self)


        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        #layout.addWidget(self.tableLayout)

        self.setLayout(layout)

        self.setData()
        self.setGeometry(400,400,500,500)
        self.canvas.draw()
        self.show()

    def setData(self):
        i = 0
        for plotType, data in self.data.items():
            self.producePlot(data, plotType, i)
            i+=1

    
    # def produceTable(self, title, data):
    #     table_widget = QTableWidget()
    #     table_widget.setColumnCount(self.numToEval)
    #     table_widget.setRowCount(len(data[0]))

    #     for col, line_data in enumerate(data):
    #         for row, value in enumerate(line_data):
    #             item = QTableWidgetItem(str(value))
    #             table_widget.setItem(row, col, item)

    #     return table_widgets

    
    def producePlot(self, datas, title, col):
        ax = self.figure.add_subplot(2, 2, col + 1)
        for name, data in datas:
            ax.plot(self.xVals, data, 'o-', label = name)
            ax.set_aspect('auto')
            ax.set_xbound(lower = 0.45,upper = 1.05)
            ax.set_xticks([0.5,0.6,0.7,0.8,0.9,1])

            if col == 0:
                ax.legend()
                ax.set_xlabel("IOU Matching Threshold")
        ax.set_title(title)
        plt.tight_layout()




class PlotWindow(QWidget):
    def __init__(self, parent, propDict):
        super().__init__()
        self.parent = parent
        pg.setConfigOption('imageAxisOrder', 'row-major')
        self.hLayout = QHBoxLayout()
        self.setLayout(self.hLayout)
        self.createTable()
        self.propDict = propDict
        pg.setConfigOption('imageAxisOrder', 'row-major')
        self.cmap = pg.colormap.get("jet", source = 'matplotlib')

        # print(propDict)
        # print(propDict["single"])
        self.singleCellPopulations = self.getSingleCellPopulations(propDict["single"]) if "single" in propDict else []
        print(self.singleCellPopulations)
        self.setData()

        self.hasPlots = bool(propDict)

        self.propDict = propDict
        if self.propDict:
            self.area = DockArea()
            self.hLayout.addWidget(self.area, 7)
            self.docks = {} # key = name of dock, value = (dock object, widget contained in dock)
            
            for plotType, props in self.propDict.items():
                if props:

                    # dock
                    dockName = f"{plotType} plots"
                    newDock = Dock(dockName, closable = True)
                    dockLayout = DockArea()
                    newDock.addWidget(dockLayout)
                    self.docks[plotType] = (newDock, dockLayout)
                    self.area.addDock(newDock, "bottom")

                    for i, prop in enumerate(props):
                        if i == 0:
                            position = "bottom"
                            relativeTo = None
                        else:
                            position = "above"
                            plotRelativeTo = props[i-1]
                            relativeTo = self.docks[f"{plotType} {plotRelativeTo}"][0]

                        if "heatmap" in plotType:
                            name = f'{prop}'
                            imDock = Dock(name, closable = True)
                            imData = self.getHeatMap(prop)
                            view  = pg.PlotWidget(title = name)
                            im = pg.ImageItem(colorMap = self.cmap)
                            im.setImage(imData, autoLevels = True)
                            view.addItem(im)
                            imDock.addWidget(view)
                            self.docks[f"{plotType} {prop}"] = (imDock,im)
                            dockLayout.addDock(imDock, position, relativeTo)
                        else:
                            name = f"{prop}"
                            plotDock = Dock(name, closable = True)
                            newPlot = pg.PlotWidget(title=name)
                            
                            data = self.getData(plotType, prop)
                            if type(data) is dict:
                                newPlot.addLegend()
                                i = 0
                                for cellnum, dataset in data.items():
                                    color = parent.cmaps[0][i][0:-1]
                                    newPlot.plot(dataset, name = str(cellnum), pen = tuple(color))
                                    i+=1
                            else:
                                newPlot.plot(data)

                            self.docks[f"{plotType} {prop}"] = (plotDock,newPlot)

                            plotDock.addWidget(newPlot)    
                            dockLayout.addDock(plotDock, position, relativeTo)
        if self.hasPlots:
            self.setGeometry(500, 100, 1500, 1500)
    
    def getSingleCellPopulations(self, singleCelProps):
        return [prop.split("-")[0] for prop in singleCelProps]

    def updateSingleCellPlots(self):
        for plotName, widgets in self.docks.items():
            if " " in plotName:
                s = plotName.split(" ")
                plotType, prop = s[0], s[1]
                if "single" in plotType:
                    _, newPlot = widgets
                    newPlot.clear()
                    data = self.getData(plotType, prop) 

                    if type(data) is dict:     
                        newPlot.addLegend()
                        i = 0
                        for cellnum, dataset in data.items():
                            color = self.parent.cmaps[0][i][0:-1]
                            newPlot.plot(dataset, name = str(cellnum), pen = tuple(color))
                            i+=1
                    else:
                        newPlot.plot(data)

    def updatePopulationPlots(self):
        for plotName, widgets in self.docks.items():
            if " " in plotName:
                s = plotName.split(" ")
                plotType, prop = s[0], s[1]
                if "population" in plotType:
                    _, im = widgets
                    imData = self.getHeatMap(prop)
                    im.setImage(imData, autoLevels = True)
    
    def updateHeatMaps(self):
        for plotName, widgets in self.docks.items():
            if " " in plotName:
                s = plotName.split(" ")
                plotType, prop = s[0], s[1]
                if "heatmap" in plotType:
                    _, newPlot = widgets
                    newPlot.clear()
                    data = self.getData(plotType, prop)        
                    newPlot.plot(data)

    def setData(self):
        self.data = self.parent.getCellData()

    def getDataIdx(self, population):
        try:
            idx = self.parent.labelSelect.items().index(population)
            print(idx)
            return idx
        except ValueError:
            return None

    def getDataToUse(self, prop):
        if self.data:
            idx = self.getDataIdx(prop)
            if idx is not None:
                return self.data[idx]
            else:
                return None
        else:
            return None

    def getData(self, cellType, property):
        s = property.split("-")
        population, property = s[0],"-".join(s[1:])
        dataToUse = self.getDataToUse(population)
        if type(dataToUse) is pd.core.frame.DataFrame:
            if cellType == "population":
                cellNum = "population"
                data = (dataToUse.loc[dataToUse["labels"] == cellNum][property]).tolist()[0]
            elif cellType == "single" and self.parent.selectedCells:
                data = {}
                for cellNum in self.parent.selectedCells:
                    data[cellNum] = (dataToUse.loc[dataToUse["labels"] == cellNum][property]).tolist()[0]
            elif cellType == "single" and not (self.parent.selectedCells):
                data =  [0]*10
            return data
        else:
            return [0]*10
    
    def getHeatMap(self, property):
        dataToUse = self.getDataToUse("heatmap")
        if type(dataToUse) is pd.core.frame.DataFrame:
            return np.array(dataToUse[property][0:-1].to_list())
        else:
            return np.zeros((10,10))

    def updatePlots(self):
        self.updateSingleCellPlots()
        self.updatePopulationPlots()
        self.updateHeatMaps()

    def createTable(self):
        # create table with dummy data
        self.table = QtWidgets.QTableView()
        self.model = TableModel(self.parent.getCellDataAbbrev(), self.parent)
        self.table.setModel(self.model)

        self.table.selectionModel().currentChanged.connect(self.model.handleselect)

        # configure the table appearance
        self.table.verticalHeader().setVisible(False) # turn off vertical header
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch) # stretch last section to take full width of allottedd area
        self.table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.table.resizeColumnsToContents()
        self.hLayout.addWidget(self.table, 4)
        
    def resizeTableToContents(self):
        vh = self.table.verticalHeader()
        hh = self.table.horizontalHeader()
        size = QtCore.QSize(hh.length(), vh.length())  # Get the length of the headers along each axis.
        size += QtCore.QSize(vh.size().width(), hh.size().height())  # Add on the lengths from the *other* header
        size += QtCore.QSize(20, 20)  # Extend further so scrollbars aren't shown.
        self.resize(size)
    
    def closeEvent(self, event):
        self.parent.plotButton.setCheckState(False)