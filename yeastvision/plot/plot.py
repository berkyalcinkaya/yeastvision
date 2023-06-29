from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QApplication, QRadioButton, QButtonGroup, QSlider, QStyle, QStyleOptionSlider, QMainWindow, QPushButton, QDialog,
                            QDialogButtonBox, QLineEdit, QFormLayout, QMessageBox,
                             QFileDialog, QVBoxLayout, QCheckBox, QFrame, QSpinBox, QLabel, QWidget, QComboBox, QTableWidget, QSizePolicy, QGridLayout, QHBoxLayout)
import pyqtgraph as pg
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from yeastvision.plot.cell_table import TableModel
from yeastvision.plot.types import SingleCellUpdatePlot, HeatMap, PopulationPlot
from yeastvision.track.data import PopulationReplicate, Population
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class PlotProperty():
    def __init__(self, parent, plotType, setName, property, populationName = None):
        self.parent = parent
        self.plotType = plotType
        self.labelIdx = self.getLabelIdx(setName)
        self.prop = property
        self.population = populationName
    
    def getPlotName(self):
        labelName = self.getLabelName()
        if self.populationName:
            return "f{labelName}-{self.population}-{self.prop}"
        else:
            return "f{labelName}-{self.prop}"
    
    def getLabelIdx(self, setName):
        return self.parent.labelSelect.items().index(setName)
    
    def getLabelName(self):
        return self.labelSelect.items()[self.labelIdx]
    
    def get_timeseries_obj(self):
        return self.parent.cellData[self.labelIdx]
    
    @staticmethod
    def cluster_property(plot_property_list):
        by_property = {}
        for plot_property_obj in plot_property_list:
            prop = plot_property_obj.prop
            if prop not in by_property:
                by_property[prop] = [plot_property_obj]
            else:
                by_property[prop].append(plot_property_obj)
        return by_property

    @staticmethod
    def cluster_population(plot_property_list):
        by_population = {}
        for plot_property_obj in plot_property_list:
            pop = plot_property_obj.population
            if pop not in by_population:
                by_population[pop] = [plot_property_obj]
            else:
                by_population[pop].append(plot_property_obj)
        return by_population

    

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
        pg.setConfigOption('imageAxisOrder', 'row-major')
        self.cmap = pg.colormap.get("jet", source = 'matplotlib')
        self.setData()
        
        self.propDict = propDict


        if self.hasPlots():
            self.area = DockArea()
            self.hLayout.addWidget(self.area, 7)
            self.docks = {} # key = name of dock, value = (dock object, widget contained in dock)
            if self.hasSingleCellPlots():
                self.addSingleCellPlots()
            if self.hasHeatmaps():
                self.addHeatmaps()
            if self.hasPopulationPlots():
                self.addPopulationsPlots()
            self.setGeometry(500, 100, 1500, 1500)
    
    def hasPlots(self):
        return self.hasSingleCellPlots() or self.hasHeatmaps() or self.hasPopulationPlots()

    def hasSingleCellPlots(self):
        return "single" in self.propDict and self.propDict["single"]
    
    def hasMplPlots(self):
        return self.hasPopulationPlots()
    
    def hasHeatmaps(self):
        return "heatmap" in self.propDict and self.propDict["heatmap"]
    
    def hasPopulationPlots(self):
        return "population" in self.propDict and self.propDict["population"]

    def updateAll(self):
        if self.hasSingleCellPlots():
            self.updateSingleCellPlots()
        if self.hasHeatmaps():
            self.updateHeatMaps()
        if self.hasPopulationPlots():
            self.updatePopulationPlots

    
    def addSingleCellPlots(self):
        self.singleCellPlots = []
        single_cell_prop_obj = self.propDict["single"]
        clustered_by_property = PlotProperty.cluster_property(single_cell_prop_obj)
        for property in clustered_by_property:
            timeseries = [prop.get_timeseries_obj() for prop in clustered_by_property[property]]
            self.singleCellPlots.append(SingleCellUpdatePlot(self.parent, property, timeseries, self.parent.selectedCells))
        
 
        dockName = f"single plots"
        newDock = Dock(dockName, closable = True)
        dockLayout = DockArea()
        newDock.addWidget(dockLayout)
        self.area.addDock(newDock, "bottom")
        

        for i, plot in enumerate(self.singleCellPlots):
            if i == 0:
                position = "bottom"
                relativeTo = None
            else:
                position = "above"
                plotRelativeTo = self.singleCellPlots[i-1]
                relativeTo = self.docks[plotRelativeTo]

            name = f"{plot.property}"
            plotDock = Dock(name, closable = True)
            plotDock.addWidget(plot.getPlotWidget())
            self.docks[plot] = (plotDock)
  
            dockLayout.addDock(plotDock, position, relativeTo)
    
    def addHeatmaps(self):
        self.heatMaps = []
        heatmap_prop_obj = self.propDict["heatmap"]
        clustered_by_property = PlotProperty.cluster_property(heatmap_prop_obj)
        for property, properties in clustered_by_property.items():
            timeseries = [prop.get_timeseries_obj() for prop in properties]
            populations = [prop.population for prop in properties]
            for ts, population in zip(timeseries, populations):
                self.heatMaps.append(HeatMap(self, property, population, ts))
    
        dockName = f"heatmaps"
        newDock = Dock(dockName, closable = True)
        dockLayout = DockArea()
        newDock.addWidget(dockLayout)
        self.area.addDock(newDock, "bottom")

        for i, plot in enumerate(self.heatMaps):
            if i == 0:
                position = "bottom"
                relativeTo = None
            else:
                position = "above"
                plotRelativeTo = self.heatMaps[i-1]
                relativeTo = self.docks[plotRelativeTo]

            name = f"{plot.property}"
            plotDock = Dock(name, closable = True)
            plotDock.addWidget(plot.getPlotWidget())
            self.docks[plot] = (plotDock)

            dockLayout.addDock(plotDock, position, relativeTo)
    
    def addPopulationsPlots(self):
        self.populationPlots = []
        population_prop_obj = self.propDict["population"]
        clustered_by_property = PlotProperty.cluster_property(population_prop_obj)
        for property, properties in clustered_by_property.items():
            timeseries = [prop.get_timeseries_obj() for prop in properties]
            populations = [prop.population for prop in properties]
            for ts, population in zip(timeseries, populations):
                self.populationPlots.append(PopulationPlot(self, property, ts, population))
    
        dockName = f"population plots"
        newDock = Dock(dockName, closable = True)
        dockLayout = DockArea()
        newDock.addWidget(dockLayout)
        self.area.addDock(newDock, "bottom")

        for i, plot in enumerate(self.populationPlots):
            if i == 0:
                position = "bottom"
                relativeTo = None
            else:
                position = "above"
                plotRelativeTo = self.populationPlots[i-1]
                relativeTo = self.docks[plotRelativeTo]

            name = f"{plot.property}"
            plotDock = Dock(name, closable = True)
            plotDock.addWidget(plot.getPlotWidget())
            self.docks[plot] = (plotDock)

            dockLayout.addDock(plotDock, position, relativeTo)
    
    def addMplPlots():
        pass

    def updateSingleCellPlots(self):
        for plot in self.singleCellPlots:
            plot.update()
        

    def updatePopulationPlots(self):
        for plot in self.populationPlots:
            plot.update()
    
    def updateHeatMaps(self):
        for plot in self.heatMaps:
            plot.update()

    def setData(self):
        self.data = self.parent.getCellData()

    def getDataIdx(self, population):
        try:
            idx = self.parent.labelSelect.items().index(population)
            print(idx)
            return idx
        except ValueError:
            return None

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