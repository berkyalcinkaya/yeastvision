from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QApplication, QRadioButton, QButtonGroup, QSlider, QStyle, QStyleOptionSlider, QMainWindow, QPushButton, QDialog,
                            QDialogButtonBox, QLineEdit, QFormLayout, QMessageBox,
                             QFileDialog, QVBoxLayout, QCheckBox, QFrame, QSpinBox, QLabel, QWidget, QComboBox, QTableWidget, QSizePolicy, QGridLayout, QHBoxLayout)
import pyqtgraph as pg
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from yeastvision.plot.cell_table import TableModel
from yeastvision.plot.types import SingleCellUpdatePlot, HeatMap, PopulationPlot, SingleFrameUpdatePlot
from yeastvision.track.data import PopulationReplicate, Population
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

class HeatmapCanvas(FigureCanvas):
    def __init__(self, intervals, max_frame, interp_intervals, parent=None):
        fig, self.ax = plt.subplots(figsize=(10, 4))  # Increased figure size for better layout
        super().__init__(fig)
        self.setParent(parent)

        self.create_1d_heatmap(intervals, max_frame, interp_intervals)

    def create_1d_heatmap(self, intervals, max_frame, interp_intervals):
        # Create an array for the heatmap
        heatmap = np.zeros(max_frame)

        for interval in interp_intervals:
            start = interval['start']
            stop = interval['stop']
            interp = int(interval['interp'])
            interp_factor = 2 ** interp

            i = start
            while i < stop:
                heatmap[i+1:(i+1) + (interp_factor-1)] = interp
                i = (i+1) + (interp_factor-1)
        #print("Heatmap values:", heatmap)  # Debugging line to check heatmap values

        # Create a modern-looking colormap
        colors = [
            (1, 1, 1, 0),       # Transparent for 0 interpolation
            (0.8, 0.8, 1, 1),   # Light blue for 2x
            (0.6, 0.6, 1, 1),   # Medium blue for 4x
            (0.4, 0.4, 1, 1),   # Dark blue for 8x
            (0.2, 0.2, 1, 1),   # Darker blue for 16x
        ]
        cmap = LinearSegmentedColormap.from_list("interpolation_cmap", colors)
        norm = BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)

        # Create the 1D heatmap
        heatmap_image = heatmap.reshape(1, -1)

        im = self.ax.imshow(heatmap_image, aspect='auto', cmap=cmap, norm=norm, extent=[0, max_frame, 0, 1])

        # Customize the plot
        self.ax.set_yticks([])
        self.ax.set_xlabel('Frame')
        self.ax.set_title('Interpolation Levels')

        # Draw lines beneath the graph for each interval
        for interval in interp_intervals:
            start = interval['start']
            stop = interval['stop']
            interp = int(interval['interp'])
            interp_factor = 2 ** interp

            self.ax.hlines(-0.1, start, stop, colors=cmap(norm(interp)), lw=4)
            self.ax.vlines(start, -0.15, -0.05, colors='black', lw=1)
            self.ax.vlines(stop, -0.15, -0.05, colors='black', lw=1)

        # Add a custom horizontal legend below the x-axis
        handles = [
            plt.Line2D([0], [0], color=cmap(norm(1)), lw=4, label='2x'),
            plt.Line2D([0], [0], color=cmap(norm(2)), lw=4, label='4x'),
            plt.Line2D([0], [0], color=cmap(norm(3)), lw=4, label='8x'),
            plt.Line2D([0], [0], color=cmap(norm(4)), lw=4, label='16x')
        ]
        legend = self.ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.3),
                                ncol=4, title="Interpolation Levels")

        plt.tight_layout(pad=3.0)  # Ensure the layout fits properly
        self.draw()

class InterpolationHeatmapWindow(QWidget):
    def __init__(self, parent, intervals, max_frame, interp_intervals):
        super().__init__()
        self.parent = parent
        self.setWindowTitle("Interpolation Heatmap Viewer")

        layout = QVBoxLayout(self)

        label = QLabel(f"Interpolation Intervals (Original Movie) {str(intervals)}")
        layout.addWidget(label)

        self.heatmap_canvas = HeatmapCanvas(intervals, max_frame, interp_intervals, 
                                            parent = self)
        layout.addWidget(self.heatmap_canvas)


def plot_tree(matrix, selected_cells=[]):
    '''
    PARAMETERS
    -------------
    matrix: np.ndarray
        matrix that encodes lineage and cell info as follows:
            column 0: cell idx (sorted numerically, 1-indexed)
            column 1: cell birth frame (0 indexed)
            column 2: cell death frame (0 indexed)
            column 3: mother idx (-1 if has no mother)
        lineage tree places initial, seed cells on seperate frames
    
    selected cell: list (optional, default [])
        cells whose daughters should be highlighted
    

    RETURNS
    ---------------
    figure: matplotlib.pyplot.Figure
        can be embedded into a GUI window or shown with matplotlib.pylot.show
    
    '''
    fig, ax = plt.subplots(figsize=(10, 6))
    y_positions = {}
    current_y = 0  # Starting position

    # Bright color list for selected cells
    bright_colors = [
        '#FF0000',  # Bright red
        '#00FF00',  # Bright green
        '#0000FF',  # Bright blue
        '#FF00FF',  # Bright magenta
        '#00FFFF',  # Bright cyan
        '#FFFF00',  # Bright yellow
    ]

    default_horizontal_color = 'blue'
    default_vertical_color = 'red'

    if selected_cells:
        cell_colors = {selected_cells[i]: bright_colors[i % len(bright_colors)] for i in range(len(selected_cells))}
    else:
        cell_colors = {}

    def plot_lineage(mother, y):
        daughters = matrix[matrix[:, 3] == mother]
        daughters = daughters[daughters[:, 1].argsort()[::-1]]

        mother_color = cell_colors.get(mother, default_horizontal_color)

        for daughter in daughters:
            cell_index, birth_frame, death_frame, _ = daughter
            y += 1

            while y in y_positions.values():
                y += 1

            horizontal_color = mother_color if cell_index in cell_colors or mother_color != default_horizontal_color else default_horizontal_color
            vertical_color = mother_color if mother_color != default_horizontal_color else default_vertical_color
            
            ax.plot([birth_frame, death_frame], [y, y], color=horizontal_color)
            ax.text(death_frame, y, str(int(cell_index)), color=horizontal_color, verticalalignment='center',
                    horizontalalignment='left')
            ax.plot([birth_frame, birth_frame], [y_positions[mother], y], color=vertical_color)

            y_positions[cell_index] = y
            y = plot_lineage(cell_index, y)

        return y

    no_mothers = matrix[matrix[:, 3] == -1]
    for row in no_mothers:
        cell_index, birth_frame, death_frame, _ = row
        color = cell_colors.get(cell_index, default_horizontal_color)
        ax.plot([birth_frame, death_frame], [current_y, current_y], color=color)
        ax.text(death_frame, current_y, str(int(cell_index)), color=color, verticalalignment='center',
                horizontalalignment='left')
        y_positions[cell_index] = current_y
        current_y = plot_lineage(cell_index, current_y) + 1

    ax.set_ylim(-1, current_y + 1)
    ax.set_xlabel('Frame')
    ax.get_yaxis().set_visible(False)
    plt.grid(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    return fig



class LineageTreeWindow(QWidget):
    def __init__(self, parent, matrix, selected_cells=[]):
        super().__init__()
        self.parent = parent
        self.matrix = matrix
        self.selected_cells = selected_cells
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Create the matplotlib canvas and add it to the layout
        fig = plot_tree(self.matrix, self.selected_cells)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        # Optionally add more widgets like buttons to the layout
        # button = QPushButton('Close')
        # button.clicked.connect(self.close)
        # layout.addWidget(button)

        self.setLayout(layout)
        self.setWindowTitle('Cell Lineage Tree')
        self.show()

    def closeEvent(self, event):
        self.parent.showTreeButton.setCheckState(False)


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
        return self.parent.experiment().labels[self.labelIdx].celldata
    
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

        self.numPlots = len(data_dict)
        self.evalNames = [n for n,_ in list(self.data.values())[0]]
        self.numToEval = len(self.evalNames)
        self.xVals = IOUs

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

class SingleFramePlotWindow(QWidget):
    def __init__(self, parent, propDict, data, allFrames=False, plots_per_row=3):
        super().__init__()
        self.parent = parent
        self.propDict = propDict
        self.data = data
        self.allFrames = allFrames
        self.plots_per_row = plots_per_row
        self.docks = []
        self.plots = []  # Store the plot instances for updating
        self.initUI()

        if self.allFrames:
            self.update(self.parent.tIndex)

        #self.setGeometry(500, 100, 1500, 1500)
    
    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        self.area = DockArea()
        layout.addWidget(self.area)
        
        self.createPlots()
    
    def createPlots(self):
        row = 0
        col = 0
        
        for mask, properties in self.propDict.items():
            for prop in properties:
                plot = SingleFrameUpdatePlot(self, prop, mask, self.data[mask])
                plot_widget = plot.getPlotWidget()
                
                dock_name = f"{mask}_{prop}"
                dock = Dock(dock_name, closable=True)
                dock.addWidget(plot_widget)
                
                if col == 0 and row == 0:
                    self.area.addDock(dock, 'top')
                else:
                    position = 'right' if col > 0 else 'bottom'
                    relative_to = self.docks[-1] if col > 0 else self.docks[-self.plots_per_row]
                    self.area.addDock(dock, position, relative_to)
                
                self.docks.append(dock)
                self.plots.append(plot)
                col += 1
                if col >= self.plots_per_row:
                    col = 0
                    row += 1
    
    def update(self, frame_index):
        for plot in self.plots:
            plot.update(frame_index)
    
    def closeEvent(self, event):
        self.parent.sfPlotWindowOn = False


class TimeSeriesPlotWindow(QWidget):
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
            self.updatePopulationPlots()

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
        self.parent.tsPlotWindowOn = False