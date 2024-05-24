import numpy as np
from skimage.measure import regionprops_table
import pyqtgraph as pg
from PyQt5 import QtCore
'''
Plots types are divided into two large categories

Single Cell Plots - defined by cellDataDf (cell select change)
    Image Plots - mean, var, median vs time
    Morphological Plots - cell properties vs time
    Space plots - y vs x


Population Plots (label change)
    Average Image (remove zeros)
    Average Morphology (remove zeros)
    Count - vs time: option for curve fit
    Heatmaps (clustered)

Plot format
--Single Cell--                       ----Population Plots----
                                          Count vs Time 
Intensity Plots - tabbed                  Mean Intensity
Morphogical Plots - tabbed                Mean Morphologies
Space plot                                Heatmaps


Option to add a plot


Option to add a plot:                                                             New Plots
Property Plot
    Add single cell plot [] -->  Cell Number [dropdown] Property [dropdown]       -------------
    Add Population Plot [] -->  Plot Type                                         -------------

Correlation Plot
'''

def yvv(celldata):
    pass

class SingleFrameUpdatePlot:
    def __init__(self, parent, property_name, mask_name, data):
        self.parent = parent
        self.property_name = property_name
        self.mask_name = mask_name
        self.data = data
        self.plot = pg.PlotWidget(title=f"{self.property_name} for {self.mask_name}")
    
    def getData(self, frame_index=None):
        if frame_index is not None and isinstance(self.data, list):
            if frame_index < len(self.data):
                frame_data = self.data[frame_index]
                if self.property_name in frame_data:
                    return np.array(frame_data[self.property_name])
        elif self.property_name in self.data:
            return np.array(self.data[self.property_name])
        return np.array([])
    
    def update(self, frame_index=None):
        self.plot.clear()
        data = self.getData(frame_index)
        if data.size > 0:
            y, x = np.histogram(data, bins="auto")
            self.plot.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150))
    
    def getPlotWidget(self):
        self.update()
        return self.plot

class SingleCellUpdatePlot():
    def __init__(self, parent, property, timeseries, selectedCells):
        self.parent = parent
        self.property = property
        self.timeseries = timeseries
        self.tsNames = [self.parent.getTimeSeriesDataName(ts) for ts in self.timeseries]
        self.selectedCells = selectedCells
        self.plot = pg.PlotWidget(title=self.property)
    
    def getData(self):
        data = {}
        if self.selectedCells:
            for cell in self.selectedCells:
                for ts, name in zip(self.timeseries, self.tsNames):
                    if cell in ts.label_vals:
                        data[f"{name}-cell{cell}"] = ts.get(self.property, cell)
        return data
    
    def update(self):
        self.plot.clear()
        data = self.getData()
        if data:
            if len(data) > 1:
                self.plot.addLegend()
    
            i = 0
            for cellnum, dataset in data.items():
                color = self.parent.cmap[i][0:-1]
                self.plot.plot(dataset, name = str(cellnum), pen = tuple(color))
                i+=1
            
    def getPlotWidget(self):
        self.update()
        return self.plot


class PopulationPlot():
    def __init__(self, parent, property, timeseries, populationName):
        self.parent = parent
        self.property = property
        self.data = timeseries
        self.population = populationName
        self.plot = pg.PlotWidget(title=f"{self.property}-{self.population}")
    
    def getData(self):
        return self.data.get_population_data(self.property, self.population)

    def getVar(self):
        return self.data.get_population_data(self.property, self.population, func = np.std)
    
    def update(self):
        self.plot.clear()
        data = self.getData()
        variance = self.getVar()

        plot = self.plot

        x = np.arange(len(data))
        # Plot the data with a bold line
        plot.plot(data, pen='b', width=2, name=f"{self.property}-mean")

        # Plot the data points plus or minus the variance
        plot.plot(data + variance, pen='r', style=QtCore.Qt.DashLine, width=1)
        plot.plot(data - variance, pen='r', style=QtCore.Qt.DashLine, width=1)

        # Add error bars at each time point
        errorbar = pg.ErrorBarItem(x=x, y=data, height=variance*2, beam=0.2)
        plot.addItem(errorbar)

        # Shade the region between the plus or minus variance datapoints
        fill = pg.FillBetweenItem(curve1=plot.getPlotItem().listDataItems()[1],
                                    curve2=plot.getPlotItem().listDataItems()[-1],
                                    brush=(100, 100, 100, 100))
        plot.addItem(fill)

        # Set the x-axis and y-axis labels
        plot.setLabel('bottom', 'Time')
        plot.setLabel('left', 'Value')

        # Add a legend
        plot.addLegend()

    def getPlotWidget(self):
        self.update()
        return self.plot


class HeatMap():
    def __init__(self, parent, prop, populationName, data):
        self.parent = parent
        self.data = data
        self.populationName = populationName
        self.property = prop
        self.cmap = pg.colormap.get("jet", source = 'matplotlib')
        self.view  = pg.PlotWidget(title = f"{self.populationName}-{self.property}")
        self.im = pg.ImageItem(colorMap = self.cmap)
        self.bar = pg.ColorBarItem( values=(0,1), colorMap=self.cmap)
        self.bar.setImageItem(self.im, insert_in = self.view.getPlotItem())
        self.view.addItem(self.im)
    
    def getData(self):
        return self.data.get_heatmap_data(self.property, self.populationName)
    
    def update(self):
        imData = self.getData()
        self.im.setImage(imData, levels = [0,1])
    
    def getPlotWidget(self):
        self.update()
        return self.view
