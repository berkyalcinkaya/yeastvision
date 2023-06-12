import numpy as np
from skimage.measure import regionprops_table

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