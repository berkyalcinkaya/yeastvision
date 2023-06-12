import shutil
import os 
from os.path import join
import glob
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import Qt


def loadCustomModel(parent):
    filter = "Cellpose Archive File (*);; HDF File (*.h5 *.hdf5)"
    currDir = parent.getCurrImDir() if parent.imLoaded else os.getcwd()
    path = QFileDialog.getOpenFileName(
        parent = parent,
        caption = "Select Custom Weights to Load",
        directory = currDir,
        filter = filter
    )
    customModelFile = os.path.split(path)[1]
    customModelName = customModelFile.split(".")[0]
    validName = False
    for modelType in parent.modelTypes:
        if modelType in customModelName:
            valid = True
            break
    if not validName:
        parent.showErrorMessage("Custom Model Must Have Model Type in File Name")
        return
    dst = join(parent.pathToModels, modelType, customModelFile)
    shutil.copy2(path,dst)
    parent.updateFileNames()

