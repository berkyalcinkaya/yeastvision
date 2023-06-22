import shutil
import os 
from os.path import join
import glob
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import Qt
from yeastvision.models.utils import MODEL_DIR


def loadCustomModel(parent):
    filter = "Archive File (*);; HDF File (*.h5 *.hdf5)"
    currDir = parent.getCurrImDir() if parent.imLoaded else os.getcwd()
    path,_ = QFileDialog.getOpenFileName(
        parent = parent,
        caption = f'''Select Custom Weights to Load. The name of the model architecture should be included in the filename.Current models are {parent.modelTypes}''',
        directory = currDir,
        filter = filter
    )

    if not path:
        return

    customModelFile = os.path.split(path)[1]
    customModelName = customModelFile.split(".")[0]
    validName = False
    for modelType in parent.modelTypes:
        if modelType.lower() in customModelName.lower():
            validName = True
            break
    if not validName:
        parent.showError("Custom Model Must Have Model Type in File Name")
        return
    dst = join(MODEL_DIR, modelType, customModelFile)
    shutil.copy2(path,dst)
    parent.updateModelNames()

