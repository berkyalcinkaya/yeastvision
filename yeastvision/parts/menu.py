from PyQt5.QtWidgets import QAction
from yeastvision.disk import io

def menubar(parent):
    mainMenu = parent.menuBar()
    fileMenu = mainMenu.addMenu("File")

    loadIm = QAction("Load Experiment", parent)
    loadIm.triggered.connect(parent.userSelectExperiment)
    fileMenu.addAction(loadIm)

    editMenu = mainMenu.addMenu("Edit")
    addBlankMask = QAction("Add Blank Label", parent)
    addBlankMask.triggered.connect(parent.addBlankMasks)
    editMenu.addAction(addBlankMask)
    removeAll = QAction("Clear All", parent)
    removeAll.triggered.connect(parent.setEmptyDisplay)
    clearCurrent = QAction("Clear current mask", parent)
    clearCurrent.triggered.connect(parent.clearCurrMask)
    editMenu.addAction(clearCurrent)
    editMenu.addAction(removeAll)
    labelObjects = QAction("Label Unique Regions", parent)
    labelObjects.triggered.connect(parent.labelCurrMask)
    editMenu.addAction(labelObjects)

    preMenu = mainMenu.addMenu("Preprocess")
    medianFilter = QAction("Median Filter", parent)
    medianFilter.triggered.connect(parent.doMedian)
    preMenu.addAction(medianFilter)
    gaussFilter = QAction("Gaussian Filter", parent)
    gaussFilter.triggered.connect(parent.doGaussian)
    preMenu.addAction(gaussFilter)
    adaptHist = QAction("Adaptive Histogram Equalization", parent)
    adaptHist.triggered.connect(parent.doAdaptHist)
    preMenu.addAction(adaptHist)
    rescale = QAction("Rescale", parent)
    # cannot add a threadd for rescale because of error with parent from other thread
    rescale.triggered.connect(lambda: parent.doRescale())
    preMenu.addAction(rescale)
    zNorm =  QAction("Z-normalize channel", parent)
    zNorm.triggered.connect(parent.doZNorm)
    preMenu.addAction(zNorm)

    exportMenu = mainMenu.addMenu("Export")
    imSave = QAction("Save Images", parent)
    imSave.triggered.connect(parent.saveIms)
    exportMenu.addAction(imSave)
    maskSave = QAction("Save Masks", parent)
    maskSave.triggered.connect(parent.saveMasks)
    exportMenu.addAction(maskSave)
    overlaySave = QAction("Save Figure", parent)
    overlaySave.triggered.connect(parent.saveFigure)
    exportMenu.addAction(overlaySave)
    dataSave = QAction("Export Cell Data", parent)
    dataSave.triggered.connect(parent.saveCellData)
    exportMenu.addAction(dataSave)
    hmSave = QAction("Export Heatmaps", parent)
    hmSave.triggered.connect(parent.saveHeatMaps)
    exportMenu.addAction(hmSave)
    lineageSave = QAction("Export Lineage Data", parent)
    lineageSave.triggered.connect(parent.saveLineageData)
    exportMenu.addAction(lineageSave)
    figureCreate = QAction("Create Label Overlay", parent)
    figureCreate.triggered.connect(parent.createMaskOverlay)
    exportMenu.addAction(figureCreate)

    modelMenu = mainMenu.addMenu("Models")
    trainModel = QAction("Train Model", parent)
    trainModel.triggered.connect(parent.showTW)
    modelMenu.addAction(trainModel)

    loadCustom = QAction("Load Custom Model", parent)
    loadCustom.triggered.connect(lambda: io.loadCustomModel(parent))
    modelMenu.addAction(loadCustom)

    evaluate = QAction("Evaluate Predictions", parent)
    evaluate.triggered.connect(parent.evaluate)
    modelMenu.addAction(evaluate)

    plotMenu = mainMenu.addMenu("Plot")

    getCellData = QAction("Update Cell Data", parent)
    getCellData.triggered.connect(parent.updateCellData)
    plotMenu.addAction(getCellData)


    