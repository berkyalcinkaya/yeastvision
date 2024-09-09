import shutil
import os 
import itertools
from os.path import join
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QDialog
from yeastvision.models.utils import MODEL_DIR, CUSTOM_MODEL_TYPES, getBuiltInModelTypes
from yeastvision.parts.dialogs import ComboBoxDialog

def custom_model_type_search(key):
    for model in CUSTOM_MODEL_TYPES:
        for k in CUSTOM_MODEL_TYPES[model]:
            if k==key:
                return model
    return "proSeg"

def loadCustomModel(parent):
    filter = "Archive File (*);; Pytorch File (*.pth)"
    currDir = parent.getCurrImDir() if parent.imLoaded else os.getcwd()
    path,_ = QFileDialog.getOpenFileName(
        parent = parent,
        caption = f'''Select Custom Weights to Load. The name of a pre-existing model architecture should be included in the filename if the model
        is a derivative of one of{",".join(parent.modelTypes)}''',
        directory = currDir,
        filter = filter
    )

    if not path:
        return

    customModelFile = os.path.split(path)[1]
    customModelName = customModelFile.split(".")[0]
    validName = False
    
    modelTypes = getBuiltInModelTypes()
    for modelType in modelTypes:
        
        if modelType.lower() == customModelName.lower():
            parent.showError(f"custom model name must not be one of {','.join(modelTypes)}")            
        
        if modelType.lower() in customModelName.lower():
            validName = True
            break
    if not validName:
        choices = list(itertools.chain.from_iterable(CUSTOM_MODEL_TYPES.values()))
        dialog = ComboBoxDialog(choices,
                                "Model Type: ", 
                                "Specify Custom Model Type. Pick the most applicable",
                                parent=parent)
        if dialog.exec_() == QDialog.Accepted:
            selectedModelType = dialog.getSelection()
            if selectedModelType:
                modelType = custom_model_type_search(selectedModelType)
        else:
            parent.showError("Please specify a model type")
            return
    dst = join(MODEL_DIR, modelType, customModelFile)
    shutil.copy2(path,dst)
    parent.newModels()

