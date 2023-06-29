from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data, main):
        super(TableModel, self).__init__()
        self.main = main
        self._data = data
        self.columns = {"cell":0, "birth":1, "death": 2, "mother":3, "confidence":4}
        self.hasLineages = "mother" in self._data.columns
        self.confidenceGradColors = plt.get_cmap("Greens", 12)
        if self.hasLineages:
            self.findPotentialErrors()

    def data(self, index, role):
        if index.isValid():
            if role == Qt.DisplayRole or role == Qt.EditRole:
                value = self._data.iloc[index.row(), index.column()]
                return str(value)
            
        if role == Qt.DisplayRole:
            value = (self._data.iloc[index.row(), index.column()]).copy()
            return str(value)
        
        if role == Qt.BackgroundRole:
            if self.hasLineages:
                if (index.row(), index.column()) in self.turnRed:
                    return QtGui.QColor('red')
                if index.column() == self.columns["confidence"]:
                    value = (self._data.iloc[index.row(), index.column()]).copy()
                    value = float(value)
                    value = round(value, 1)*10
                    r,g,b,a = self.confidenceGradColors(int(value)+1)
                    return QtGui.QColor(int(r*255),int(g*255),int(b*255),int(a*125))

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Vertical:
                return str(self._data.index[section])
    
    def flags(self, index):
        if index.column() ==  self.columns["mother"]:
            return Qt.ItemIsSelectable|Qt.ItemIsEnabled|Qt.ItemIsEditable
        else:
            return Qt.ItemIsSelectable|Qt.ItemIsEnabled
    
    def setData(self, index, value, role):
        if role == Qt.EditRole:
            self._data.iloc[index.row(),index.column()] = value
            self.main.saveData()
            return True
            
    def findPotentialErrors(self):
        groups = self._data.groupby(["mother", "birth"])
        result = groups.filter(lambda x: len(x) > 1)
        idxs = result.index.tolist()
        self.turnRed = list(zip(idxs, [self.columns["mother"]]*len(idxs)))
    
    def handleselect(self, new, previous):
        if previous.column() == -1:
            return
        


        if (new.column() == self.columns["cell"] or new.column() == self.columns["mother"]):
            value = self._data.iloc[new.row(), new.column()]
            if value == np.NaN:
                return
            birth = int(self._data.iloc[value-1, self.columns["birth"]])
            self.main.tIndex = birth
            self.main.selectCellFromNum(value)

        
        if (new.column() in [self.columns["birth"], self.columns["death"]]):
            self.main.tIndex = int(self._data.iloc[new.row(), new.column()])
            self.main.drawIm()
            self.main.drawMask()

        

        
