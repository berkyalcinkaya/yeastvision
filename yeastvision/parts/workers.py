from yeastvision.models.utils import is_RGB
import torch
import numpy as np
from PyQt5 import QtCore

class SegmentWorker(QtCore.QObject):
    '''Handles Multithreading'''
    finished = QtCore.pyqtSignal(object, object, object, object, object, object, object, object)
    def __init__(self, modelClass,ims, params, exp_idx, weightPath, modelType, mask_template_i = None):
        super(SegmentWorker,self).__init__()
        self.mc = modelClass
        self.ims = ims
        self.params = params
        self.weight = weightPath
        self.mType = modelType
        self.exp_idx = exp_idx
        self.mask_i = mask_template_i

    def run(self):
        im_sample = self.ims[0]
        if is_RGB(im_sample):
            row, col, d = im_sample.shape
        else:
            row, col = im_sample.shape
        
        newImTemplate = np.zeros((len(self.ims), row, col))
        tStart, tStop = int(self.params["T Start"]), int(self.params["T Stop"])
        with torch.no_grad():
            output = self.mc.run(self.ims[tStart:tStop+1],self.params, self.weight)
        self.finished.emit(output, self.mc, newImTemplate, self.params, self.weight, self.mType, self.exp_idx, self.mask_i)

class TrackWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, object, object)

    def __init__(self, func, cells, z, exp_idx, obj = None):
        super(TrackWorker,self).__init__()
        self.trackfunc = func
        self.z = z
        self.cells = cells
        self.exp_idx = exp_idx
        self.obj = obj
    
    def run(self):
        if self.obj is not None:
            out =  self.trackfunc(self.obj, self.cells)
        else:
            out = self.trackfunc(self.cells, transpose_out=True)
        self.finished.emit(self.z, self.exp_idx, out)

class InterpolationWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, object, object, object, object, object, object)

    def __init__(self, ims, newname, annotations, experiment_index, func, intervals, original_len):
        super(InterpolationWorker, self).__init__()
        self.ims = ims
        self.name = newname
        self.annotations = annotations
        self.exp_idx = experiment_index
        self.intervals = intervals
        self.func = func
        self.original_len = original_len

    def run(self):
        ims, new_intervals = self.func(self.ims, self.intervals)
        self.finished.emit(ims, self.exp_idx, self.name, self.annotations, 
                           self.intervals, self.original_len, new_intervals)

class FiestWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, object, object, object)
    def __init__(self, func, exp_idx, channel_id, lineage):
        super(FiestWorker,self).__init__()
        self.func = func
        self.exp = exp_idx
        self.channel_id = channel_id
        self.lineage = lineage
    
    def run(self):
        self.finished.emit(self.func(), self.exp, self.channel_id, self.lineage)