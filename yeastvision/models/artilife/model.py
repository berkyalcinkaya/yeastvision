import cv2
from cv2 import merge, resize
import numpy as np
from yeastvision.models.cp import CustomCPWrapper
from skimage.measure import label
from yeastvision.models.utils import addMasks
from yeastvision.models.artilife.budSeg.model import BudSeg
from skimage.measure import label
from skimage.morphology import remove_small_objects
from yeastvision.track.track import trackYeasts
from yeastvision.track.mat import get_mating_data
from yeastvision.track.cell import Cell, getBirthFrame, getCellData, getDeathFrame, getLifeData
from yeastvision.models.utils import normalizeIm, produce_weight_path

class Artilife(CustomCPWrapper):
    hyperparams  = { 
    "Mean Diameter":30, 
    "Flow Threshold":0.4, 
    "Cell Probability Threshold": 0,
    "Mating Cells": False,
    "Sporulating Cells": False,
    "Time Series": False}
    types = [None, None,None, bool, bool, bool, bool]

    '''
    params also include:
        matStart - start index of mating cells
        matStop - stop index of mating cells (inclusive)
        matSeg - weights to use for matSeg 

        sporeStart - start index of sporulating cells
        sporeStop - stop index of sporulating cells (inclusive)
        tetradSeg - weights to use for tetradSeg
    these are hard coded via guiparts.ArtilifeParamDialog
    '''
    def __init__(self, params, weights):
        super().__init__(params, weights)

        self.matSeg, self.tetradSeg = None,None
        self.matMasks, self.matprobs = None, None
        self.tetraMasks, self.tetraprobs = None, None


        self.budSeg = BudSeg()
        if params["Mating Cells"]:
            from models.matSeg.model import MatSeg
            self.matSeg = MatSeg(self.params, produce_weight_path("matSeg", self.params["matSeg"]))

        if params["Sporulating Cells"]:
            from models.tetradSeg.model import TetradSeg
            self.tetradSeg = TetradSeg(self.params, produce_weight_path("tetradSeg", self.params["tetradSeg"]))

    def addTetrads(self, ims):
        tetraSlice = slice(int(self.params["sporeStart"]), int(self.params["sporeStop"])+1)
        self.tetraMasks, tetraFlows, _, _ = self.tetradSeg.model.eval(ims[tetraSlice], 
                                                                diameter = self.params["Mean Diameter"], 
                                                                channels = [0,0],
                                                                cellprob_threshold = self.params["Flow Threshold"],
                                                                do_3D = False)
        self.tetraprobs = [flow[2] for flow in tetraFlows]
        self.tetraprobs = np.array(self.processProbability(self.tetraprobs), dtype = np.uint8)

        if self.params["Time Series"]:
            newTetraMasks, newMasks = track_obj([im[:,:,0] for im in ims[tetraSlice]], self.masks[tetraSlice], self.tetraMasks[tetraSlice], False)
            tetraSlice = slice(int(self.params["sporeStart"]), int(self.params["sporeStop"])-1)
        else:
            for tetraMask, cellMask in zip(self.tetraMasks, self.masks):
                if np.any(tetraMask>=1):
                    newMask = addMasks(tetraMask, cellMask)
                    newMasks.append(newMask)
                else:
                    newMasks.append(cellMask)
            newTetraMasks = self.tetraMasks

        self.masks[tetraSlice] = newMasks
        tetraMasks = np.zeros_like(self.masks)
        tetraMasks[tetraSlice] = np.array(newTetraMasks, dtype = np.uint16)
        self.tetraMasks = tetraMasks


    def addMatingCells(self, ims):
        matSlice = slice(int(self.params["matStart"]),int(self.params["matStop"]) + 1)
        self.matMasks, matFlows, _, _ = self.matSeg.model.eval(ims[matSlice], 
                                                                diameter = self.params["Mean Diameter"], 
                                                                channels = [0,0],
                                                                cellprob_threshold = self.params["Flow Threshold"],
                                                                do_3D = False)
        print("got mating cells")
        self.matprobs = [flow[2] for flow in matFlows]
        self.matprobs = np.array(self.processProbability(self.matprobs), dtype = np.uint8)
        
        if self.params["Time Series"]:
            matSlice = slice(int(self.params["matStart"]), int(self.params["matStop"])-1)
            newMatMasks, newMasks = get_mating_data(self.matMasks[matSlice], self.masks[matSlice])
        else:
            newMasks = []
            for matMask, cellMask in zip(self.matMasks, self.masks):
                if np.any(matMask>=1):
                    newMask = addMasks(matMask, cellMask)
                    newMasks.append(newMask)
                else:
                    newMasks.append(cellMask)
            newMatMasks = self.matMasks

        self.masks[matSlice] = newMasks
        matMasks = np.zeros_like(self.masks)
        matMasks[matSlice] = np.array(newMatMasks, dtype = np.uint16)
        self.matMasks = matMasks

    def findSmallBud(self, ims, masks):
        smallBuds = []
        for imOrig, mask in zip(ims,masks):
            im = resize((normalizeIm(imOrig)*255).astype(np.uint8), (160,160))
            im = merge((im,im,im))
            pred = self.budSeg.model.predict(np.expand_dims(im,0))
            pred = resize(pred[0,:,:,1],(50,50))
            pred = (pred==1).astype(np.uint8)
            r,c = pred.shape
            pred = np.pad(pred[10:r-10,10:c-10], 10)
            pred[mask>0] = 0
            smallBud = self.keepLargestObject(pred)
            smallBuds.append(smallBud)
        return np.array(smallBuds, dtype = np.uint8)
    
    def keepLargestObject(self, mask):
        labeled = label(mask.copy())
        labeled = label(remove_small_objects(labeled, 5))
        areas = []
        for num in np.unique(labeled):
            if num>0:
                areas.append((labeled==num).sum())
        if areas:
            cellToKeep = areas.index(max(areas))+1
            return (labeled==cellToKeep).astype(np.uint8)
        else:
            return np.zeros_like(mask)

    def addSmallCells(self, ims, cellData):      
        newMasks = self.masks.copy()
        imsPadded = np.array([np.pad(im, 25, mode = "symmetric") for im in ims], dtype = np.uint16)
        masksPadded = np.array([np.pad(mask, 25) for mask in newMasks], dtype = np.uint16)
        
        for val in range(len(cellData["birth"])):
            cellVal = int(val)+1
            if cellVal>0:
                birthFrame = cellData["birth"][cellVal-1]
                
                if birthFrame > 0:
                    cell = Cell(masksPadded[birthFrame,:,:], cellVal)
                    r,c = cell.centroid
                    r,c = round(r), round(c)
                    index = 2 if birthFrame>1 else 1
                    budSlice = slice(birthFrame-index,birthFrame), slice(r-25,r+25), slice(c-25,c+25)
                    smallBuds = self.findSmallBud(imsPadded[budSlice], masksPadded[budSlice])
                    if np.any(smallBuds>0):
                        birth = getBirthFrame(smallBuds, 1)
                        masksPadded[budSlice][smallBuds>0] = cellVal
        r,c = masksPadded[0].shape
        self.masks = masksPadded[:,25:r-25,25:c-25].astype(np.uint16)

    @classmethod
    def run(cls, ims, params, weights):
        params = params if params else cls.hyperparams
        model = cls(params, weights)
        ims3D = [cv2.merge((im,im,im)) for im in ims]
        assert len(ims3D[0].shape)==3

        if not params["Mean Diameter"]:
            evaluator = model.cpAlone
            model.masks, flows, _ = evaluator.eval(ims3D, 
                                                    diameter = model.params["Mean Diameter"], 
                                                    channels = [0, 0],
                                                    cellprob_threshold = model.params["Flow Threshold"], 
                                                    do_3D=False)
        else:
            evaluator = model.model
    
            model.masks, flows, _, model.diams = evaluator.eval(ims3D, 
                                                    diameter = model.params["Mean Diameter"], 
                                                    channels = [0, 0],
                                                    cellprob_threshold = model.params["Flow Threshold"], 
                                                    do_3D=False)
        print("process probability")
        model.cellprobs = [flow[2] for flow in flows]
        model.cellprobs = np.array((model.processProbability(model.cellprobs)), dtype = np.uint8)
        
        print("formatting return")
        model.masks = np.array(model.masks, dtype = np.uint16)

        if model.params["Time Series"]:
            model.masks = trackYeasts(model.masks)
            cellData = getLifeData(model.masks)
            model.addSmallCells(ims, cellData)
            model.masks = trackYeasts(model.masks)

        if model.matSeg:
            print('add mating cell')
            model.addMatingCells(ims3D)
        if model.tetradSeg:
            model.addTetrads(ims3D)
        
        print("finished")
        
        return {"artilife": (model.masks, model.cellprobs), 
                "mating": (model.matMasks, model.matprobs),
                "tetrads": (model.tetraMasks, model.tetraprobs)
                }


        
    
