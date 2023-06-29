import pandas as pd
import numpy as np
from skimage.morphology import binary_dilation, binary_erosion, dilation, erosion, disk
from skimage.measure import label, regionprops
from yeastvision.models.unet import unet
from yeastvision.track.cell import getBirthFrame, getLifeData
import matplotlib.pyplot as plt 
from yeastvision.track.utils import normalize_dict_by_sum
from tqdm import tqdm
from skimage.morphology import skeletonize

def count_objects(labeledMask):
    return np.count_nonzero(np.unique(labeledMask))

def get_daughters(tree, cell):
    potentialDaughters = tree[cell-1]
    daughters = list(np.where(potentialDaughters)[0]+1)
    return daughters

def has_daughters(tree, cell):
    return np.any(tree[cell-1,:])   

    
def get_generations(tree, initial_generation):
    generations = [[]]
    for val in initial_generation:
        generations[0].append(val)
        get_generations_recur(tree, 1, generations, 0, val)
    print(generations)
    return generations

def get_generations_recur(tree, generation, generations, max_generation, cell):
    if not has_daughters(tree, cell):
        return 
    
    daughters = get_daughters(tree, cell)
    if generation>=len(generations):
        generations.append([])
        max_generation = generation
    for daughter in daughters:
        generations[generation].append(daughter)
        get_generations_recur(tree, generation+1, generations, max_generation, daughter)

def get_gametes(skeleton, cyto, idx):
    found_gametes = False
    overlap = []
    previous_overlaps = []
    while (not found_gametes) and idx>=0:
        mask = cyto[idx]
        overlap = np.unique(mask[skeleton>0])
        overlap = overlap[overlap!=0]

        previous_overlaps.append(overlap)
        found_gametes = len(overlap)==2
        idx -=1
    
    if len(overlap)==0:
        for previous_overlap in previous_overlaps:
            if len(previous_overlap)>0:
                overlap = previous_overlap
                break

    if len(overlap)>2:
        overlap = overlap[0:2]
    else:
        overlap = np.append(overlap, [None, None])
    return found_gametes, overlap

def is_mating_cell(mat, cyto, cell):
    return np.all(mat[cyto==cell]==cell)

def compute_mating_lineage(tracked, cyto):

    cellVals = np.unique(cyto)
    cellVals = cellVals[cellVals!=0]

    matingCells = np.unique(tracked)
    matingCells = matingCells[matingCells!=0]

    numCells = np.count_nonzero(matingCells)
    
    print("computing lineages for", numCells, "cells")
    gamete_dict = {
                  "gamete1":[],
                    "gamete2":[],
                    "isMating":[],
                    "foundGametes":[]}

    for label in tqdm(cellVals):
        isMating = label in matingCells
        if isMating:
            birth = getBirthFrame(tracked, label)
            mat_birth_mask = (tracked[birth] == label).astype(np.uint8)
            found_gametes, gametes = get_gametes(skeletonize(mat_birth_mask),
                                                 cyto, birth-1)
        else:
            found_gametes = False
            gametes = [None, None]
        
        gamete_dict["gamete1"].append(gametes[0])
        gamete_dict["gamete2"].append(gametes[1])
        gamete_dict["isMating"].append(isMating)
        gamete_dict["foundGametes"].append(found_gametes)
    
    return pd.DataFrame.from_dict(gamete_dict)

class LineageConstruction():
    def __init__(self, segMasks, budMasks = None, backskip = 0, forwardskip = 0):
        self.cellMasks = segMasks 
        self.neckMasks = budMasks
        self.backskip = backskip
        self.forskip = forwardskip
        self.maxT = len(self.cellMasks)-1
        self.budNET = None
        
        self.labelMasks()

        # print("--shaving masks--")
        # self.shaveMasks()

        # self.labelMasks()
    

    
    def shaveMasks(self):
        i = 0
        for cellmask, neckmask in zip(self.cellMasks, self.neckMasks):
            print("\tshave mask", i)
            newNeckMask = np.zeros_like(neckmask)
            if np.any(neckmask):
                for neckI in np.unique(neckmask):
                    if neckI>0:
                        print("\t\t", neckI)
                        newNeckMask += self.processNeckMask(cellmask, neckmask==neckI)
            self.neckMasks[i] = newNeckMask      
            i+=1


    def processNeckMask(self, segIm, neckMask):
        processCount = 0
        overlapCount = count_objects(segIm[neckMask])
        r = 1
        newMask = neckMask.copy()
        footprint =  disk(r)
        while overlapCount != 2 and processCount<2:
            if overlapCount>2:
                newMask = binary_erosion(newMask, footprint = footprint)
            elif overlapCount<2:
                newMask = binary_dilation(newMask, footprint = footprint)
            overlapCount = count_objects(segIm[newMask])
            processCount+=1
        
        if overlapCount !=2:
            return neckMask.astype(np.uint8)
        else:
            return newMask
        
            
    def labelMasks(self):
        self.neckMasks = [label((mask>0).astype(np.uint16)) for mask in self.neckMasks]

    def getInitialCells(self):
        if count_objects(self.cellMasks[0])>1:
                self.initialCells = np.unique(self.neckMasks[0])
                self.initialCells = self.initialCells[self.initialCells!=0]

                if count_objects(self.neckMasks[0])>0:
                    pass
        else:
            self.initialCells = None

    def computeLineages(self):
        numCells = np.count_nonzero(np.unique(self.cellMasks))
        cellVals = np.unique(self.cellMasks)
        cellVals = cellVals[cellVals!=0]

        print("computing lineages for", numCells, "cells")
        daughterArray = np.zeros((numCells, numCells), dtype = bool) # rows = cell, column = daughter
        motherDict = getLifeData(self.cellMasks)
        motherDict["cell"] = cellVals
        motherDict["mother"] = []
        motherDict["confidence"] = []

        for cellId in tqdm(cellVals):
            mother, confidence = self.getMother(cellId)
            if mother:
                daughterArray[mother-1, cellId-1] = True
            motherDict["mother"].append(mother)
            motherDict["confidence"].append(confidence)
        
        motherDF = pd.DataFrame.from_dict(motherDict)
        #daughterDF = pd.DataFrame(data = daughterArray, index = cellVals, columns=cellVals)
        #motherDF = motherDF.set_index("cell")
        return daughterArray, motherDF[["birth", "death", "mother", "confidence"]]
    
    def getMother(self, cellNum):
        '''
        1) get bud that overlaps with daughter the most
        2) '''
        birth = getBirthFrame(self.cellMasks, cellNum)
        birthFrame = self.cellMasks[birth]
        eligibleFrames = slice(max(0, birth-self.backskip), min(self.maxT+1, birth+self.forskip))


        overlapcounts = {}
        i = 0
        for cellframe, budframe in zip(self.cellMasks[eligibleFrames], self.neckMasks[eligibleFrames]):
            if i < birth:
                cellframe = self.cellMasks[birth]

            potentialBuds, counts= np.unique(budframe[cellframe==cellNum], return_counts=True)
            if np.any(potentialBuds>0):
                potentialBuds, counts = potentialBuds[potentialBuds!=0], counts[potentialBuds!=0]
                correctBud = potentialBuds[np.argmax(counts)]
                mothers, counts = np.unique(cellframe[budframe==correctBud], return_counts=True)
                for mother, count in zip(mothers, counts):
                    if mother !=cellNum and mother!=0:
                        if mother in overlapcounts:
                            overlapcounts[mother] += count
                        else:
                            overlapcounts[mother] = count  
            i+=1   
        if overlapcounts:
            overlapcounts = normalize_dict_by_sum(overlapcounts)
            mother = max(overlapcounts, key = overlapcounts.get)
            confidence = overlapcounts[mother]

            if birth==0:
                motherArea = np.sum(birthFrame == mother)
                daughterArea = np.sum(birthFrame==cellNum)
                if daughterArea > motherArea:
                    return None, 0
            return mother, round(confidence,2)
        else:
            return None, 0














    



    
    






