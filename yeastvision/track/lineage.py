from typing import Tuple
import pandas as pd
import logging
import numpy as np
from skimage.morphology import binary_dilation, binary_erosion, disk
from skimage.measure import label
from yeastvision.track.cell import getBirthFrame, getLifeData
from yeastvision.track.utils import normalize_dict_by_sum
from tqdm import tqdm
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)

class LineageConstruction:
    """
    A class for computing asexual lineages of Saccharomyces cerevisiae using tracked cell segmentation masks and bud neck masks.

    This class provides methods to analyze time-series images of yeast cells, identify mother-daughter relationships,
    and construct lineages based on cell segmentation and bud neck detection. The main method is computeLineages() which
    produces mother-daughter infomration. The main algorithm for lineage assignments is presented in getMothers()

    Attributes:
        cellMasks (np.ndarray): A 3D numpy array containing the segmentation masks for cells across different time frames.
        neckMasks (np.ndarray): A 3D numpy array containing the bud neck masks across different time frames.
        backskip (int): The number of frames to look backward when determining mother cells.
        forskip (int): The number of frames to look forward when determining mother cells.
        maxT (int): The maximum time frame index, calculated from the length of cellMasks.
        budNET (Any): A placeholder for future implementation (currently not used).
    """

    def __init__(self, segMasks, budMasks=None, backskip=0, forwardskip=0):
        """
        Initializes the LineageConstruction object with segmentation masks and optional bud neck masks.

        Args:
            segMasks (np.ndarray): A 3D numpy array containing the segmentation masks for cells across different time frames.
            budMasks (np.ndarray, optional): A 3D numpy array containing the bud neck masks across different time frames. Defaults to None.
            backskip (int, optional): The number of frames to look backward when determining mother cells. Defaults to 0.
            forwardskip (int, optional): The number of frames to look forward when determining mother cells. Defaults to 0.
        """
        self.cellMasks = segMasks
        self.neckMasks = budMasks
        self.backskip = backskip
        self.forskip = forwardskip
        self.maxT = len(self.cellMasks) - 1
        self.budNET = None

    def shaveMasks(self):
        """
        Processes and refines the bud neck masks to ensure proper separation of bud neck regions.

        This method iterates over each time frame and refines the bud neck masks by applying morphological operations.
        """
        i = 0
        for cellmask, neckmask in zip(self.cellMasks, self.neckMasks):
            logger.info(f"\tshave mask  {i}")
            newNeckMask = np.zeros_like(neckmask)
            if np.any(neckmask):
                for neckI in np.unique(neckmask):
                    if neckI > 0:
                        logger.info(f"\t\t {neckI}")
                        newNeckMask += self.processNeckMask(cellmask, neckmask == neckI)
            self.neckMasks[i] = newNeckMask
            i += 1

    def processNeckMask(self, segIm, neckMask):
        """
        Refines a single bud neck mask to ensure it overlaps with exactly two cells.

        Args:
            segIm (np.ndarray): A 2D array representing the segmentation image for a single time frame.
            neckMask (np.ndarray): A 2D binary array representing the bud neck mask to be processed.

        Returns:
            np.ndarray: The refined bud neck mask, adjusted to overlap with exactly two cells if possible.
        """
        processCount = 0
        overlapCount = count_objects(segIm[neckMask])
        r = 1
        newMask = neckMask.copy()
        footprint = disk(r)
        while overlapCount != 2 and processCount < 2:
            if overlapCount > 2:
                newMask = binary_erosion(newMask, footprint=footprint)
            elif overlapCount < 2:
                newMask = binary_dilation(newMask, footprint=footprint)
            overlapCount = count_objects(segIm[newMask])
            processCount += 1

        if overlapCount != 2:
            return neckMask.astype(np.uint8)
        else:
            return newMask

    def labelMasks(self):
        """
        Labels each connected component in the bud neck masks.

        This method applies connected component labeling to the bud neck masks, where each connected region of non-zero pixels
        is assigned a unique label.
        """
        self.neckMasks = [label((mask > 0).astype(np.uint16)) for mask in self.neckMasks]

    def getInitialCells(self):
        """
        Identifies the initial cells present in the first time frame based on cell and bud neck masks.

        This method counts the cells and bud necks in the first frame and identifies the initial cells involved in budding events.
        """
        if count_objects(self.cellMasks[0]) > 1:
            self.initialCells = np.unique(self.neckMasks[0])
            self.initialCells = self.initialCells[self.initialCells != 0]

            if count_objects(self.neckMasks[0]) > 0:
                pass
        else:
            self.initialCells = None

    def computeLineages(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Determines mother-daughter pairs and constructs cell lineages based on segmentation and bud neck masks.

        This method iterates through all cells in `self.cellMasks` and identifies mother-daughter relationships by calling `self.getMother`.
        It returns an array indicating the daughter relationships and a DataFrame containing cell life data. See getMother for more detailed
        information about confidence scores

        Returns:
            tuple:
                np.ndarray: A boolean array where rows represent mother cells and columns with True indicate daughter cells.
                pd.DataFrame: A DataFrame containing cell life data with columns 'birth', 'death', 'mother', and 'confidence'.
        """
        numCells = np.count_nonzero(np.unique(self.cellMasks))
        cellVals = np.unique(self.cellMasks)
        cellVals = cellVals[cellVals != 0]

        logger.info(f"computing lineages for {numCells} cells")
        daughterArray = np.zeros((numCells, numCells), dtype=bool)  # rows = cell, column = daughter
        motherDict = getLifeData(self.cellMasks)
        motherDict["cell"] = cellVals
        motherDict["mother"] = []
        motherDict["confidence"] = []

        for cellId in tqdm(cellVals):
            mother, confidence = self.getMother(cellId)
            if mother:
                daughterArray[mother - 1, cellId - 1] = True
            motherDict["mother"].append(mother)
            motherDict["confidence"].append(confidence)

        motherDF = pd.DataFrame.from_dict(motherDict)
        return daughterArray, motherDF[["birth", "death", "mother", "confidence"]]

    def getMother(self, cellNum):
        """
        Determines the most likely mother cell of a given daughter cell in budding yeast colonies,
        based on the overlap of bud necks and cell regions across a series of frames starting from the birth of the daughter.

        This function processes images over a range of frames from the daughter cell's birth frame up to three additional frames.
        It considers all potential bud necks overlapping with the daughter cell and calculates the overlaps with surrounding cells.
        The function returns the mother cell with the highest normalized overlap count, suggesting the strongest connection through the bud neck.

        Args:
            cellNum (int): The unique identifier for the daughter cell whose mother cell needs to be identified.

        Returns:
            tuple: A tuple containing the mother cell's identifier and a confidence level (rounded to two decimal places).
                   Returns (None, 0) if no mother is identified due to no overlaps or other criteria not being met.

        Pseudocode:
        1. Get the birth frame of the daughter cell.
        2. Calculate the area of the daughter cell at the birth frame.
        3. Define the range of frames to analyze (from birth to three frames post-birth).
        4. Initialize a dictionary to store overlap counts for potential mother cells.
        5. Loop through each frame in the defined range:
           a. For each frame, identify and filter bud necks that overlap with the daughter cell.
           b. For each valid bud neck, determine overlapping cells and calculate their areas.
           c. Add to the overlap count for cells (potential mothers) if they are not the daughter and are larger than the daughter.
        6. Normalize the overlap counts to determine the confidence level for each potential mother.
        7. Return the mother cell with the highest confidence level, or (None, 0) if no suitable mother is found.
        """
        birth = getBirthFrame(self.cellMasks, cellNum)
        birthFrame = self.cellMasks[birth]
        daughterArea = np.sum(birthFrame == cellNum)
        eligibleFrames = range(birth, min(birth + self.forskip, self.maxT + 1))

        overlapcounts = {}

        for frameIndex in eligibleFrames:
            cellframe = self.cellMasks[frameIndex]
            budframe = self.neckMasks[frameIndex]
            potentialBuds, counts = np.unique(budframe[cellframe == cellNum], return_counts=True)
            valid_buds = potentialBuds > 0
            potentialBuds = potentialBuds[valid_buds]
            counts = counts[valid_buds]

            for bud, count in zip(potentialBuds, counts):
                mothers, mcounts = np.unique(cellframe[budframe == bud], return_counts=True)
                valid_mothers = (mothers != cellNum) & (mothers != 0)

                for mother, mcount in zip(mothers[valid_mothers], mcounts[valid_mothers]):
                    motherArea = np.sum(cellframe == mother)
                    if motherArea > daughterArea:
                        if mother in overlapcounts:
                            overlapcounts[mother] += mcount
                        else:
                            overlapcounts[mother] = mcount

        if overlapcounts:
            overlapcounts = normalize_dict_by_sum(overlapcounts)
            mother = max(overlapcounts, key=overlapcounts.get)
            confidence = overlapcounts[mother]
            return mother, round(confidence, 2)
        else:
            return None, 0


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
    logger.info(generations)
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
    
    logger.info(f"computing lineages for {numCells} cells")
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

    # def getMother(self, cellNum):
    #     '''
    #     1) get bud that overlaps with daughter the most
    #     2) '''
    #     birth = getBirthFrame(self.cellMasks, cellNum)
    #     birthFrame = self.cellMasks[birth]
    #     eligibleFrames = slice(max(0, birth-self.backskip), min(self.maxT+1, birth+self.forskip))


    #     overlapcounts = {}
    #     i = 0
    #     for cellframe, budframe in zip(self.cellMasks[eligibleFrames], self.neckMasks[eligibleFrames]):
    #         if i < birth:
    #             cellframe = self.cellMasks[birth]

    #         potentialBuds, counts= np.unique(budframe[cellframe==cellNum], return_counts=True)
    #         if np.any(potentialBuds>0):
    #             potentialBuds, counts = potentialBuds[potentialBuds!=0], counts[potentialBuds!=0]
    #             correctBud = potentialBuds[np.argmax(counts)]
    #             mothers, counts = np.unique(cellframe[budframe==correctBud], return_counts=True)
    #             for mother, count in zip(mothers, counts):
    #                 if mother !=cellNum and mother!=0:
    #                     if mother in overlapcounts:
    #                         overlapcounts[mother] += count
    #                     else:
    #                         overlapcounts[mother] = count  
    #         i+=1   
    #     if overlapcounts:
    #         overlapcounts = normalize_dict_by_sum(overlapcounts)
    #         mother = max(overlapcounts, key = overlapcounts.get)
    #         confidence = overlapcounts[mother]

    #         if birth==0:
    #             motherArea = np.sum(birthFrame == mother)
    #             daughterArea = np.sum(birthFrame==cellNum)
    #             if daughterArea > motherArea:
    #                 return None, 0
    #         return mother, round(confidence,2)
    #     else:
    #         return None, 0











    



    
    






