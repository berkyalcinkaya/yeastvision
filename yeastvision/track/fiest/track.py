from typing import Optional, List
import time
import numpy as np
from scipy.stats import mode
import scipy.io as sio
from skimage.morphology import thin, disk, binary_opening, dilation, opening
from yeastvision.track.fiest.utils import remove_artif, binar, OAM_23121_tp3
import logging
from yeastvision.models.utils import produce_weight_path
from yeastvision.ims.interpolate import interpolate_intervals, get_interp_labels, deinterpolate
from yeastvision.models.proSeg.model import ProSeg
from yeastvision.models.budSeg.model import BudSeg
from yeastvision.track.track import track_proliferating
from yeastvision.track.lineage import LineageConstruction

logger = logging.getLogger(__name__)

def fiest_basic(ims:np.ndarray, interp_intervals:Optional[List[dict]], proSeg_params:Optional[dict], proSeg_weights:Optional[str]=None)->Tuple[np.ndarray[np.uint16], np.ndarray[np.uint8], np.ndarray[np.uint8]]:
    '''Implementation 1 of Frame Interpolation Enhanced Single-cell Tracking (FIEST) from the paper 'Deep learning-driven 
    imaging of cell division and cell growth across an entire eukaryotic life cycle' for tracking of timeseries movies of
    yeasts that proliferate asexually. This novel algorithm is based on enhancing the overlap of single 
    cell masks in consecutive images through deep learning video frame interpolation. 
    
    Steps: (1) interpolate with RIFE, (2) segment with proSeg, (3) track, (4) deinterpolate
    
    Args:
        ims (np.ndarray): an (n x h x w) array with n images of h x w dimensions to be interpolated and segmented
        interp_intervals (optional, lists of dict): optional list specifying interpolation levels and intervals, each entry
                                                    of the list should be a dictionary with the keys (start, stop, interp)
                                                    where interp is an integer value in [1,2,3,4]. If none, no interpolation is
                                                    performed
        proSeg_params (optional dict): a dictionary specifying the parameters for segmentation with proSeg, should have the keys 
                                    'mean_diameter,' 'flow_threshold,' and 'cell_probability_threshold.' If none, defaults are used
        proSeg_weights (optional, path): path to weights to use. If none, loads default proSeg weights from the miranda lab. Assumes weights
                                        have been downloaded
                                    
    Returns:
        tuple:
            np.ndarray[uint16] - the tracked masks
            np.ndarray[uint8] - the cell probability masks
            np.ndarray[uint8] - the cell flows masks
    '''
    
    logger.info("---FIEST Tracking Basic---")
    
    if interp_intervals:
        logger.info(f"Performing interpolation over intervals:\n {interp_intervals}")
        interpolated, new_intervals = interpolate_intervals(ims, interp_intervals)
        interp_locs = get_interp_labels(len(ims), interp_intervals)

    proSeg, proSeg_params, proSeg_weights = _get_proSeg(proSeg_params, proSeg_weights)
    
    masks, probs, flows = proSeg.run(interpolated, proSeg_params, proSeg_weights)
    masks = track_proliferating(masks)
    
    if interp_intervals:
        masks, probs, flows = [deinterpolate(mask_type, interp_locs) for mask_type in [masks, probs, flows]]
    
    return masks, probs, flows
    

def fiest_basic_with_lineage(ims:np.ndarray, interp_intervals:Optional[List[dict]], proSeg_params:Optional[dict], proSeg_weights:Optional[str]=None,
                            budSeg_params:Optional[dict]=None, budSeg_weights:Optional[str]=None)->Tuple[np.ndarray[np.uint16], np.ndarray[np.uint8], np.ndarray[np.uint8]]:
    '''
    Implementation 2 of Frame Interpolation Enhanced Single-cell Tracking (FIEST) and lineage reconstruction alogirthm 
    from the paper 'Deep learning-driven imaging of cell division and cell growth across an entire eukaryotic life cycle' 
    for tracking cells and lineages of yeasts that proliferate asexually. This novel algorithm is based on enhancing the overlap of single 
    cell masks and mother-daughter pairs in consecutive images through deep learning video frame interpolation.
    
    Steps: (1) interpolate with RIFE, (2) segment with proSeg, (3) track, (4) construct lineages, (4) deinterpolate
    
    Args:
        ims (np.ndarray): an (n x h x w) array with n images of h x w dimensions to be interpolated and segmented
        interp_intervals (list, optional): optional list specifying interpolation levels and intervals, each entry
                                                    of the list should be a dictionary with the keys (start, stop, interp)
                                                    where interp is an integer value in [1,2,3,4]. If none, no interpolation is
                                                    performed
        proSeg_params (dict, optional): a dictionary specifying the parameters for segmentation with proSeg, should have the keys 
                                    'mean_diameter,' 'flow_threshold,' and 'cell_probability_threshold.' If none, defaults are used
        proSeg_weights (str, optional): path to weights to use. If none, loads default proSeg weights from the miranda lab. Assumes weights
                                        have been downloaded
        budSeg_params (dict, optional): budSeg parameters for budding cell segmentation. See proSeg_params
        budSeg_weights (str, optional): path to weights for budSeg predictions, see proSeg_weights. 
                                    
    Returns:
        tuple:
            np.ndarray[uint16] - the tracked masks
            np.ndarray[uint8] - the cell probability masks
            np.ndarray[uint8] - the cell flows masks
            np.ndarray[uint16] - cell daughters array
            pd.DataFrame - a DataFrame containing cell life data with columns 'birth', 'death', 'mother', and 'confidence'.
    '''
    logger.info("---FIEST Tracking With Lineage Reconstruction---")
    
    if interp_intervals:
        logger.info(f"Performing interpolation over intervals:\n {interp_intervals}")
        interpolated, new_intervals = interpolate_intervals(ims, interp_intervals)
        interp_locs = get_interp_labels(len(ims), interp_intervals)


    proSeg, proSeg_params, proSeg_weights = _get_proSeg(proSeg_params, proSeg_weights)
    budSeg, budSeg_params, budSeg_weights = _get_budSeg(budSeg_params, budSeg_weights)
    
    masks, probs, flows = proSeg.run(interpolated, proSeg_params, proSeg_weights)
    masks = track_proliferating(masks)
    
    buds, _, _ = budSeg.run(interpolated, budSeg_params, budSeg_weights)
    
    lineage = LineageConstruction(masks, buds, forwardskip=3)
    daughters, mothers = lineage.computeLineages()
    
    if interp_intervals:
        masks, probs, flows = [deinterpolate(mask_type, interp_locs) for mask_type in [masks, probs, flows]]
    
    return masks, probs, flows, daughters, mothers


def _get_proSeg(proSeg_params, proSeg_weights)->ProSeg: 
    if not proSeg_params:
        proSeg_params = ProSeg.hyperparams
    if not proSeg_weights:
        proSeg_weights = produce_weight_path("proSeg", "proSeg")
    return ProSeg(proSeg_params, proSeg_weights), proSeg_params, proSeg_weights

def _get_budSeg(budSeg_params, budSeg_weights)->BudSeg: 
    if not budSeg_params:
        budSeg_params = BudSeg.hyperparams
    if not budSeg_weights:
        budSeg_weights = produce_weight_path("budSeg", "budSeg")
    return BudSeg(budSeg_params, budSeg_weights), budSeg_params, budSeg_weights


def fiest_full_lifecycle(ims: np.ndarray, interpolation_intervals:Optional[List[dict]]=None, 
                         proSeg_params:Optional[dict]=None, matSeg_params:Optional[dict]=None, 
                         spoSeg_params: Optional[dict]=None):
    '''Implementation of Frame Interpolation Enhanced Single-cell Tracking (FIEST) from the paper 'Deep learning-driven 
    imaging of cell division and cell growth across an entire eukaryotic life cycle' for tracking of timeseries movies of
    yeasts that mate, sporulate, and proliferate asexually. This novel algorithm is based on enhancing the overlap of single 
    cell masks in consecutive images through deep learning video frame interpolation. 
    
    Steps:
    1) segment ims with matSeg, proSeg, and spoSeg
    '''
    return



##### one of the steps of the fiest full lifecycle tracking code
##### TODO: port this to main tracking module



def track_correct_artilife(art_masks:np.ndarray, shock_period:Optional[List[int]]=None):

    Arti_v = 11
    cell_prob = 0.5
    flow_threshold = 0.9
    disk_size = 6


    Masks3 = art_masks

    im_no1 = 0
    im_no = Masks3.shape[2]
    mm = range(im_no) # time points to track
    
    IS1 = np.copy(Masks3[:,:,im_no1]).astype('uint16') # start tracking at first time point # plt.imshow(IS1)
    IS1 = remove_artif(IS1, disk_size) # remove artifacts and start tracking at first time point # plt.imshow(IS1)
    masks = np.zeros((IS1.shape[0], IS1.shape[1], im_no)) # contains the re-labeled masks according to labels in the last tp mask
    masks[:,:,im_no1] = IS1.copy() # first time point defines indexing; IS1 is first segmentation output


    """
    Allocate a mask for cells where there's a gap in the segmentation; IblankG is updated within the loops it1 and itG
    """
    IblankG = np.zeros(IS1.shape, dtype="uint16")
    tic = time.time()
    for it0 in mm: # notice IS1 will be updated in the loop
        print(f'it0={it0}')
        # Load the future cellpose mask, IS2: IS2 is the current image being re-indexed for tracking
        IS2 = np.copy(Masks3[:,:,it0]).astype('uint16') # plt.imshow(IS2)
    
        IS2 = remove_artif(IS2, disk_size) # set disk_size as needed # 5 is ideal to match MATLAB's disk_size=6
        
        IS2C = np.copy(IS2) # plt.imshow(IS2C) # <--- a copy of IS2, gets updated in it1
        IS1B = binar(IS1)
        
        IS3 = IS1B.astype('uint16') * IS2 # past superimposed to future; updated in it1
        tr_cells = np.unique(IS1[IS1 != 0]) # the tracked cells present in the present mask, IS1
        
        gap_cells = np.unique(IblankG[IblankG != 0]) # the tracked cells that had a gap in their segmentation; were not detected in IS1
        cells_tr = np.concatenate((tr_cells, gap_cells)) # all the cells that have been tracked up to this tp for this position

        
        # Allocate space for the re-indexed IS2 according to tracking
        Iblank0 = np.zeros_like(IS1)
        
        # Go to the previously tracked cells and find corresponding index in current tp being processed, IS2 -> Iblank0: mask of previously tracked cells with new position in IS2
        
        if cells_tr.sum() != 0: # this is required in case the mask goes blank because cells mate immediately during germination
            for it1 in np.sort(cells_tr): # cells are processed in order according to birth/appearance
                IS5 = (IS1 == it1).copy() # go to past mask, IS1, to look for the cell
                IS6A = np.uint16(thin(IS5, max_num_iter=1)) * IS3

                if IS5.sum() == 0: # if the cell was missing in the past mask; look at the gaps in segmentation, otherwise, continue to look at the past mask
                    IS5 = (IblankG == it1).copy()
                    IS6A = np.uint16(thin(IS5, max_num_iter=1)) * IS2C
                    IblankG[IblankG == it1] = 0 # remove the cell from the segmentation gap mask - it'll be updated in the past mask for next round of processing

                # Find the tracked cell's corresponding index in IS2, update IS3 and IS2C to avoid overwriting cells 
                if IS6A.sum() != 0:
                    IS2ind = 0 if not IS6A[IS6A != 0].any() else mode(IS6A[IS6A != 0])[0]
                    Iblank0[IS2 == IS2ind] = it1
                    IS3[IS3 == IS2ind] = 0
                    IS2C[IS2 == IS2ind] = 0

            # Define cells with segmentation gap, update IblankG, the segmentation gap mask
            seg_gap = np.setdiff1d(tr_cells, np.unique(Iblank0)) # cells in the past mask (IS1), that were not found in IS2 

            if seg_gap.size > 0:
                for itG in seg_gap:
                    IblankG[IS1 == itG] = itG

            # Define cells that were not relabelled in IS2; these are the buds and new cells entering the frame
            Iblank0B = Iblank0.copy()
            Iblank0B[Iblank0 != 0] = 1
            ISB = IS2 * np.uint16(1 - Iblank0B)
            
            # Add new cells to the mask with a new index Iblank0->Iblank, Iblank0 with new cells added
            newcells = np.unique(ISB[ISB != 0])
            Iblank = Iblank0.copy()
            A = 1

            if newcells.size > 0:
                for it2 in newcells:
                    Iblank[IS2 == it2] = np.max(cells_tr) + A # create new index that hasn't been present in tracking
                    A += 1

            masks[:, :, it0] = np.uint16(Iblank).copy() #<---convert tracked mask to uint16 and store
            IS1 = masks[:, :, it0].copy() # IS1, past mask, is updated for next iteration of it0

        else:
            masks[:, :, it0] = IS2.copy()
            IS1 = IS2.copy()

    toc = time.time()
    print(f'Elapsed time is {toc - tic} seconds.')


    """
    Vizualize All Ob
    """
    obj = np.unique(masks)
    no_obj = int(np.max(obj))
    im_no = masks.shape[2]
    all_ob = np.zeros((no_obj, im_no))

    tic = time.time()

    for ccell in range(1, no_obj + 1):
        Maa = (masks == ccell)

        for i in range(im_no):
            pix = np.sum(Maa[:, :, i])
            all_ob[ccell-1, i] = pix

    """
    Tracks as a tensor
    """

    im_no = masks.shape[2]
    # Find all unique non-zero cell identifiers across all time points
    ccell2 = np.unique(masks[masks != 0])
    # Initialize Mask2 with zeros of the same shape as masks
    Mask2 = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2]))


    # TODO: instead of np use cpypy
    # Process each unique cell ID
    for itt3 in range(len(ccell2)):  # cells
        pix3 = np.where(masks == ccell2[itt3])
        Mask2[pix3] = itt3 + 1  # ID starts from 1

    """
    Get Cell Presence
    """

    # Get cell presence
    Mask3 = Mask2.copy()
    numbM = im_no
    obj = np.unique(Mask3)
    no_obj1 = int(obj.max())
    A = 1

    tp_im = np.zeros((no_obj1, im_no))

    for cel in range(1, no_obj1+1):
        Ma = (Mask3 == cel)

        for ih in range(numbM):
            if Ma[:, :, ih].sum() != 0:
                tp_im[cel-1, ih] = 1


    # plt.figure()
    # plt.imshow(tp_im, aspect='auto', interpolation="nearest")
    # plt.title("Cell Presence Over Time")
    # plt.xlabel("Time")
    # plt.ylabel("Cells")
    # plt.show()


    """
    Split Inturrupted time series
    """

    tic = time.time()
    for cel in range(1, no_obj1+1):
        print(cel)
        tp_im2 = np.diff(tp_im[cel-1, :])
        tp1 = np.where(tp_im2 == 1)[0]
        tp2 = np.where(tp_im2 == -1)[0]
        maxp = (Mask3[:, :, numbM - 1] == cel).sum()

        if len(tp1) == 1 and len(tp2) == 1 and maxp != 0:  # has one interruption
            for itx in range(tp1[0], numbM):
                tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                Mask3[:, :, itx] = tp3.copy()
            no_obj1 += A
        
        elif len(tp1) == 1 and len(tp2) == 1 and maxp == 0:  # has one interruption
            pass
        
        elif len(tp1) == len(tp2) + 1 and maxp != 0:
            tp2 = np.append(tp2, numbM-1)

            for itb in range(1, len(tp1)):  # starts at 2 because the first cell index remains unchanged
                for itx in range(tp1[itb] + 1, tp2[itb] + 1):
                    tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                    Mask3[:, :, itx] = tp3.copy()
                no_obj1 += A
        
        elif len(tp2) == 0 or len(tp1) == 0:  # it's a normal cell, it's born and stays until the end
            pass
        
        elif len(tp1) == len(tp2):
            if tp1[0] > tp2[0]:
                tp2 = np.append(tp2, numbM-1) #check this throughly
                for itb in range(len(tp1)):
                    for itx in range(tp1[itb]+1, tp2[itb + 1] + 1):
                        tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A) #+1 here
                        Mask3[:, :, itx] = tp3.copy()    
                    no_obj1 += A
            elif tp1[0] < tp2[0]:
                for itb in range(1, len(tp1)): 
                    for itx in range(tp1[itb] + 1, tp2[itb] + 1):  # Inclusive range
                        tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                        Mask3[:, :, itx] = tp3.copy()
                    no_obj1 += A
            elif len(tp2) > 1:
                for itb in range(1, len(tp1)):
                    for itx in range(tp1[itb] + 1, tp2[itb] + 1):
                        tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                        Mask3[:, :, itx] = tp3.copy()    
                    no_obj1 += A
    toc = time.time()
    print(f'Elapsed time is {toc - tic} seconds.')


    """
    Get cell presence 2
    """
    numbM = im_no
    obj = np.unique(Mask3)

    # Get cell presence 2
    tp_im = np.zeros((int(max(obj)), im_no))

    for cel in range(1, int(max(obj)) + 1):
        Ma = (Mask3 == cel)

        for ih in range(numbM):
            if Ma[:, :, ih].sum() != 0:
                tp_im[cel-1, ih] = 1


    # plt.figure()
    # plt.imshow(tp_im, aspect='auto', interpolation="nearest")
    # plt.title("Cell Presence Over Time")
    # plt.xlabel("Time")
    # plt.ylabel("Cells")
    # plt.show()

    """
    Get good cells

    """
    cell_artifacts = np.zeros(tp_im.shape[0])

    for it05 in range(tp_im.shape[0]):
        arti = np.where(np.diff(tp_im[it05, :]) == -1)[0]  # Find artifacts in the time series

        if arti.size > 0:
            cell_artifacts[it05] = it05 + 1  # Mark cells with artifacts

    goodcells = np.setdiff1d(np.arange(1, tp_im.shape[0] + 1), cell_artifacts[cell_artifacts != 0])  # Identify good cells


    # plt.figure()
    # plt.imshow(tp_im, aspect='auto', interpolation="nearest")
    # plt.title("Cell Presence Over Time")
    # plt.xlabel("Time")
    # plt.ylabel("Cells")
    # plt.show()


    """
    Tracks as a tensor 2
    """
    im_no = Mask3.shape[2]
    Mask4 = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2]))

    for itt3 in range(goodcells.size):
        pix3 = np.where(Mask3 == goodcells[itt3])
        Mask4[pix3] = itt3 + 1  # IDs start from 1


    """
    # Get cell presence 3
    """

    Mask5 = Mask4.copy()
    numbM = im_no
    obj = np.unique(Mask4)
    no_obj1 = int(obj.max())
    A = 1

    tp_im = np.zeros((no_obj1, im_no))

    for cel in range(1, no_obj1+1):
        Ma = (Mask5 == cel)

        for ih in range(numbM):
            if Ma[:, :, ih].sum() != 0:
                tp_im[cel-1, ih] = 1

    # plt.figure()
    # plt.imshow(tp_im, aspect='auto', interpolation="nearest")
    # plt.title("Cell Presence Over Time")
    # plt.xlabel("Time")
    # plt.ylabel("Cells")
    # plt.show()

    #######
    cell_exists0 = np.zeros((2, tp_im.shape[0]))
    for itt2 in range(tp_im.shape[0]):
        # Find indices of non-zero elements
        non_zero_indices = np.where(tp_im[itt2, :] != 0)[0]
        
        # If there are non-zero elements, get first and last
        if non_zero_indices.size > 0:
            first_non_zero = non_zero_indices[0]
            last_non_zero = non_zero_indices[-1]
        else:
            first_non_zero = -1  # Or any placeholder value for rows without non-zero elements
            last_non_zero = -1   # Or any placeholder value for rows without non-zero elements
        
        cell_exists0[:, itt2] = [first_non_zero, last_non_zero]

    sortOrder = sorted(range(cell_exists0.shape[1]), key=lambda i: cell_exists0[0, i])
    ########
        

    # Reorder the array based on the sorted indices
    cell_exists = cell_exists0[:, sortOrder]

    # Re-label
    Mask6 = np.zeros_like(Mask5)
        
    for itt3 in range(len(sortOrder)):
        pix3 = np.where(Mask5 == sortOrder[itt3] + 1)  # here
        Mask6[pix3] = itt3 + 1# reassign

    """
    # Get cell presence 4
    """
    Mask7 = Mask6.copy()
    numbM = im_no
    obj = np.unique(Mask6)
    no_obj1 = int(obj.max())
    A = 1

    tic = time.time()
    tp_im = np.zeros((no_obj1, im_no))
    for cel in range(1, no_obj1 + 1):
        tp_im[cel - 1, :] = ((Mask7 == cel).sum(axis=(0, 1)) != 0).astype(int)
    toc = time.time()
    print(f'Elapsed time is {toc - tic} seconds.')

    # plt.figure()
    # plt.imshow(tp_im, aspect='auto', interpolation="nearest")
    # plt.title("Cell Presence Over Time")
    # plt.xlabel("Time")
    # plt.ylabel("Cells")
    # plt.show()

    # Calculate size
    obj = np.unique(Mask7)
    no_obj = int(np.max(obj))
    im_no = Mask7.shape[2]
    all_ob = np.zeros((no_obj, im_no))


    tic = time.time()
    for ccell in range(1, no_obj + 1):
        Maa = (Mask7 == ccell)

        for i in range(im_no):
            pix = np.sum(Maa[:, :, i])
            all_ob[ccell-1, i] = pix
    toc = time.time()
    print(f'Elapsed time is {toc - tic} seconds.')