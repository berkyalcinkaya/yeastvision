from scipy.stats import mode
from skimage.morphology import thin
from yeastvision.track.fiest.utils import remove_artif, binar, OAM_23121_tp3
import numpy as np
from typing import List, Optional
import time
import logging

logger = logging.getLogger(__name__)

##### one of the steps of the fiest full lifecycle tracking code
##### TODO: port this to main tracking module
## corresponds to step5 of the pipeline
def track_general_masks(Art_MT, disk_size = 6, transpose_out=False):
    '''
    Tracking alogrithm for budding yeast with built-in track corrections. Iterative frame-assignmentper object asssumes frame-to-frame overlap. For fast
    growing colonies or cells that move significantly between frames, explore using interpolation before calling this function (see yeastvision.track.fiest.track for the FIEST algorithm or 
    yeastvision.ims.interpolate for fine tuned control of interpolation)
    
    Args: 
        Art-MT (np.ndarray or list of np.ndarray) - 
        disk_size (int, default 6)
        
    Returns:
        np.ndarray - '''
    logger.info("----- General tracking of Saccharomyces cerevisiae using mask overlap----")
    Masks3 = Art_MT
        
    im_no1 = 0
    im_no = len(Masks3)
    mm = range(im_no) # time points to track
    
    IS1 = np.copy(Masks3[im_no1]).astype('uint16') # start tracking at first time point # plt.imshow(IS1)
    IS1 = remove_artif(IS1, disk_size) # remove artifacts and start tracking at first time point # plt.imshow(IS1)
    masks = np.zeros((IS1.shape[0], IS1.shape[1], im_no)) # contains the re-labeled masks according to labels in the last tp mask
    masks[:,:,im_no1] = IS1.copy()
 
    """
    Allocate a mask for cells where there's a gap in the segmentation; IblankG is updated within the loops it1 and itG
    """
    IblankG = np.zeros(IS1.shape, dtype="uint16")
    tic = time.time()
    for it0 in mm: # notice IS1 will be updated in the loop it0=0
        #print(f'it0={it0}')
        # Load the future cellpose mask, IS2: IS2 is the current image being re-indexed for tracking
        IS2 = np.copy(Masks3[it0]).astype('uint16') # plt.imshow(IS2)
 
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
    logger.info(f'GENERAL TRACKING: Elapsed time is {toc - tic} seconds.')


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


    # instead of np use cpypy?
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


    """
    Split Interrupted time series
    """

    tic = time.time()
    for cel in range(1, no_obj1+1):
        #print(cel)
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
    logger.info(f'GENERAL TRACKING: Elapsed time is {toc - tic} seconds.')


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

    """
    Get good cells

    """
    cell_artifacts = np.zeros(tp_im.shape[0])

    for it05 in range(tp_im.shape[0]):
        arti = np.where(np.diff(tp_im[it05, :]) == -1)[0]  # Find artifacts in the time series

        if arti.size > 0:
            cell_artifacts[it05] = it05 + 1  # Mark cells with artifacts

    goodcells = np.setdiff1d(np.arange(1, tp_im.shape[0] + 1), cell_artifacts[cell_artifacts != 0])  # Identify good cells


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
    art_cell_exists = cell_exists


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
    logger.info(f'GENERAL TRACKING: Elapsed time is {toc - tic} seconds.')

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

    # plt.imshow(Mask7[:,:,im_no-1])
    # MASK7 CONTAINS THE TRACKED SPO-CORRECTED PROSEG MASKS, INCLUDING THE MATING EVENTS 


    tic = time.time()
    for ccell in range(1, no_obj + 1):
        Maa = (Mask7 == ccell)

        for i in range(im_no):
            pix = np.sum(Maa[:, :, i])
            all_ob[ccell-1, i] = pix
    toc = time.time()
    logger.info(f'GENERAL TRACKING: Elapsed time is {toc - tic} seconds.')


    ## COMES FROM: THESE VARIABLES WILL BE USED IN THE LATER STEPS ALONG WITH MASK7
    Art_obj = np.unique(Mask7)
    Art_no_obj = int(np.max(obj))
    Art_im_no = Mask7.shape[2]
    Art_all_ob = all_ob
    
    if transpose_out:
        return np.transpose(Mask7, (2, 0, 1)).astype(np.uint16)  
    return Mask7, art_cell_exists