


# from functions.SR_240222_cal_allob import cal_allob this should load callobj1 in the final version
import os
import math
import numpy as np
import re
import scipy.io as sio
from skimage.io import imread
from skimage.morphology import skeletonize
from scipy.stats import mode
from yeastvision.track.fiest.utils import (cal_celldata, binar, cal_allob, cal_allob1, 
                                           resize_image, replace_none_with_empty_array, OAM_23121_tp3, 
                                           remove_artif, cal_allob2)
import matplotlib.pyplot as plt

# for step 4

#from skimage.transform import resize
#from skimage.morphology import binary_erosion, disk, erosion, square
from skimage.filters import threshold_otsu
from skimage.morphology import disk, erosion, square

# for step 5
from skimage.morphology import thin, dilation, opening #  binary_openin, and disk taken out
from glob import glob
import time

# for step 6
from skimage.transform import resize
from skimage.measure import regionprops, label
from scipy.ndimage import binary_fill_holes


def FL(artilife_output, spoSeg_output, matSeg_output, mating_interval, tetrad_interval, movie_length, shock_period):
    
    cell_prob = 0.5
    flow_threshold = 0.9
    disk_size = 6 # step 5

    # Define thresholds and parameters
    thresh_percent = 0.015 # step 2
    thresh_remove_last_mask = 10 # step 2
    thresh_next_cell = 400 # step 2
    thresh = 80 # step 2 
    
    # tetrad masks are extended to go from index 0 of the movie to the end of the tetrad interval
    tet_masks = [None] * tetrad_interval[-1]
    im_shape = spoSeg_output[0].shape
    for i in range(len(tet_masks)): 
        if i >= tetrad_interval[0]:
            tet_masks[i] = spoSeg_output[i]
        else:
            tet_masks[i] = np.zeros(im_shape, dtype=np.uint16)
    
    # Remove shock-induced timepoints, if possible
    if shock_period:
        for start, end in [shock_period]:
            for i in range(start, end):
                tet_masks[i] = None

    ###  find timepoint of first TET detection
    start = -1
    for its in range(len(tet_masks)):
        if tet_masks[its] is not None and np.sum(tet_masks[its]) > 0:
            start = its
            break

    ### determine the period to track or rang
    if start != -1:
        rang = range(start, len(tet_masks))
        I2 = tet_masks[start]
        A = np.zeros_like(tet_masks[start])
    else:
        rang = range(len(tet_masks))
        I2 = tet_masks[0]
        A = np.zeros_like(tet_masks[0])


    # Tracking all detections
    #  actual tracking loop

    IS6 = np.zeros_like(I2)
    TETC = [None] * 2
    TETC[0] = [None] * len(tet_masks)
    TETC[1] = [None] * len(tet_masks)
    xx = start
    rang2 = rang
    ccel = 1

    while xx != -1:
        k = 0
        for im_no in rang2:
            # im_no = 72
            I2 = tet_masks[im_no] if ccel == 1 else TETC[1][im_no]
            if I2 is None or I2.size == 0:
                continue
            if im_no == min(rang2):
                ind1 = np.unique(I2)[1:]  # Exclude background
                I3 = (I2 == ind1[0])
                I3A = I3.copy()
            else:
                I3A = IS6.copy()

            I3A = skeletonize(binar(I3A))
            I2A = I2.copy()
            I3B = I3A.astype(np.uint16) * I2A.astype(np.uint16)
            ind = mode(I3B[I3B != 0])[0]

            if (ind == 0 or math.isnan(ind)) and ccel == 1:
                k += 1
                if k > thresh_next_cell:
                    for im_no_1 in range(im_no, rang[-1] + 1):
                        if tet_masks[im_no_1] is not None:
                            TETC[0][im_no_1] = np.zeros_like(tet_masks[start])
                        TETC[1][im_no_1] = tet_masks[im_no_1]
                    break
                else:
                    TETC[0][im_no] = I3B.copy()
                    TETC[1][im_no] = I2A.copy()
                    continue
            elif (ind == 0 or math.isnan(ind)) and ccel != 1:
                k += 1
                if k > thresh_next_cell:
                    break
                else:
                    continue

            k = 0
            pix = np.where(I2A == ind)
            pix0 = np.where(I2A != ind)
            
            # pix = np.flatnonzero(I2A == ind)
            # pix0 = np.flatnonzero(I2A != ind)
            
            I2A[pix] = ccel
            I2A[pix0] = 0
            
            IS6 = I2A.copy()
            I22 = np.zeros_like(I2)
            
            pix1 = np.where(IS6 == ccel)
            
            I2[pix1] = 0
            pix2 = np.unique(I2)
            pix2 = pix2[1:]  # Exclude background

            if ccel == 1:
                for ity, p2 in enumerate(pix2):
                    pix4 = np.where(I2 == p2)
                    I22[pix4] = ity + 1
                TETC[0][im_no] = IS6.copy()
            else:
                if len(pix2) > 0:
                    for ity, p2 in enumerate(pix2):
                        pix4 = np.where(I2 == p2)
                        I22[pix4] = ity + 1
                else:
                    I22 = I2.copy()
                IS61 = TETC[0][im_no]
                IS61[pix] = ccel
                TETC[0][im_no] = IS61.astype(np.uint16)

            TETC[1][im_no] = I22.copy()

        xx = -1
        for i in rang:
            if TETC[1][i] is not None and np.sum(TETC[1][i]) > 0:
                xx = i
                break
        ccel += 1
        rang2 = range(xx, len(tet_masks))
        print(xx + 1)


    ccel -= 1  # number of cells tracked

    # Removing the shock-induced points from rang
    rang3 = list(rang)
    for start, end in [shock_period]:
        for i in range(start-1, end):
            if i in rang3:
                rang3.remove(i)

    all_obj = cal_allob1(ccel, TETC, rang)

    # x_scale = 200
    # y_scale = 4
    # plt.imshow(all_obj, extent=[0, x_scale, 0, y_scale], aspect='auto',interpolation='nearest')

    # sio.savemat(os.path.join(sav_path, "art_py.mat"), {
    #                 'all_ob_py': all_obj
    #             })
    cell_data = cal_celldata(all_obj, ccel) ## double check values

    k = 1
    cell_artifacts = []
    for iv in range(ccel):
        if cell_data[iv, 2] < thresh_percent * len(rang3) or cell_data[iv, 4] > thresh:
            cell_artifacts.append(iv + 1)
            k += 1

    all_ccel = list(range(1, ccel + 1))

    if len(cell_artifacts) > 0:
        cell_artifacts = list(set(cell_artifacts))
        for iv in cell_artifacts:
            for its in rang3:
                pix = np.where(TETC[0][its] == iv)
                TETC[0][its][pix] = 0

    # Retaining and relabeling the new cells
    good_cells = sorted(set(all_ccel) - set(cell_artifacts))

    for iv in range(len(good_cells)):
        for its in rang3:
            pix = np.where(TETC[0][its] == good_cells[iv])
            TETC[0][its][pix] = iv + 1

    # Correcting the SpoSeg track masks or filling the empty spaces between the first and last appearance
    # Removing artifacts
    all_obj1 = cal_allob1(len(good_cells), TETC, rang)
    # plt.imshow(all_obj1, extent=[0, x_scale, 0, y_scale], aspect='auto',interpolation='nearest')
    cell_data1 = cal_celldata(all_obj1, len(good_cells))

    for iv in range(len(good_cells)):
        for its in range(int(cell_data1[iv, 0] + 1), int(cell_data1[iv, 1])):
            if all_obj1[iv, its] == 0:
                prev = np.where(all_obj1[iv, :its] > 0)[0][-1]
                all_obj1[iv, its] = all_obj1[iv, prev]
                pix = np.where(TETC[0][prev] == iv + 1)
                TETC[0][its][pix] = iv + 1

    # Cell array that contains the fully tracked TetSeg masks
    TETmasks = [TETC[0][i] for i in range(len(TETC[0]))]

    # Calculate the size of tetrads
    def cal_allob2(ccel, TETC, rang):
        # Initialize the all_obj array with zeros
        all_obj = np.zeros((ccel, len(TETC)))

        for iv in range(ccel):  # Adjusted to 1-based index
            for its in rang:
                if TETC[its] is not None: #and np.sum(TETC[its]) > 0:  # Check if the array is not None and not empty
                    all_obj[iv, its] = np.sum(TETC[its] == iv + 1)  # Adjusted for 1-based index logic
                else:
                    all_obj[iv, its] = -1

        return all_obj

    TET_obj = len(good_cells)
    all_obj_final = cal_allob2(TET_obj, TETmasks, list(range(len(TETmasks))))
    plt.imshow(all_obj1, aspect='auto', interpolation='nearest')
    TET_Size = all_obj_final.copy()

    # Calculate first detection and last detection of tetrads
    TET_exists = np.zeros((2, TET_obj), dtype=int)
    for iv in range(TET_obj):
        TET_exists[0, iv] = np.where(TET_Size[iv, :] > 0)[0][0]  # 1st occurrence
        TET_exists[1, iv] = np.where(TET_Size[iv, :] > 0)[0][-1]  # last occurrence

    tet_masks_exists_tp = rang3
    TETmasks = replace_none_with_empty_array(TETmasks)


    # STEP 4
    
    Art_MT = [artilife_output[i] for i in range(len(artilife_output))]
    shock_period[0] =  shock_period[0]-1
    shock_period[1] = shock_period[1]-1
    for iv in range(TET_obj):
            # iv = 0;
            if TET_exists[1][iv] >= shock_period[0]-1:#!  shock period corrected produce a single integer!!!
                tp_end = shock_period[0]
            else:
                tp_end = TET_exists[1][iv] 

            for its in range(TET_exists[0][iv], tp_end):# minus one added
                # its = 42 ;
                A1 = Art_MT[its].astype(np.double)
                plt.imshow(A1)
                if shock_period[0]-1 <= its <= shock_period[1]:#!!shock period corrected produce a single integer!!!!!!!!!!!!!!!!!!!!
                    T1 = (TETmasks[shock_period[0]-1] == iv + 1).astype(np.double)#! TETmasks correctd and shock period corrected produce a single integer[]
                    thresh = 0.6
                else:
                    T1 = (TETmasks[its] == iv + 1).astype(np.double)#!!!!!!!!!!!!!!!!!!  
                    thresh = 0.95 # plt.imshow(T1)

                T1 = resize_image(T1, A1.shape,).astype(np.float64)
            #  plt.imshow(T1, aspect='auto',interpolation='nearest')
                # plt.imshow(T1)
                Im1 = T1 > threshold_otsu(T1)
                # plt.imshow(Im1)
                Im2 = erosion(Im1, square(9))
                # plt.imshow(Im2)
                Im3 = A1 * Im2
                # plt.imshow(Im3)
                

                pix11 = []
                pix1 = np.unique(A1[Im3 != 0])
                for it2 in pix1:
                    r1 = np.sum(Im3 == it2) / np.sum(Im3 > 0)
                    if r1 > 0.2:
                        pix11.append(it2)

                if len(pix11) == 1:
                    r = np.sum(A1 == pix11[0]) / np.sum(T1)
                    if r > thresh:
                        pass
                    else:
                        Art_MT[its][A1 == pix11[0]] = 0
                        Art_MT[its][T1 == 1] = np.max(Art_MT[its]) + 1
                elif not pix11:
                    Art_MT[its][T1 == 1] = np.max(Art_MT[its]) + 1
                else:
                    for it2 in pix11:
                        Art_MT[its][A1 == it2] = 0
                    Art_MT[its][T1 == 1] = np.max(Art_MT[its]) + 1


    #plt.imshow(Art_MT[148])

    for iv in range(TET_obj):
            # iv = 0
            if TET_exists[1][iv] > shock_period[1] and TET_exists[0][iv] < shock_period[0]:
                s1 = np.sum(TETmasks[shock_period[1]+1] == iv+1)#!!!!!!!!!!!!!!!!!!!!!!
                for its in range(shock_period[1]+1, TET_exists[1][iv]):#!!!!!!!!!!!!!!!!!!!!!!!!
                    # its = 134;
                    A1 = Art_MT[its].astype(np.double)
                    # plt.imshow(A1)
                    T1 = (TETmasks[its] == iv + 1).astype(np.double)
                    # plt.imshow(T1)
                    
                    

                    s2 = np.sum(TETmasks[its] == iv + 1)
                    if its == TET_exists[1][iv]:
                        s3 = np.sum(TETmasks[its] == iv + 1)
                    else:
                        s3 = np.sum(TETmasks[its + 1] == iv + 1)

                    if s2 < s1 - 0.1 * s1:
                        if s3 > s2 + 0.1 * s2:
                            T1 = (TETmasks[its - 1] == iv + 1).astype(np.double)
                        else:
                            break

                    s1 = s2
                    #T1 = resize(T1, A1.shape, order=0, preserve_range=True)
                    T1 = resize_image(T1, A1.shape,).astype(np.float64)
                    # plt.imshow(T1)
                    Im1 = T1 > threshold_otsu(T1)
                    # plt.imshow(Im1)
                    Im2 = erosion(Im1, square(9))
                    # plt.imshow(Im2)
                    Im3 = A1 * Im2
                    # plt.imshow(Im3)

                    pix11 = []
                    pix1 = np.unique(A1[Im3 != 0])
                    for it2 in pix1:
                        r1 = np.sum(Im3 == it2) / np.sum(Im3 > 0)
                        if r1 > 0.2:
                            pix11.append(it2)

                    if len(pix11) == 1:
                        r = np.sum(A1 == pix11[0]) / np.sum(T1)
                        if r > thresh:
                            pass
                        else:
                            Art_MT[its][A1 == pix11[0]] = 0
                            Art_MT[its][T1 == 1] = np.max(Art_MT[its]) + 1
                    elif not pix11:
                        Art_MT[its][T1 == 1] = np.max(Art_MT[its]) + 1
                    else:
                        for it2 in pix11:
                            Art_MT[its][A1 == it2] = 0
                        Art_MT[its][T1 == 1] = np.max(Art_MT[its]) + 1
                        
                        
    # # transpose is only to save to matlab and compare no needed for python            
    # MAT1=np.transpose(Art_MT,(1,2,0));
    # sio.savemat(os.path.join(sav_path, f'{pos}_ART_Masks_py.mat'), {"Art_MT": MAT1, "shock_period": shock_period})

    ### STEP 5

    #Helper Functions
        
    # def export(variable, name):
    #     sio.savemat(sav_path + f'{name}_py.mat', { f'{name}_py': variable})


    
    """
    Load the first mask that begins the indexing for all the cells; IS1 is updated to most recently processed tracked mask at the end of it0
    """    
    take_img = False
    art_mask_path= ""
    ## Obtain and correctly load Masks3 as tensor
    if not take_img:
        Masks3 = Art_MT
            
        im_no1 = 0
        im_no = len(Masks3)
        mm = range(im_no) # time points to track
        
        IS1 = np.copy(Masks3[im_no1]).astype('uint16') # start tracking at first time point # plt.imshow(IS1)
        IS1 = remove_artif(IS1, disk_size) # remove artifacts and start tracking at first time point # plt.imshow(IS1)
        masks = np.zeros((IS1.shape[0], IS1.shape[1], im_no)) # contains the re-labeled masks according to labels in the last tp mask
        masks[:,:,im_no1] = IS1.copy() # first time point defines indexing; IS1 is first segmentation output
        
    else:
        file_list = sorted(glob(os.path.join(art_mask_path, '*_Ph3_000_cp_masks.tif')))
        mm = range(len(file_list))
        IS1 = imread(file_list[0]).astype(np.uint16)
        # Remove artifacts and start tracking at first timepoint
        IS1 = remove_artif(IS1, disk_size)
        # plt.imshow(IS1)
        # Contains the re-labeled masks according to labels in the last tp mask
        masks = np.zeros((IS1.shape[0], IS1.shape[1], len(mm)), dtype=np.uint16)
        # First timepoint defines indexing; IS1 is first segmentation output
        masks[:, :, mm[0]] = IS1

    """
    Allocate a mask for cells where there's a gap in the segmentation; IblankG is updated within the loops it1 and itG
    """
    IblankG = np.zeros(IS1.shape, dtype="uint16")
    tic = time.time()
    for it0 in mm: # notice IS1 will be updated in the loop it0=0
        print(f'it0={it0}')
        # Load the future cellpose mask, IS2: IS2 is the current image being re-indexed for tracking
        if not take_img:
            IS2 = np.copy(Masks3[it0]).astype('uint16') # plt.imshow(IS2)
        else:
            IS2 = imread(file_list[it0]).astype(np.uint16)
            
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

    # plt.figure()
    # plt.imshow(all_ob, aspect='auto', cmap='viridis', interpolation="nearest")
    # plt.title("all_obj")
    # plt.xlabel("Time")
    # plt.ylabel("Cells")
    # plt.show()

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


    # plt.figure()
    # plt.imshow(tp_im, aspect='auto', interpolation="nearest")
    # plt.title("Cell Presence Over Time")
    # plt.xlabel("Time")
    # plt.ylabel("Cells")
    # plt.show()


    """
    Split Interrupted time series
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



    Art_obj = np.unique(Mask7)
    Art_no_obj = int(np.max(obj))
    Art_im_no = Mask7.shape[2]
    Art_all_ob = all_ob

    plt.figure()
    plt.imshow(Art_all_ob, aspect='auto', cmap='viridis',interpolation="nearest")
    plt.title("Cell Sizes Over Time")
    plt.xlabel("Time")
    plt.ylabel("Cells")
    plt.show()



    # # just for matlab comparison
    # from scipy.io import savemat
    # save_path = os.path.join(sav_path, f'ART_Track0_{im_no}_py.mat')
    # savemat(save_path, {
    #     'all_ob_py': all_ob,
    #     'Mask7_py': Mask7,
    #     'no_obj_py': no_obj,
    #     'im_no_py': im_no,
    #     'ccell2_py': ccell2,
    #     'cell_exists': cell_exists,
    #     'cell_prob': cell_prob,
    #     'Arti_v': Arti_v,
    #     'flow_threshold': flow_threshold
    # })


    # Step 3

    # Define thresholds and parameters

    #shock_period = [122, 134]

    # Load image file names
    # tetrad masks are extended to go from index 0 of the movie to the end of the tetrad interval
    mat_masks = [None] * mating_interval[-1]
    im_shape = matSeg_output[0].shape
    for i in range(len(mat_masks)):
        if i >= mating_interval[0]:
            mat_masks[i] = matSeg_output[i]
        else:
            mat_masks[i] = np.zeros(im_shape, dtype=np.uint16)
    

    # Remove shock-induced timepoints
    mat_masks_original = mat_masks.copy()
    for start, end in [shock_period]:
        for i in range(start, end):
            mat_masks[i] = None
    start = -1
    for its in range(len(mat_masks)):
        # if mat_masks[its] is not None and np.sum(mat_masks[its]) > 0:
        if mat_masks[its] is not None and np.sum(mat_masks[its]) > 0:
            start = its
            break

    # Tracking all detections
    print("Tracking All Detections")
    if start != -1:
        rang = range(start, len(mat_masks))
        I2 = mat_masks[start]
        A = np.zeros_like(mat_masks[start])
    else:
        rang = range(len(mat_masks))
        I2 = mat_masks[0]
        A = np.zeros_like(mat_masks[0])

    IS6 = np.zeros_like(I2)
    MATC = [None] * 2
    MATC[0] = [None] * len(mat_masks)
    MATC[1] = [None] * len(mat_masks)
    xx = start
    rang2 = rang
    ccel = 1

    # uq = mat_masks[50]
    # print(np.unique(uq)[0:])
    # plt.figure()
    # plt.imshow(np.uint16(uq), cmap='gray')
    # plt.title('uq')
    # plt.show()

    while xx != -1:
        for im_no in rang2:

            if ccel == 1:
                I2 = mat_masks[im_no]
            else:
                I2 = MATC[1][im_no]
                
            if I2 is None or I2.size == 0:
                continue
                
            if im_no == min(rang2):
                ind1 = np.unique(I2)[1:]  # Exclude background
                I3 = (I2 == ind1[0])
                I3A = I3.copy()
            else:
                I3A = np.copy(IS6)
                    
            I3A = skeletonize(I3A > 0)
            I2A = np.copy(I2)
            I3B = I3A.astype(np.uint16) * I2A.astype(np.uint16)
            
            ind = mode(I3B[I3B != 0])[0]

            if (ind == 0 or math.isnan(ind)) and ccel == 1:
                MATC[0][im_no] = I3B
                MATC[1][im_no] = I2A
                continue
            elif (ind == 0 or math.isnan(ind)) and ccel != 1:
                continue
            
            pix = np.where(I2A == ind)
            pix0 = np.where(I2A != ind)
            
            I2A[pix] = ccel
            I2A[pix0] = 0
            IS6 = np.copy(I2A)

            I22 = np.zeros_like(I2)
            pix1 = np.where(IS6 == ccel)
            I2[pix1] = 0
            
            pix2 = np.unique(I2)
            pix2 = pix2[1:] # Exclude background
            
            if ccel == 1:
                # for ity in range(len(pix2)):
                #     pix4 = np.where(I2 == pix2[ity])
                #     I22[pix4] = ity + 1'
                for ity, p2 in enumerate(pix2):
                    pix4 = np.where(I2 == p2)
                    I22[pix4] = ity + 1
                MATC[0][im_no] = np.copy(IS6)
            else:
                if len(pix2) > 0:
                    # for ity in range(len(pix2)):
                    #     pix4 = np.where(I2 == pix2[ity])
                    #     I22[pix4] = ity + 1
                    for ity, p2 in enumerate(pix2):
                        pix4 = np.where(I2 == p2)
                        I22[pix4] = ity + 1
                else:
                    I22 = I2.copy()
                IS61 = np.copy(MATC[0][im_no])
                IS61[pix] = ccel
                MATC[0][im_no] = IS61.astype(np.uint16)

            MATC[1][im_no] = np.copy(I22)
            
        xx = -1
        for i in rang:
            if MATC[1][i] is not None and MATC[1][i].size > 0 and np.sum(MATC[1][i]) > 0:
                xx = i
                break
        ccel += 1
        rang2 = range(xx, len(mat_masks))

        print(xx + 1)




    ccel -= 1  # number of cells tracked


    # Removing the shock-induced points from rang
    rang3 = list(rang)
    for start, end in [shock_period]:
        for i in range(start-1, end):
            if i in rang3:
                rang3.remove(i)

    # Correction Code
    all_obj = cal_allob(ccel, MATC, rang)
    cell_data = cal_celldata(all_obj, ccel)


    #plt.imshow(all_obj, extent=[0, len(rang2), 0, ccel], aspect='auto', interpolation='nearest')


    # sio.savemat('st3_allob.mat', {
    #     "all_obj_py": all_obj
    # })

    for iv in range(ccel):
        if np.any(all_obj[iv, min(rang):shock_period[-1]] > 0):
            if all_obj[iv, shock_period[-1] + 1] != 0:
                for its in range(shock_period[-1] + 1, rang[-1] + 1):
                    if all_obj[iv, its] != -1:
                        pix = np.where(MATC[0][its] == iv + 1)
                        MATC[0][its][pix] = 0
                        all_obj[iv, its] = np.sum(MATC[0][its] == iv + 1)

    cell_data = cal_celldata(all_obj, ccel)

    k = 1
    cell_artifacts = []
    for iv in range(ccel):
        if cell_data[iv, 2] == 1 or cell_data[iv, 4] > 80:
            cell_artifacts.append(iv + 1)
            k += 1

    all_ccel = list(range(1, ccel + 1))

    if cell_artifacts:
        cell_artifacts = list(set(cell_artifacts))
        for iv in cell_artifacts:
            for its in rang3:
                pix = np.where(MATC[0][its] == iv + 1)
                MATC[0][its][pix] = 0

    good_cells = sorted(set(all_ccel) - set(cell_artifacts))

    for iv in range(len(good_cells)):
        for its in rang3:
            pix = np.where(MATC[0][its] == good_cells[iv])
            MATC[0][its][pix] = iv + 1

    ccel = len(good_cells)
    all_obj = cal_allob(ccel, MATC, rang)
    cell_data = cal_celldata(all_obj, ccel)

    for iv in range(ccel):
        tp_data = {
            iv: [np.diff(np.where(all_obj[iv, :] > 0)[0]), np.where(all_obj[iv, :] > 0)[0]]
        }
        a = np.where(tp_data[iv][0] > 10)[0]
        if len(a) > 0:
            if a[0] == len(tp_data[iv][0]):
                pix = np.where(MATC[0][tp_data[iv][1][a[0] + 1]] == iv + 1)
                MATC[0][tp_data[iv][1][a[0] + 1]][pix] = 0
            else:
                for its in range(np.where(all_obj[iv, :] > 0)[0][0], tp_data[iv][1][a[0] + 1] - 1):
                    pix = np.where(MATC[0][its] == iv + 1)
                    MATC[0][its][pix] = 0

    for iv in range(ccel):
        for its in range(np.where(all_obj[iv, :] > 0)[0][0] + 1, np.where(all_obj[iv, :] > 0)[0][-1]):
            if all_obj[iv, its] == 0:
                prev = np.where(all_obj[iv, :its] > 0)[0][-1]
                all_obj[iv, its] = (all_obj[iv, prev]).copy()
                pix = np.where(MATC[0][prev] == iv + 1)
                MATC[0][its][pix] = iv + 1

    all_obj = cal_allob(ccel, MATC, rang)
    cell_data = cal_celldata(all_obj, ccel)

    # plt.imshow(all_obj, extent=[0, len(rang2), 0, ccel], aspect='auto', interpolation='nearest')

    no_obj = ccel

    rang3=range(len(MATC[0]))

    Matmasks = [MATC[0][i] for i in rang3]

        
    Matmasks =replace_none_with_empty_array(Matmasks)
        

    #MAT1=np.transpose(Matmasks,(1,2,0));

    # # Save results
    # sio.savemat(f'{sav_path}{pos}_sam_MAT_16_18_Track_py.mat', {
    #     "Matmasks": Matmasks,
    #     "no_obj": no_obj,
    #     "all_obj": all_obj,
    #     "cell_data": cell_data,
    #     "rang": rang,
    #     "rang3": rang3,
    #     "shock_period": shock_period,
    #   # "mat_masks_original": mat_masks_original,
    #     "start": start
    # }, do_compression=True)


    ## Step 3
    mat_masks = [None] * mating_interval[-1]
    im_shape = matSeg_output[0].shape
    for i in range(len(mat_masks)):
        if i >= mating_interval[0]:
            mat_masks[i] = matSeg_output[i]
        else:
            mat_masks[i] = np.zeros(im_shape, dtype=np.uint16)

    mat_masks_original = mat_masks.copy()
    for start, end in [shock_period]:
        for i in range(start-1, end):
            mat_masks[i] = None

    start = -1
    for its in range(len(mat_masks)):
        # if mat_masks[its] is not None and np.sum(mat_masks[its]) > 0:
        if mat_masks[its] is not None and np.sum(mat_masks[its]) > 0:
            start = its
            break

    # Tracking all detections
    print("Tracking All Detections")
    if start != -1:
        rang = range(start, len(mat_masks))
        I2 = mat_masks[start]
        A = np.zeros_like(mat_masks[start])
    else:
        rang = range(len(mat_masks))
        I2 = mat_masks[0]
        A = np.zeros_like(mat_masks[0])

    IS6 = np.zeros_like(I2)
    MATC = [None] * 2
    MATC[0] = [None] * len(mat_masks)
    MATC[1] = [None] * len(mat_masks)
    xx = start
    rang2 = rang
    ccel = 1

    # uq = mat_masks[50]
    # print(np.unique(uq)[0:])
    # plt.figure()
    # plt.imshow(np.uint16(uq), cmap='gray')
    # plt.title('uq')
    # plt.show()

    while xx != -1:
        for im_no in rang2:

            if ccel == 1:
                I2 = mat_masks[im_no]
            else:
                I2 = MATC[1][im_no]
                
            if I2 is None or I2.size == 0:
                continue
                
            if im_no == min(rang2):
                ind1 = np.unique(I2)[1:]  # Exclude background
                I3 = (I2 == ind1[0])
                I3A = I3.copy()
            else:
                I3A = np.copy(IS6)
                    
            I3A = skeletonize(I3A > 0)
            I2A = np.copy(I2)
            I3B = I3A.astype(np.uint16) * I2A.astype(np.uint16)
            
            ind = mode(I3B[I3B != 0])[0]

            if (ind == 0 or math.isnan(ind)) and ccel == 1:
                MATC[0][im_no] = I3B
                MATC[1][im_no] = I2A
                continue
            elif (ind == 0 or math.isnan(ind)) and ccel != 1:
                continue
            
            pix = np.where(I2A == ind)
            pix0 = np.where(I2A != ind)
            
            I2A[pix] = ccel
            I2A[pix0] = 0
            IS6 = np.copy(I2A)

            I22 = np.zeros_like(I2)
            pix1 = np.where(IS6 == ccel)
            I2[pix1] = 0
            
            pix2 = np.unique(I2)
            pix2 = pix2[1:] # Exclude background
            
            if ccel == 1:
                # for ity in range(len(pix2)):
                #     pix4 = np.where(I2 == pix2[ity])
                #     I22[pix4] = ity + 1'
                for ity, p2 in enumerate(pix2):
                    pix4 = np.where(I2 == p2)
                    I22[pix4] = ity + 1
                MATC[0][im_no] = np.copy(IS6)
            else:
                if len(pix2) > 0:
                    # for ity in range(len(pix2)):
                    #     pix4 = np.where(I2 == pix2[ity])
                    #     I22[pix4] = ity + 1
                    for ity, p2 in enumerate(pix2):
                        pix4 = np.where(I2 == p2)
                        I22[pix4] = ity + 1
                else:
                    I22 = I2.copy()
                IS61 = np.copy(MATC[0][im_no])
                IS61[pix] = ccel
                MATC[0][im_no] = IS61.astype(np.uint16)

            MATC[1][im_no] = np.copy(I22)
            
        xx = -1
        for i in rang:
            if MATC[1][i] is not None and MATC[1][i].size > 0 and np.sum(MATC[1][i]) > 0:
                xx = i
                break
        ccel += 1
        rang2 = range(xx, len(mat_masks))

        print(xx + 1)




    ccel -= 1  # number of cells tracked


    # Removing the shock-induced points from rang
    rang3 = list(rang)
    for start, end in [shock_period]:
        for i in range(start, end):
            if i in rang3:
                rang3.remove(i)

    # Correction Code
    all_obj = cal_allob(ccel, MATC, rang)
    cell_data = cal_celldata(all_obj, ccel)


    #plt.imshow(all_obj, extent=[0, len(rang2), 0, ccel], aspect='auto', interpolation='nearest')



    # sio.savemat('st3_allob.mat', {
    #     "all_obj_py": all_obj
    # })

    for iv in range(ccel):
        if np.any(all_obj[iv, min(rang):shock_period[-1]] > 0):
            if all_obj[iv, shock_period[-1] + 1] != 0:
                for its in range(shock_period[-1] + 1, rang[-1] + 1):
                    if all_obj[iv, its] != -1:
                        pix = np.where(MATC[0][its] == iv + 1)
                        MATC[0][its][pix] = 0
                        all_obj[iv, its] = np.sum(MATC[0][its] == iv + 1)

    cell_data = cal_celldata(all_obj, ccel)

    k = 1
    cell_artifacts = []
    for iv in range(ccel):
        if cell_data[iv, 2] == 1 or cell_data[iv, 4] > 80:
            cell_artifacts.append(iv + 1)
            k += 1

    all_ccel = list(range(1, ccel + 1))

    if cell_artifacts:
        cell_artifacts = list(set(cell_artifacts))
        for iv in cell_artifacts:
            for its in rang3:
                pix = np.where(MATC[0][its] == iv + 1)
                MATC[0][its][pix] = 0

    good_cells = sorted(set(all_ccel) - set(cell_artifacts))

    for iv in range(len(good_cells)):
        for its in rang3:
            pix = np.where(MATC[0][its] == good_cells[iv])
            MATC[0][its][pix] = iv + 1

    ccel = len(good_cells)
    all_obj = cal_allob(ccel, MATC, rang)
    cell_data = cal_celldata(all_obj, ccel)

    for iv in range(ccel):
        tp_data = {
            iv: [np.diff(np.where(all_obj[iv, :] > 0)[0]), np.where(all_obj[iv, :] > 0)[0]]
        }
        a = np.where(tp_data[iv][0] > 10)[0]
        if len(a) > 0:
            if a[0] == len(tp_data[iv][0]):
                pix = np.where(MATC[0][tp_data[iv][1][a[0] + 1]] == iv + 1)
                MATC[0][tp_data[iv][1][a[0] + 1]][pix] = 0
            else:
                for its in range(np.where(all_obj[iv, :] > 0)[0][0], tp_data[iv][1][a[0] + 1] - 1):
                    pix = np.where(MATC[0][its] == iv + 1)
                    MATC[0][its][pix] = 0

    for iv in range(ccel):
        for its in range(np.where(all_obj[iv, :] > 0)[0][0] + 1, np.where(all_obj[iv, :] > 0)[0][-1]):
            if all_obj[iv, its] == 0:
                prev = np.where(all_obj[iv, :its] > 0)[0][-1]
                all_obj[iv, its] = (all_obj[iv, prev]).copy()
                pix = np.where(MATC[0][prev] == iv + 1)
                MATC[0][its][pix] = iv + 1

    all_obj = cal_allob(ccel, MATC, rang)
    cell_data = cal_celldata(all_obj, ccel)

    #plt.imshow(all_obj, extent=[0, len(rang2), 0, ccel], aspect='auto', interpolation='nearest')

    mat_no_obj = ccel # this has to be changed to mat_obj!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    rang3=range(len(MATC[0]))

    Matmasks = [MATC[0][i] for i in rang3]; 
        
    Matmasks =replace_none_with_empty_array(Matmasks)
        

    # #MAT1=np.transpose(Matmasks,(1,2,0));

    # # Save results
    # sio.savemat(f'{sav_path}{pos}_sam_MAT_16_18_Track_py.mat', {
    #     "Matmasks": Matmasks,
    #     "no_obj": no_obj,
    #     "all_obj": all_obj,
    #     "cell_data": cell_data,
    #     "rang": rang,
    #     "rang3": rang3,
    #     "shock_period": shock_period,
    #   # "mat_masks_original": mat_masks_original,
    #     "start": start
    # }, do_compression=True)



    ## STEP 6


    # Extract variables from loaded data

    if mat_no_obj  != 0:
        MTrack = Matmasks
        art_masks = Mask7 # this is a tensor do not transpose
    
        mat_artifacts = []

        # Resize MTrack to match ART masks
        for its in range(len(MTrack)):
            if MTrack[its].size > 2:
                MTrack[its] = resize(MTrack[its], art_masks[:,:,its].shape, order=0, preserve_range=True, anti_aliasing=False)# art_masks[:,:,its] added to correct for tensor to list format

        tp_end = len(art_masks[0][0])# tensor correction, gets last time point
        if len(MTrack) != tp_end: # loop for adding zeros matrixes 
            for its in range(len(MTrack[its]), tp_end):
                MTrack.append(np.zeros_like(MTrack[int(min(cell_data[:, 0])) - 1], dtype=np.uint16))

        # Correcting mating tracks
        cor_data = np.zeros((3, mat_no_obj ))
        size_cell = np.zeros((mat_no_obj , len(MTrack)))
        morph_data = np.zeros((mat_no_obj , len(MTrack)))
        outlier_tps = [None] * mat_no_obj 
        good_tps = [None] * mat_no_obj 
        

        for iv in range(mat_no_obj ):
            # iv = 0;
            int_range = range(int(cell_data[iv,0]), int(cell_data[iv,1]))  # Adjusting for 0-based indexing
            for its in int_range:
                # its = 240;
                M = np.uint16(MTrack[its] == iv + 1).T
                
    # =============================================================================
    #             plt.figure()
    #             plt.imshow(np.uint16(M), cmap='gray')
    #             plt.title('M')
    #             plt.show()
    # =============================================================================
                
                size_cell[iv, its] = np.sum(M)
                props = regionprops(M)
                morph_data[iv, its] = props[0].eccentricity if props else 0
            cor_data[0, iv] = np.mean(size_cell[iv, int_range])
            cor_data[1, iv] = np.std(size_cell[iv, int_range])
            cor_data[2, iv] = 1 * cor_data[1, iv]
            outlier_tps[iv] = [t for t in int_range if abs(size_cell[iv, t] - cor_data[0, iv]) > cor_data[2, iv]]
            good_tps[iv] = np.setdiff1d(int_range, outlier_tps[iv])



        for iv in range(mat_no_obj ):
            # iv = 0
            int_range = range(int(cell_data[iv,0]), int(cell_data[iv,1]))
            if np.var(morph_data[iv, int_range]) > 0.02:
                mat_artifacts.append(iv)




        for iv in range(mat_no_obj ):
            outlier = sorted(outlier_tps[iv])
            good = sorted(good_tps[iv])
            int_range = range(int(cell_data[iv,0]), int(cell_data[iv,1]))
            while outlier:
                its = min(outlier)
                gtp = max([g for g in good if g < its], default=min([g for g in good if g > its], default=its))
                A = art_masks[:,:,its]
                
    # =============================================================================
    #             plt.figure()
    #             plt.imshow(np.uint16(M), cmap='gray')
    #             plt.title('M')
    #             plt.show()
    # =============================================================================
                
                M1 = (MTrack[gtp] == (iv + 1))
                M2 = thin(M1, 30)
                M3 = A * M2
                
                # plt.figure()
                # plt.imshow(np.uint16(M3), cmap='gray')
                # plt.title('M3')
                # plt.show()
                
                indx = np.unique(A[M3 != 0])
                if indx.size > 0:
                    X1 = np.zeros_like(MTrack[its])
                    for itt2 in indx:
                        if np.sum(M3 == itt2) > 5:
                            X1[A == itt2] = 1
                    X1 = binary_fill_holes(X1)
                    # plt.imshow(X1)
                    X2 = label(X1)
                    if np.max(X2) <= 1 and abs(np.sum(X1) - cor_data[0, iv]) <= 2 * cor_data[1, iv]:
                        MTrack[its][MTrack[its] == (iv + 1)] = 0
                        (MTrack[its])[X1 == 1] = iv + 1
                    else:
                        MTrack[its][MTrack[its] == (iv + 1)] = 0
                        MTrack[its][MTrack[gtp] == (iv + 1)] = iv + 1
                outlier = [o for o in outlier if o != its]
                good.append(its)
                good = sorted(good)




        for iv in range(mat_no_obj ):
            if cell_data[iv,1] != tp_end:
                count = 0
                for its in range(int(cell_data[iv,1])+1, tp_end): # its=156 check plus one is needed!!!!!!!!!!!!!
                    A = art_masks[:,:,its]
                    M1 = (MTrack[its - 1] == (iv + 1))
                    M2 = thin(M1, 30)
                    M3 = A * M2
                    indx = np.unique(A[M3 != 0])
                    if indx.size > 0:
                        X1 = np.zeros_like(MTrack[its])
                        for itt2 in indx:
                            if np.sum(M3 == itt2) > 5:
                                X1[A == itt2] = 1 # plt.imshow(X1)
                        if abs(np.sum(X1) - cor_data[0, iv]) > 2 * cor_data[1, iv]:
                            count += 1
                            MTrack[its][MTrack[its - 1] == (iv + 1)] = iv + 1
                        else:
                            MTrack[its][X1 == 1] = iv + 1
                    else:
                        count += 1
                        MTrack[its][MTrack[its - 1] == (iv + 1)] = iv + 1
                if count / (tp_end - cell_data[iv, 0]) > 0.8:
                    mat_artifacts.append(iv + 1)






        # Remove cell artifacts and rename
        if mat_artifacts:
            all_ccel = list(range(1, mat_no_obj  + 1))
            mat_artifacts = sorted(set(mat_artifacts))
            for iv in mat_artifacts:
                for its in range(len(MTrack)):
                    MTrack[its][MTrack[its] == iv] = 0
            good_cells = sorted(set(all_ccel) - set(mat_artifacts))
            for iv in range(len(good_cells)):
                for its in range(len(MTrack)):
                    MTrack[its][MTrack[its] == good_cells[iv]] = iv + 1
            mat_no_obj = len(good_cells)
            
            

        # Recalculating MAT Data
        
        
    
        all_obj_new = cal_allob2(mat_no_obj, MTrack, list(range(len(MTrack))))
    
        # Display the image with the adjusted scales
        plt.imshow(all_obj_new, interpolation='nearest')
        
    
    
        
        cell_data_new = cal_celldata(all_obj_new, no_obj)

        mat_cell_data = cell_data_new
        mat_all_obj = all_obj_new
        Matmasks = MTrack

        # sio.savemat(f'{sav_path}{pos}_MAT_16_18_Track1_py.mat', {
        #     "Matmasks_py": Matmasks,
        #     "all_obj_py": all_obj,
        #     "cell_data_py": cell_data,
        #     "no_obj_py": no_obj,
        #     "shock_period_py": shock_period,
        #     "mat_artifacts_py": mat_artifacts
        # }, do_compression=True)


    ## STEP 7
    # Removes the mating events from sposeg tracks based on the overlapping tracked indices 

    # this will be fetch all the way from step 5
    # Art_obj = np.unique(Mask7)
    # Art_no_obj = int(np.max(obj))
    # Art_im_no = Mask7.shape[2]
    # Art_all_ob = all_ob


    for iv in range(int(mat_no_obj)): # this  no_onj should be for mat_no_obj
            # iv = 3
            indx_remov = []
            final_indx_remov = []

            for its in range(int(mat_cell_data[iv,0]), int(mat_cell_data[iv,1])):
                # its = 167
                M = Matmasks[its]
                
                # sio.savemat(os.path.join(sav_path, 'M2_py.mat'), {
                #     "M2_py": M2
                # })
                
    # =============================================================================
    #                 plt.figure()
    #                 plt.imshow(M, cmap='gray')
    #                 plt.title('M')
    #                 plt.show()
    # =============================================================================
                
                M0 = (M == iv+1).astype(np.uint16)
                
    # =============================================================================
    #                 plt.figure()
    #                 plt.imshow(M0, cmap='gray')
    #                 plt.title('M0')
    #                 plt.show()
    # =============================================================================
                
                A = Mask7[:,:,its]
                
                # plt.figure()
                # plt.imshow(A, cmap='gray')
                # plt.title('A')
                # plt.show()
                
                
                M1 = binar(M0)
                
    # =============================================================================
    #                 plt.figure()
    #                 plt.imshow(M1, cmap='gray')
    #                 plt.title('M1')
    #                 plt.show()
    # =============================================================================
                
                M2 = thin(M1, 30)
                #M2 = skeletonize(M1)
                
                # plt.figure()
                # plt.imshow(M2, cmap='gray')
                # plt.title('M2')
                # plt.show()
                
                M3 = A * M2
                
                # plt.figure()
                # plt.imshow(M3, cmap='gray')
                # plt.title('M3')
                # plt.show()
                
            
                
                indx = np.unique(A[M3 != 0])
                
                
                if indx.size > 0:
                    for itt2 in indx:
                        if np.sum(M3 == itt2) > 5:
                            indx_remov.append(itt2)
            
            
            
        
            if len(indx_remov) > 0:
                indx_remov_inter = np.unique(indx_remov)
                final_indx_remov = np.unique(indx_remov)
                for itt1 in indx_remov_inter:
                    # itt1 = 6
                    dist_data = -1 * np.ones(len(Mask7[0][0]))
                    for its1 in range(int(mat_cell_data[iv,0]), int(art_cell_exists[1,int(itt1)-1])):
                        # its1 = 141
                        if its1 >= art_cell_exists[0,int(itt1)-1]:
                            M6 = (Mask7[:,:,its1] == itt1)  #  plt.imshow(M6, cmap='gray')
                            M7 = (Matmasks[its1] == iv + 1) #  plt.imshow(M7, cmap='gray')
                            dist_data[its1] = np.sum(M6 * M7) / np.sum(M6)
                    
                    if np.any(dist_data != -1):
                        first_ov = np.where(dist_data != -1)[0][0]
                        last_ov = np.where(dist_data != -1)[0][-1]
                        val_avg = np.median(dist_data[first_ov:last_ov])
                        if val_avg <= 0.4:
                            final_indx_remov = np.setdiff1d(final_indx_remov, itt1)
                
                for its in range(int(mat_cell_data[iv,0]), len(Mask7[0][0])):
                    for itt in final_indx_remov:
                        Mask7[:,:,its][Mask7[:,:,its] == itt] = 0
            print(iv)
    
        
    
    # plt.imshow(Mask7[:,:,its], cmap='gray')    
    
        
    
        
    
        
    
    # shock_period = mat['shock_period']
    # no_obj = art['no_obj']
    # ccell2 = art['ccell2']
    # cell_exists = art['cell_exists']
    # im_no = art['im_no']
    
        # sio.savemat(os.path.join(sav_path, f'{pos}_ART_Track1.mat'), {
        #     "no_obj": no_obj,
        #     "shock_period": shock_period,
        #     "Mask7": Mask7,
        #     "im_no": im_no,
        #     "ccell2": ccell2,
        #     "cell_exists": cell_exists
        # }, do_compression=True)

















