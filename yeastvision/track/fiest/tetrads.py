import numpy as np
from typing import List, Optional
import math
from skimage.transform import resize
from skimage.morphology import skeletonize, erosion, square
from skimage.filters import threshold_otsu
from scipy.stats import mode
from yeastvision.track.fiest.utils import cal_allob2, cal_allob1, cal_celldata, replace_none_with_empty_array, binar

def track_tetrads(spoSeg_output:np.ndarray, tetrad_interval: List, movie_length:int, shock_period:Optional[List[int]]=None):
    # define some global variables
    thresh_percent = 0.015
    thresh_remove_last_mask = 10
    thresh_next_cell = 400
    thresh = 80

    # tetrad masks are extended to go from index 0 of the movie to the end of the tetrad interval
    tet_masks = [None] * tetrad_interval[-1]
    im_shape = spoSeg_output[0].shape
    for i in range(len(tet_masks)): 
        if i >= tetrad_interval[0]:
            tet_masks[i] = spoSeg_output[i]
        else:
            tet_masks[i] = np.zeros(im_shape, dtype=np.uint16)
    
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
        #print(xx + 1)


    ccel -= 1  # number of cells tracked

    # Removing the shock-induced points from rang
    rang3 = list(rang)
    for start, end in [shock_period]:
        for i in range(start, end):
            if i in rang3:
                rang3.remove(i)

    all_obj = cal_allob1(ccel, TETC, rang)
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

    TET_obj = len(good_cells)
    all_obj_final = cal_allob2(TET_obj, TETmasks, list(range(len(TETmasks))))
    TET_Size = all_obj_final.copy()

    # Calculate first detection and last detection of tetrads
    TET_exists = np.zeros((2, TET_obj), dtype=int)
    for iv in range(TET_obj):
        TET_exists[0, iv] = np.where(TET_Size[iv, :] > 0)[0][0]  # 1st occurrence
        TET_exists[1, iv] = np.where(TET_Size[iv, :] > 0)[0][-1]  # last occurrence

    tet_masks_exists_tp = rang3
    TETmasks = replace_none_with_empty_array(TETmasks)
    return TET_obj, TET_exists, TETmasks

