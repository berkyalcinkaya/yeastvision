from typing import List, Optional
import math
import numpy as np
import scipy.io as sio
from skimage.morphology import skeletonize, thin
from scipy.stats import mode
from yeastvision.track.fiest.utils import cal_allob, cal_celldata, replace_none_with_empty_array

# step 3
def track_mating(matSeg_output:np.ndarray, mating_interval:List[int], shock_period:Optional[List[int]]):
    '''
    Tracking of mating cells
    '''
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
    #print("Tracking All Detections")
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

        #print(xx + 1)




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
    Mat_cell_data = cal_celldata(all_obj, ccel) # CHANGED FROM CELL DATA TOfinal_mat_cell_data

    #plt.imshow(all_obj, extent=[0, len(rang2), 0, ccel], aspect='auto', interpolation='nearest')

    mat_no_obj = ccel # this has to be changed to mat_obj!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    rang3=range(len(MATC[0]))

    Matmasks = [MATC[0][i] for i in rang3] # MATMASKS CONTAIN ONLY TRACKED MASKS FOR MATING CELLS    
    Matmasks =replace_none_with_empty_array(Matmasks)
    #     Mat_cell_data, mat_no_obj,
    
    return Matmasks, mat_no_obj, Mat_cell_data, cell_data