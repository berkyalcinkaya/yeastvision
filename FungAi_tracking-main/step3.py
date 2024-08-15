# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:02:00 2024

@author: samarth
"""
import os
import math
import numpy as np
import scipy.io as sio
from skimage.io import imread
from skimage.morphology import skeletonize, thin
from functions.SR_240222_cal_allob import cal_allob
from functions.SR_240222_cal_celldata import cal_celldata 
from scipy.stats import mode
import matplotlib.pyplot as plt


# Define thresholds and parameters
pos = 'Pos0_2'
path = f'/Users/samarth/Documents/MATLAB/Full_Life_Cycle_tracking/Pos0_2/'
# path = f'/Users/samarth/Documents/MATLAB/Full_Life_Cycle_tracking/Python_Track_Test/MAT/'
sav_path = '/Users/samarth/Documents/MATLAB/Full_Life_Cycle_tracking/saved_res/py_res/'
shock_period = [122, 134]

# Load image file names
file_names = [f for f in os.listdir(path) if f.endswith('_Ph3_000_MAT16_18_masks.tif')]
file_numbers = [int(f.split('img_')[1].split('_Ph3_000_MAT16_18_masks.tif')[0]) for f in file_names]

sorted_indices = np.argsort(file_numbers)
sorted_numbers = np.array(file_numbers)[sorted_indices]
mat_masks_path = [os.path.join(path, file_names[i]) for i in sorted_indices]

# Read images
mat_masks = [None] * (sorted_numbers[-1] + 1)
for i, img_path in enumerate(mat_masks_path):
    mat_masks[sorted_numbers[i]] = imread(img_path)

for i in range(min(sorted_numbers), len(mat_masks)):
    if mat_masks[i] is None:
        mat_masks[i] = np.zeros_like(mat_masks[min(sorted_numbers)], dtype=np.uint16)

# Remove shock-induced timepoints
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
    for i in range(start-1, end):
        if i in rang3:
            rang3.remove(i)

# Correction Code
all_obj = cal_allob(ccel, MATC, rang)
cell_data = cal_celldata(all_obj, ccel)

sio.savemat('st3_allob.mat', {
    "all_obj_py": all_obj
})

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

no_obj = ccel
# in matlab the array size is 777 filled with values after 240th index, try increasing?
Matmasks = [MATC[0][i] for i in rang]

def replace_none_with_empty_array(data):
    if isinstance(data, list):
        return [replace_none_with_empty_array(item) for item in data]
    elif data is None:
        return np.array([])
    else:
        return data
    
Matmasks = replace_none_with_empty_array(Matmasks)
mat_masks_original = replace_none_with_empty_array(mat_masks_original)
# Save results
sio.savemat(f'{sav_path}{pos}_sam_MAT_16_18_Track.mat', {
    "Matmasks": Matmasks,
    "no_obj": no_obj,
    "all_obj": all_obj,
    "cell_data": cell_data,
    "rang": rang,
    "rang3": rang3,
    "shock_period": shock_period,
    "mat_masks_original": mat_masks_original,
    "start": start
}, do_compression=True)
