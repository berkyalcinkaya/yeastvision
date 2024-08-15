#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 19:17:47 2024

@author: samarth
"""

import numpy as np
from skimage import io, morphology
from skimage.morphology import thin, disk, binary_opening, dilation, opening
from scipy.ndimage import binary_opening, binary_dilation
from scipy.io import savemat
import os
from glob import glob
import time
import cv2
import matplotlib.pyplot as plt
from skimage import img_as_uint
from scipy import stats
import scipy.io as sio
import h5py

Arti_v = 11
cell_prob = 0.5
flow_threshold = 0.9
disk_size = 3

def OAM_231216_bina(IS1):
    IS1B = IS1.copy()
    IS1B[IS1 != 0] = 1
    return IS1B

def OAM_230919_remove_artif(I2A,disk_size): # I2A = IS2 % disk radius is 3 for ~500x~1000, 6 for larger images
# we need a function to define the disk size base in the average cell size
    I2AA=np.copy(I2A) #   plt.imshow(IS2)
    # Applying logical operation and morphological opening
    I2A1 =OAM_231216_bina(I2A);#binar(I2A) plt.imshow(I2A1)     plt.imshow(I2A)
    
    # nam1=mask_paths[0]
    # img_name2 = nam1.replace("masks.tif", "masks_I2A1.tif")
    # imageio.imwrite(img_name2, I2A1, format='tif')
 

    # Create a disk-shaped structuring element with radius 3
    selem = disk(disk_size)
    # Perform morphological opening
    I2B = opening(I2A1, selem)
    # plt.imshow(I2B)
    # nam1=mask_paths[0]
    # img_name2 = nam1.replace("masks.tif", "masks_I2B.tif")
    # imageio.imwrite(img_name2, I2B, format='tif')
       
    # Morphological dilation   plt.imshow(I2B)
    I2C = dilation(I2B, disk(disk_size))  # Adjust the disk size as needed
    #  plt.imshow(I2C)
    # Element-wise multiplication of I2A with I2C
 
    # nam1=mask_paths[0]
    # img_name2 = nam1.replace("masks.tif", "masks_I2C.tif")
    # imageio.imwrite(img_name2, I2C, format='tif')

    I3 = I2AA * I2C # plt.imshow(I3)

    # Extract unique objects
    objs = np.unique(I3)
    objs = objs[1:len(objs)]
    
    # Initialize an image of zeros with the same size as I2A
    I4 = np.uint16(np.zeros((I3.shape[0], I3.shape[1])))
    # Mapping the original image values where they match the unique objects
    AZ=1
    for obj in objs:
        I4[I2A == obj] = AZ
        AZ=AZ+1

    # nam1=mask_paths[0]
    # img_name2 = nam1.replace("masks.tif", "masks_I4.tif")
    # imageio.imwrite(img_name2, I4, format='tif')
    #     # Returning the final image  plt.imshow(I4)
    
    return I4





# def remove_artif(I2A):  # I2A = IS2
#     I2A = IS2.copy()
#     plt.imshow(I2A)
#     I2AA = np.copy(I2A)  # plt.imshow(IS2)
#     # Applying logical operation and morphological opening
#     I2A1 = OAM_231216_bina(I2AA)  # binar(I2A) plt.imshow(I2A1)
#     plt.imshow(I2A1)
#     # I2B = binary_opening(I2A1, disk(6))  # Adjust the disk size as needed

#     # do opening equivalent to the disk 6 in matlab
#     structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     # Perform morphological opening
#     I2B = cv2.morphologyEx(I2A1, cv2.MORPH_OPEN, structuring_element)
#     plt.imshow(I2B)

#     # Morphological dilation   plt.imshow(I2B)
#     I2C = dilation(I2B, disk(6))  # Adjust the disk size as needed
#     plt.imshow(I2C)
#     # Element-wise multiplication of I2A with I2C
#     I3 = I2AA * I2C
#     plt.imshow(I3)

#     # Extract unique objects
#     objs = np.unique(I3)
#     objs = objs[1:len(objs)]

#     # Initialize an image of zeros with the same size as I2A
#     I4 = np.uint16(np.zeros((I3.shape[0], I3.shape[1])))
#     # Mapping the original image values where they match the unique objects
#     cnt = 1
#     for obj in objs:
#         I4[I2A == obj] = cnt
#         cnt = cnt + 1

#     uq4 = np.unique(I4)
#     uq3 = np.unique(I3)
#     diff = np.setdiff1d(uq3, uq4)
#     plt.imshow(I4)

#     # Returning the final image  plt.imshow(I4)

#     return I4


# def OAM_230919_remove_artif(I2A):  # I2A = IS2

#     I2A = IS2.copy()
#     plt.imshow(I2A)
#     I2AA = np.copy(I2A)  # plt.imshow(IS2)
#     # Applying logical operation and morphological opening
#     # binar(I2A) plt.imshow(I2A1)     plt.imshow(I2AA)
#     I2A1 = OAM_231216_bina(I2A)
#     # I2B = binary_opening(I2A1, disk(6))  # Adjust the disk size as needed

#     # do opening equivalent to the disk 6 in matlab
#     structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     # Perform morphological opening
#     I2B = cv2.morphologyEx(I2A1, cv2.MORPH_OPEN, structuring_element)

#     # Morphological dilation   plt.imshow(I2B)
#     I2C = dilation(I2B, disk(6))  # Adjust the disk size as needed
#     #  plt.imshow(I2C)
#     # Element-wise multiplication of I2A with I2C
#     I3 = I2AA * I2C  # plt.imshow(I3)

#     # Extract unique objects
#     objs = np.unique(I3)
#     objs = objs[1:len(objs)]

#     # Initialize an image of zeros with the same size as I2A
#     I4 = np.uint16(np.zeros((I3.shape[0], I3.shape[1])))
#     # Mapping the original image values where they match the unique objects
#     for obj in objs:
#         I4[I2A == obj] = obj

#     # Returning the final image  plt.imshow(I4)

#     return I4


# def OAM_230919_remove_artif(I2A):
#     # I2B = binary_opening(I2A.astype(bool), structure=morphology.disk(6))
#     # I2C = binary_dilation(I2B, structure=morphology.disk(6))

#     # I3 = I2A.astype(np.uint16) * I2C.astype(np.uint16)
#     # objs = np.unique(I3)

#     # I4 = np.zeros_like(I2A)

#     # for obj in objs:
#     #     I4[I2A == obj] = obj

#     # I = I4.astype(np.uint16)
#     # return I
#     # Applying logical operation and morphological opening
#     I2A = IS2.copy()
#     plt.imshow(I2A)
#     I2AA = OAM_231216_bina(I2A)
#     plt.imshow(I2AA)
#     structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     # Perform morphological opening
#     I2B = cv2.morphologyEx(I2AA, cv2.MORPH_OPEN, structuring_element)
#     plt.imshow(I2B)

#     plt.imshow(I2B - I2A)
#     I22 = I2B + I2A

#     I2C = dilation(I2B, disk(6))
#     plt.imshow(I2C)

#     # I2B = binary_opening(I2A > 0, disk(5.95))
#     # plt.imshow(I2B)

#     # # Morphological dilation
#     # I2C = dilation(I2B, disk(5.95))
#     # Element-wise multiplication of I2A with I2C
#     I3 = I2A * I2C
#     plt.imshow(I3)
#     # Extract unique objects
#     objs = np.unique(I3)
#     # objs = objs[1:len(objs)]

#     # Initialize an image of zeros with the same size as I2A
#     I4 = np.zeros_like(I2A)

#     # Mapping the original image values where they match the unique objects
#     for obj in objs:
#         I4[I2A == obj] = obj

#     # Returning the final image

#     return img_as_uint(I4)


def OAM_23121_tp3(M, cel, no_obj1, A):
    tp3 = M.copy()
    tp3[tp3 == cel] = no_obj1 + A
    return tp3


pos = 'Pos0_2'
# path to the segmented tet masks
path = '/Users/samarth/Documents/MATLAB/Full_Life_Cycle_tracking/Python_Track_Test/Pro/'
sav_path = '/Users/samarth/Documents/MATLAB/Full_Life_Cycle_tracking/saved_res/py_res/'
art_mask_path = path
file_list = sorted(glob(os.path.join(art_mask_path, '*_Ph3_000_cp_masks.tif')))
# file_list = file_list[:25]
mm = range(len(file_list))  # time points to track
# Load the first mask that begins the indexing for all the cells; IS1 is updated to most recently processed tracked mask at the end of it0
# Start tracking at first timepoint
IS1 = io.imread(file_list[0]).astype(np.uint16)
# Remove artifacts and start tracking at first timepoint
IS1 = OAM_230919_remove_artif(IS1, disk_size)
# plt.imshow(IS1)
# Contains the re-labeled masks according to labels in the last tp mask
masks = np.zeros((IS1.shape[0], IS1.shape[1], len(mm)), dtype=np.uint16)
# First timepoint defines indexing; IS1 is first segmentation output
masks[:, :, mm[0]] = IS1

# Allocate a mask for cells where there's a gap in the segmentation; IblankG is updated within the loops it1 & itG
IblankG = np.zeros_like(IS1)
tic = time.time()
for it0 in mm:
    print(f'it0={it0}')

    # Load the future cellpose mask, IS2: IS2 is the current image being re-indexed for tracking
    # it0 = 1
    IS2 = io.imread(file_list[it0]).astype(np.uint16)
    # plt.imshow(IS2)
    IS22 = IS2.copy()
    IS2 = OAM_230919_remove_artif(IS2, disk_size)  # Remove artifacts
    # plt.imshow(IS2)
    # uq22 = np.unique(IS22)
    # uq2 = np.unique(IS2)
    # uq222 = np.setdiff1d(uq2, uq22)
    # IS33 = IS22 - IS2
    # uq33 = np.unique(IS33)
    IS2C = IS2.copy()
    IS1B = OAM_231216_bina(IS1)
    # plt.imshow(IS1B)
    
    # save_path = os.path.join(sav_path, f'IS1B_{it0}.mat')
    # savemat(save_path, {
    #     'IS1B_py': IS1B,
    # })

    IS3 = IS1B.astype(np.uint16) * IS2
    # plt.imshow(IS3)
    
    # save_path = os.path.join(sav_path, f'IS3_{it0}.mat')
    # savemat(save_path, {
    #     'IS3_py': IS3,
    # })
    
    tr_cells = np.unique(IS1[IS1 != 0])
    # tr_cells = tr_cells[1:]
    gap_cells = np.unique(IblankG[IblankG != 0])
    # gap_cells = gap_cells[1:]
    cells_tr = np.concatenate([tr_cells, gap_cells])
    
    # save_path = os.path.join(sav_path, f'cells_tr_py_{it0}.mat')
    # savemat(save_path, {
    #     'cells_tr1_py': cells_tr,
    #     'gap_cells_py': gap_cells,
    #     'tr_cells_py':tr_cells
    # })

    # Allocate space for the re-indexed IS2 according to tracking
    Iblank0 = np.zeros_like(IS1)
    # is2ind_arr = []
    if cells_tr.sum() != 0:  # This is required in case the mask goes blank because cells mate immediately during germination
        for it1 in sorted(cells_tr):
            IS5 = (IS1 == it1)
            IS5A = IS5.copy()
            # plt.imshow(IS5A)
            IS6AA = thin(IS5A, 1).astype(np.uint16)
            IS6A = np.multiply(IS6AA, IS3)
            # sum(sum(IS6A))
            # plt.imshow(IS6A)
            # plt.imshow(IS3)
            # IS6A = morphology.thin(IS5).astype(np.uint16) * IS3
            # IS6A = np.multiply(IS6AA, IS3)
            # plt.imshow(IS6A)
            
            # save_path = os.path.join(sav_path, f'cells_tr_{it0}.mat')
            # savemat(save_path, {
            #     'cells_tr1': cells_tr,
            #     'tr_cells1':tr_cells,
            #     'IS5_1': IS5
            # })

            if IS5A.sum() == 0:  # If the cell was missing in past mask; look at the gaps in segmentation otherwise continue to look at the past mask
                IS5A = (IblankG == it1)
                IS5AA = IS5A.copy()
                # plt.imshow(IS5A)
                IS5AB = thin(IS5AA, 1).astype(np.uint16)
                IS6A = np.multiply(IS5AB, IS2C)
                # Remove the cell from segmentation gap mask - it'll be in the updated past mask for next round of processing
                IblankG[IblankG == it1] = 0

            if IS6A.sum() != 0:
                IS2ind = np.uint16((np.bincount(IS6A[IS6A != 0]).argmax()))
                # is2ind_arr.append(IS2ind);
                Iblank0[IS2 == IS2ind] = it1
                # plt.imshow(Iblank0)
                # print(sum(sum(Iblank0)))
                IS3[IS3 == IS2ind] = 0
                # plt.imshow(IS3)
                IS2C[IS2 == IS2ind] = 0
                # plt.imshow(IS2C)
                # save_path = os.path.join(sav_path, f'IS3_IS2c_{it0}.mat')
                # savemat(save_path, {
                #     "Iblank0_py": Iblank0,
                #     "IS3_py": IS3,
                #     "IS2C_py": IS2C,
                # })
        
        # plt.imshow(Iblank0)
        # plt.imshow(IS2)
        # plt.imshow(IS3);
        # save_path = os.path.join(sav_path, f'IS3_IS2c_{it0}.mat')
        # savemat(save_path, {
        #     "Iblank0_py": Iblank0,
        #     "IS3_py": IS3,
        #     "IS2_py": IS2,
        # })

        seg_gap = np.setdiff1d(tr_cells, np.unique(Iblank0))
        # seg_gap = seg_gap[1:]

        if seg_gap.size > 0:
            for itG in seg_gap:
                # itG = 16
                IblankG[IS1 == itG] = itG
        # plt.imshow(IblankG)

        Iblank0B = Iblank0.copy()
        Iblank0B[Iblank0 != 0] = 1
        # plt.imshow(Iblank0B)
        ISB = IS2 * (1 - Iblank0B).astype(np.uint16)
        # plt.imshow(ISB)

        newcells = np.unique(ISB[ISB != 0])
        # newcells = newcells[1:]
        Iblank = Iblank0.copy()
        A = 1

        if newcells.size > 0:
            for it2 in newcells:
                Iblank[IS2 == it2] = max(cells_tr) + A
                A += 1
        
        # plt.imshow(Iblank);
        masks[:, :, mm[it0]] = Iblank.astype(np.uint16)
        # plt.imshow(masks[:, :, mm[it0]])
        IS1 = masks[:, :, mm[it0]]
    else:
        masks[:, :, mm[it0]] = IS2
        IS1 = IS2

toc = time.time()
print(f'Elapsed time is {toc - tic} seconds.')
for it4 in range(len(file_list)):
    plt.imshow(masks[:, :, it4])

# msum = [];
# for it4 in range(len(file_list)):
#     msum.append(sum(sum(masks[:,:,it4])));


"""
Vizualize All Ob
"""
# obj = np.unique(masks)
# no_obj = int(np.max(obj))
# im_no = masks.shape[2]
# all_ob = np.zeros((no_obj, im_no))

# tic = time.time()

# for ccell in range(1, no_obj + 1):
#     Maa = (masks == ccell)

#     for i in range(im_no):
#         pix = np.sum(Maa[:, :, i])
#         all_ob[ccell-1, i] = pix

# plt.figure()
# plt.imshow(all_ob, aspect='auto', cmap='viridis')
# plt.title("all_obj")
# plt.xlabel("Time")
# plt.ylabel("Cells")
# plt.show()

# sio.savemat('st3_allob.mat', {
#     "all_obj_py": all_ob
# })

"""
Tracks as a tensor
"""

im_no = masks.shape[2]
# Find all unique non-zero cell identifiers across all time points
ccell2 = np.unique(masks[masks != 0])
# Initialize Mask2 with zeros of the same shape as masks
# Mask2 = np.zeros_like(masks)
Mask2 = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2]))


# TODO: instead of np use cpypy
# Process each unique cell ID
for itt3 in range(len(ccell2)):  # cells
    pix3 = np.where(masks == ccell2[itt3])
    Mask2[pix3] = itt3 + 1  # ID starts from 1

# for it4 in range(1, len(file_list)):
#     plt.imshow(Mask2[:, :, it4])


# save_path = os.path.join(sav_path, 'Mask2_py.mat')
# savemat(save_path, {
#     'Mask2_py': Mask2,
# })
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


plt.figure()
plt.imshow(tp_im, aspect='auto')
plt.title("Cell Presence Over Time")
plt.xlabel("Time")
plt.ylabel("Cells")
plt.show()

# right
tic = time.time()
for cel in range(1, no_obj1+1):
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

for it4 in range(len(file_list)):
    plt.imshow(Mask3[:, :, it4])

numbM = im_no
obj = np.unique(Mask3)

# Get cell presence 2
tp_im = np.zeros((int(max(obj)), im_no))

for cel in range(1, int(max(obj)) + 1):
    Ma = (Mask3 == cel)

    for ih in range(numbM):
        if Ma[:, :, ih].sum() != 0:
            tp_im[cel-1, ih] = 1


plt.figure()
plt.imshow(tp_im, aspect='auto')
plt.title("Cell Presence Over Time")
plt.xlabel("Time")
plt.ylabel("Cells")
plt.show()

save_path = os.path.join(sav_path, 'tp_im_py.mat')
savemat(save_path, {
    'tp_im_py': tp_im,
})



#done

# Get good cells
cell_artifacts = np.zeros(tp_im.shape[0])
# goodcells = np.setdiff1d(range(
#     1, tp_im.shape[0] + 1), np.where((tp_im[:, 1:] - tp_im[:, :-1]) == -1)[0] + 1)
for it05 in range(tp_im.shape[0]):
    arti = np.where(np.diff(tp_im[it05, :]) == -1)[0]  # Find artifacts in the time series

    if arti.size > 0:
        cell_artifacts[it05] = it05 + 1  # Mark cells with artifacts

goodcells = np.setdiff1d(np.arange(1, tp_im.shape[0] + 1), cell_artifacts[cell_artifacts != 0])  # Identify good cells

# display tp_im
plt.figure()
plt.imshow(tp_im, aspect='auto')
plt.title("Cell Presence Over Time")
plt.xlabel("Time")
plt.ylabel("Cells")
plt.show()

# Tracks as a tensor 2
im_no = Mask3.shape[2]
Mask4 = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2]))

for itt3 in range(goodcells.size):
    pix3 = np.where(Mask3 == goodcells[itt3])
    Mask4[pix3] = itt3 + 1  # IDs start from 1

# Get cell presence 3
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

plt.figure()
plt.imshow(tp_im, aspect='auto')
plt.title("Cell Presence Over Time")
plt.xlabel("Time")
plt.ylabel("Cells")
plt.show()

sio.savemat('st3_allob.mat', {
    'tp_im_py': tp_im
})

#Done!!

# cell_exists0 = np.zeros((2, tp_im.shape[0]))

# for itt2 in range(tp_im.shape[0]):
#     cell_exists0[0, itt2] = np.argmax(tp_im[itt2, :] != 0)
#     cell_exists0[1, itt2] = len(tp_im[itt2, :]) - \
#         1 - np.argmax(tp_im[itt2, ::-1] != 0)

# sortOrder = np.argsort(cell_exists0[0, :])
# cell_exists = cell_exists0[:, sortOrder]

cell_exists0 = np.zeros((2, tp_im.shape[0]))

# Calculate the first and last detection
# for itt2 in range(tp_im.shape[0]):
#     first_detection = np.argmax(tp_im[itt2, :] != 0)
#     last_detection = tp_im.shape[1] - np.argmax(tp_im[itt2, ::-1] != 0) - 1
#     cell_exists0[:, itt2] = [first_detection, last_detection]

# # # Sort based on the first detection
# sortOrder = np.argsort(cell_exists0[0, :]) #check the sort !TODO:
# # sorted_order = sortOrder.copy()   
# cell_exists = cell_exists0[:, sortOrder]
# sortOrder = sorted(sortOrder)
# sortOrder.append(len(sortOrder))
# sortOrder = sortOrder[1:]

# save_path = os.path.join(sav_path, 'tp_im_py.mat')
# savemat(save_path, {
#     'sortOrder_py': sortOrder
# })

# for itt2 in range(tp_im.shape[0]):
#     first_nonzero = np.argmax(tp_im[itt2, :] != 0)
#     last_nonzero = len(tp_im[itt2, :]) - np.argmax(tp_im[itt2, ::-1] != 0) - 1
#     cell_exists0[:, itt2] = [first_nonzero, last_nonzero]  # Calculate the first and last detection

# sortOrder = np.argsort(cell_exists0[0, :])

# cell_exists = cell_exists0[:, sortOrder]

######
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

# # Sort by the first non-zero index
# sortOrder = np.argsort(cell_exists0[0,:])

sortOrder = sorted(range(cell_exists0.shape[1]), key=lambda i: cell_exists0[0, i])
########
# for x in range(len(cell_exists0[0])):
#     cell_exists0[0][x] += 1
# for x in range(len(cell_exists0[1])):
#     cell_exists0[1][x] += 1
    
    
sio.savemat('st3_allob.mat', {
    'cell_exists0_py': cell_exists0,
    'sortOrder_py': sortOrder
})

# for x in range(len(sortOrder)):
#     sortOrder[x] += 1

# Reorder the array based on the sorted indices
cell_exists = cell_exists0[:, sortOrder]


# Re-label
Mask6 = np.zeros_like(Mask5)
    

for itt3 in range(len(sortOrder)):
    pix3 = np.where(Mask5 == sortOrder[itt3] + 1)  # here
    Mask6[pix3] = itt3 + 1# reassign

# Get cell presence 3
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

plt.figure()
plt.imshow(tp_im, aspect='auto')
plt.title("Cell Presence Over Time")
plt.xlabel("Time")
plt.ylabel("Cells")
plt.show()


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

sio.savemat('st3_allob.mat', {
    'all_ob_py': all_ob,
    'Mask7_py': Mask7
})

plt.figure()
plt.imshow(all_ob, aspect='auto', cmap='viridis')
plt.title("Cell Sizes Over Time")
plt.xlabel("Time")
plt.ylabel("Cells")
plt.show()

save_path = os.path.join(sav_path, f'ART_Track0_{im_no}.mat')
savemat(save_path, {
    'all_ob_py': all_ob,
    'Mask7_py': Mask7,
    'no_obj_py': no_obj,
    'im_no_py': im_no,
    'ccell2_py': ccell2,
    'cell_exists': cell_exists,
    'cell_prob': cell_prob,
    'Arti_v': Arti_v,
    'flow_threshold': flow_threshold
})

# Save results
# np.savez_compressed(os.path.join(sav_path, f'{pos}_ART_Track0_{im_no}'), all_ob=all_ob, Mask7=Mask7, no_obj=no_obj, im_no=im_no, ccell2=ccell2, cell_exists=cell_exists, cell_prob=cell_prob, Arti_v=Arti_v, flow_threshold=flow_threshold)

print(pos)
