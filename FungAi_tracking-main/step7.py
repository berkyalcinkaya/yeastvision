#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 00:11:43 2024

@author: samarth
"""
import os
import h5py
import numpy as np
import scipy.io as sio
from skimage.morphology import thin, skeletonize
from skimage.filters import threshold_otsu
from shutil import copyfile
import matplotlib.pyplot as plt
from skimage.io import imshow
from functions.OAM_231216_bina import binar

def load_mat(filename):
    try:
        return sio.loadmat(filename)
    except NotImplementedError:
        # Load using h5py for MATLAB v7.3 files
        data = {}
        with h5py.File(filename, 'r') as f:
            for key in f.keys():
                data[key] = np.array(f[key])
        return data
    
def resolve_h5py_reference(data, f):
    if isinstance(data, h5py.Reference):
        return f[data][()]
    return data


# Define parameters
pos = 'Pos0_2'
path = '/Users/samarth/Documents/MATLAB/Full_Life_Cycle_tracking/Tracks2/'  # Path to segment SpoSeg masks
sav_path = '/Users/samarth/Documents/MATLAB/Full_Life_Cycle_tracking/saved_res/'  # Path to save Track

# Removes the mating events from sposeg tracks based on the overlapping tracked indices 



mat_track_path = os.path.join(path)
if any(os.path.isfile(os.path.join(mat_track_path, f)) for f in os.listdir(mat_track_path)):
    file_list = [f for f in os.listdir(mat_track_path) if '_MAT_16_18_Track1' in f]
    file_list = sorted(file_list)

    mat = load_mat(os.path.join(path, file_list[0]))
    
    all_obj = mat['all_obj']
    #Matmasks = mat['Matmasks']
    cell_data = mat['cell_data']
    
    # print("Keys in MAT:", mat.keys())
    
# =============================================================================
#     with h5py.File(os.path.join(path, file_list[0]), 'r') as f:
#         Matmasks = [resolve_h5py_reference(mask, f) for mask in Matmasks]
# =============================================================================
        
    Matmasks = []
    with h5py.File(os.path.join(path, file_list[0]), 'r') as f:
        for i in range(len(f['Matmasks'])):
            tet_masks_refs = f['Matmasks'][i]
            for ref in tet_masks_refs:
                mask = resolve_h5py_reference(ref, f)
                Matmasks.append(mask)
    
    art_track_path = os.path.join(path)
    if any(os.path.isfile(os.path.join(art_track_path, f)) for f in os.listdir(art_track_path)):
        file_list = [f for f in os.listdir(art_track_path) if '_ART_Track' in f]
        file_list = sorted(file_list)
        art = load_mat(os.path.join(path, file_list[0]))
        
        Mask3 = []
        with h5py.File(os.path.join(path, file_list[0]), 'r') as f:
            for i in range(len(f['Mask3'])):
                masks_refs = f['Mask3'][i]
                for ref in masks_refs:
                    mask = resolve_h5py_reference(ref, f)
                    Mask3.append(mask)
        
        
        # TODO! range from no_obj (one obj should contain 1)
        cell_data = mat["cell_data"]
        arr = [1]
        # for iv in arr:
        for iv in range(int(mat['no_obj'][0, 0])):
            # iv = 0
            indx_remov = []
            final_indx_remov = []
            # TODO! remove Hardcode range (check git changes)
            # for its in range(241, 777):  # check for 10 time points
            for its in range(int(mat['cell_data'][0, iv])-1, int(mat['cell_data'][1, iv])):
                # its = 240
                M = Matmasks[its].T
                
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
                
                A = Mask3[its].T
                
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
            
            cell_exists = art["cell_exists"]
            if len(indx_remov) > 0:
                indx_remov_inter = np.unique(indx_remov)
                final_indx_remov = np.unique(indx_remov)
                for itt1 in indx_remov_inter:
                    # itt1 = 1.0
                    dist_data = -1 * np.ones(len(Mask3))
                    for its1 in range(int(mat['cell_data'][0, iv]) - 1, int(art['cell_exists'][int(itt1)-1, 1])):
                        # its1 = 240
                        if its1 >= art['cell_exists'][int(itt1)-1, 0]:
                            M6 = (Mask3[its1] == itt1).T
                            M7 = (Matmasks[its1] == iv + 1).T
                            dist_data[its1] = np.sum(M6 * M7) / np.sum(M6)
                    
                    if np.any(dist_data != -1):
                        first_ov = np.where(dist_data != -1)[0][0]
                        last_ov = np.where(dist_data != -1)[0][-1]
                        val_avg = np.median(dist_data[first_ov:last_ov])
                        if val_avg <= 0.4:
                            final_indx_remov = np.setdiff1d(final_indx_remov, itt1)
                
                for its in range(int(mat['cell_data'][0, iv]), len(Mask3)):
                    for itt in final_indx_remov:
                        Mask3[its][Mask3[its] == itt] = 0
            print(iv)
        
        shock_period = mat['shock_period']
        no_obj = art['no_obj']
        ccell2 = art['ccell2']
        cell_exists = art['cell_exists']
        im_no = art['im_no']
    else:
        art_track_path = os.path.join(sav_path, f'{pos}_ART_Track')
        file_list = [f for f in os.listdir(art_track_path) if os.path.isfile(os.path.join(art_track_path, f))]
        filename_old = file_list[0]
        filename_new = filename_old.replace('_ART_Track', '_ART_Track1')
        copyfile(os.path.join(sav_path, filename_old), os.path.join(sav_path, filename_new))

    sio.savemat(os.path.join(sav_path, f'{pos}_ART_Track1.mat'), {
        "no_obj": no_obj,
        "shock_period": shock_period,
        "Mask3": Mask3,
        "im_no": im_no,
        "ccell2": ccell2,
        "cell_exists": cell_exists
    }, do_compression=True)

