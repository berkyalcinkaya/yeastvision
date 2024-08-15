#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:08:51 2024

@author: samarth
"""

import os
import numpy as np
import scipy.io as sio
import h5py
from skimage.transform import resize
import matplotlib.pyplot as plt

pos = 'Pos0_2'
path = '/Users/samarth/Documents/MATLAB/Full_Life_Cycle_tracking/Tracks2/'
sav_path = '/Users/samarth/Documents/MirandaLabs/tracking_algo/FungAi_tracking/Tracking_toydata_Tracks'

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

# Load tet track
tet_track_path = path
file_list_tet = [f for f in os.listdir(tet_track_path) if '_TET_Track_DS' in f]
tet_track = load_mat(os.path.join(path, file_list_tet[0]))

# Load tet IDs
file_list = [f for f in os.listdir(tet_track_path) if '_TET_ID' in f]
tet_ids = load_mat(os.path.join(path, file_list[0]))

# Load descendants information
desc_path = path
file_list_desc = [f for f in os.listdir(desc_path) if '_descendants_new_art' in f]
descendants = load_mat(os.path.join(path, file_list_desc[0]))

# Load mat track
mat_track_path = path
file_list = [f for f in os.listdir(mat_track_path) if '_MAT_16_18_Track1_DS' in f]
mat_track = load_mat(os.path.join(path, file_list[0]))

    
masks = tet_track['TETmasks']
TET_obj = tet_track['TET_obj']
TET_exists = tet_track['TET_exists']
alive_tets = descendants['alive_tets']
descendants_data = descendants["descendants_data"]
MTrack = mat_track['Matmasks']
shock_period = tet_track['shock_period']
no_obj = mat_track['no_obj']
I3 = descendants['I3']
cell_data = mat_track['cell_data']

with h5py.File(os.path.join(path, file_list[0]), 'r') as f:
    masks = [resolve_h5py_reference(mask, f) for mask in masks]
    MTrack = [resolve_h5py_reference(mtrack, f) for mtrack in MTrack]
    descendants_data = [resolve_h5py_reference(desc, f) for desc in  descendants_data]
    shock_period = resolve_h5py_reference(shock_period, f)
    no_obj = resolve_h5py_reference(no_obj, f)
    I3 = resolve_h5py_reference(I3, f)
    cell_data = resolve_h5py_reference(cell_data, f)

#resolving h5py
resolved_masks = []
with h5py.File(os.path.join(path, file_list_tet[0]), 'r') as f:
    for i in range(len(f['TETmasks'])):
        tet_masks_refs = f['TETmasks'][i]
        for ref in tet_masks_refs:
            mask = resolve_h5py_reference(ref, f)
            resolved_masks.append(mask)
            
MTrack_res = []
with h5py.File(os.path.join(path, file_list[0]), 'r') as f:
    for i in range(len(f['Matmasks'])):
        tet_masks_refs = f['Matmasks'][i]
        for ref in tet_masks_refs:
            mask = resolve_h5py_reference(ref, f)
            MTrack_res.append(mask)
            
resolved_desc = []
with h5py.File(os.path.join(path, file_list_desc[0]), 'r') as f:
    for i in range(len(f['descendants_data'])):
        tet_masks_refs = f['descendants_data'][i]
        for ref in tet_masks_refs:
            desc = resolve_h5py_reference(ref, f)
            resolved_desc.append(desc)

masks = resolved_masks
MTrack = MTrack_res
descendants_data = resolved_desc

if no_obj != 0:  # positive number of MAT Detections
    masks = masks
    MTrack = MTrack 
    shock_period = shock_period.T
    cell_data = cell_data.T
    for i in range(len(masks)):
        if masks[i].size > 2:
            masks[i] = resize(masks[i].astype(np.float64), I3.shape, order=0, preserve_range=True, anti_aliasing=False)

    start = shock_period[0, 1] + 1
    tp_end = len(MTrack)
    int_range = range(int(min(cell_data[:, 0])), int(max(cell_data[:, 1])))

    region = []
    amt = []
    k = 0
    for iv in range(int(no_obj)):
        I12 = np.zeros(MTrack[min(int_range)].shape, dtype=np.uint16)
        kx = 0
        for its in int_range:
            I11 = (MTrack[its] == iv).astype(np.uint16)
            
# =============================================================================
#             plt.figure()
#             plt.imshow(I11, cmap='grey')
#             plt.show()
# =============================================================================
            
            if np.sum(I11) > 0:
                kx += 1
                if 1 <= kx <= 2:
                    I11 *= 1000
            I12 += I11

        I13 = (I12 > 0).astype(np.uint16) * I3.astype(np.uint16)
        
        I13 = I13.T
        pix = np.unique(I13)
        pix = pix[pix != 0]
        if pix.size != 0:
            amt_iv = []
            for p in pix:
                amt_iv.append(np.sum(I13 == p))
            amt.append(amt_iv)
            k += 1
            val, ind = max((v, i) for i, v in enumerate(amt_iv))
            region.append([iv, pix[ind]])

    unique_regions = np.unique(np.array(region)[:, 1])
    cell_arrays = [None] * len(unique_regions)
    for i, current_region in enumerate(unique_regions):
        cells_in_current_region = np.array(region)[np.array(region)[:, 1] == current_region, 1]
        cell_arrays[int(current_region)-1] = cells_in_current_region.astype(np.uint16)

    if alive_tets.size != 0:
        TET_ind = np.zeros((len(unique_regions), 3))
        common_indices = [] * TET_obj
        amt_1 = []
        for iv in range(int(TET_obj)):
            if tet_ids["TET_ID"][0, iv] != -1:
                if iv in alive_tets:
                    if TET_exists[iv, 1] >= shock_period[0, 1] + 1:
                        T1 = (masks[0, shock_period[0, 1] + 1] == iv).astype(np.uint16)
                    else:
                        T1 = (masks[0, TET_exists[iv, 1]] == iv).astype(np.uint16)

                    plt.figure()
                    plt.imshow(T1)
                    plt.show()

                    T2 = I3.astype(np.uint16) * T1
                    TET_ind[iv, 0] = iv
                    TET_ind[iv, 2] = tet_ids["TET_ID"][0, iv]

                    pix = np.unique(T2)
                    pix = pix[pix != 0]
                    if pix.size != 0:
                        amt_1_iv = []
                        for p in pix:
                            amt_1_iv.append(np.sum(T2 == p))
                        amt_1.append(amt_1_iv)
                        val, ind = max((v, i) for i, v in enumerate(amt_1_iv))
                        TET_ind[iv, 1] = pix[ind]

        for ixx in range(len(cell_arrays)):
            if cell_arrays[ixx] is not None:
                tet_no = np.where(TET_ind[:,1] == ixx)[0]
                descendants_data.append(cell_arrays[ixx][tet_no]) 
                tet_no = np.where(TET_ind[:, 1] == ixx)[0]
                descendants_data[tet_no, 3] = cell_arrays[ixx]

    sio.savemat(os.path.join(sav_path, f'{pos}_final_descendants.mat'), {
        "I3": I3,
        "descendants_data": descendants_data,
        "alive_tets": alive_tets,
        "TET_obj": TET_obj
    })
