#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 15:27:04 2024

@author: samarth
"""

import os
import numpy as np
import scipy.io as sio
import h5py

# Initialize variables
pos = 'Pos0_2'
path = '/Users/samarth/Documents/MATLAB/Full_Life_Cycle_tracking/Tracks2/'  # Path to segment SpoSeg masks
sav_path = '/Users/samarth/Documents/MATLAB/Full_Life_Cycle_tracking/saved_res/'  # Path to save Track

# Function to load .mat files

def load_mat_file(filename):
    try:
        return sio.loadmat(filename)
    except NotImplementedError:
        # Load using h5py for MATLAB v7.3 files
        data = {}
        with h5py.File(filename, 'r') as f:
            for key in f.keys():
                data[key] = np.array(f[key])
        return data

# Load TET IDs
file_list = [f for f in os.listdir(path) if f.endswith('_TET_ID_art_track.mat')]
TET_ID_data = load_mat_file(os.path.join(path, file_list[0]))

# Load art track
file_list = [f for f in os.listdir(path) if f.endswith('_ART_Track1_DS.mat')]
art_track_data = load_mat_file(os.path.join(path, file_list[0]))

# Load tet track
file_list = [f for f in os.listdir(path) if f.endswith('_TET_Track_DS.mat')]
tet_data = load_mat_file(os.path.join(path, file_list[0]))

# Load descendants information
desc_path = path
file_list = [f for f in os.listdir(desc_path) if f.endswith('_final_descendants.mat')]
desc_data = load_mat_file(os.path.join(path, file_list[0]))

alive_tets = desc_data['alive_tets'].flatten()

# Choose time frame during rich medium where cells will likely germinate
int_range = range(int(tet_data['shock_period'][1, 0] + 1), int(tet_data['shock_period'][1, 0] + 36))

# Initialize germination_point array
germination_point = np.zeros(TET_ID_data['TET_ID'].shape[1])
all_ob = art_track_data["all_ob"]
TET_ID_data = TET_ID_data["TET_ID"]
shock_period = tet_data["shock_period"]
# Loop through the tet cells that are alive
for iv in alive_tets:
    iv = int(iv) - 1
    if TET_ID_data[0, iv] != -1:
        y = all_ob[:,TET_ID_data[0, 0]]
        A = all_ob[int_range, TET_ID_data[0, iv]]
        v, i = max((idx, val) for (val, idx) in enumerate(A))
        k = int(i + shock_period[0, 0])
        B = y[k:k+6]
        C = np.diff(B)
        iP, iL = max((val, idx) for (idx, val) in enumerate(-C))
        k1 = k + iL - 1
        germination_point[iv] = k1  # If value = 0, then tet is dead

# Save the results
np.savez(os.path.join(sav_path, f'{pos}_germination_point.npz'), germination_point=germination_point, alive_tets=alive_tets)
