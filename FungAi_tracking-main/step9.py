#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 00:21:37 2024

@author: samarth
"""

import os
import h5py
import numpy as np
import scipy.io as sio
from skimage.morphology import thin
from skimage.filters import threshold_otsu
from skimage.measure import label
import matplotlib.pyplot as plt

# Initialize variables
pos = 'Pos0_2'
path = '/Users/samarth/Documents/MirandaLabs/tracking_algo/FungAi_tracking/Tracking_toydata_Tracks' 
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

# Load ART and MAT tracks
art_track_path = os.path.join(path)
a_file_list = [f for f in os.listdir(art_track_path) if 'ART_Track1_DS' in f]
art = load_mat(os.path.join(path, a_file_list[0]))

mat_track_path = os.path.join(path)
m_file_list = file_list = [f for f in os.listdir(mat_track_path) if 'MAT_16_18_Track1_DS' in f]
mat = load_mat(os.path.join(path, m_file_list[0]))

# Obtain the gametes indexes that give rise to the mating cell (zygote)
Mask3 = []
with h5py.File(os.path.join(path, a_file_list[0]), 'r') as f:
    for i in range(len(f['Mask3'])):
        masks_refs = f['Mask3'][i]
        for ref in masks_refs:
            mask = resolve_h5py_reference(ref, f)
            Mask3.append(mask)

    # Reading Mat Tracks from mat

MTrack = []
with h5py.File(os.path.join(path, m_file_list[0]), 'r') as f:
    for i in range(len(f['Matmasks'])):
        masks_refs = f['Matmasks'][i]
        for ref in masks_refs:
            mask = resolve_h5py_reference(ref, f)
            MTrack.append(mask)
            
gamete = np.zeros((3, int(mat['no_obj'][0, 0])))
cell_data = mat['cell_data']
shock_period = mat['shock_period']



for iv in range(int(mat['no_obj'][0, 0])):
    tp_mat_start = int(mat['cell_data'][0, iv])  # First appearance of mating event "iv"
    M1 = MTrack[tp_mat_start-1]  # Mat Track at tp_mat_start
    for its in range(tp_mat_start - 2, int(mat['shock_period'][1, 0]), -1):  # Loop through time from 1 tp before mating to one time point after shock
        A1 = Mask3[its].astype(float)
        
        
        # plt.figure()
        # plt.imshow(A1, cmap='gray')
        # plt.title('A1')
        # plt.show()
    
        
        
        M2 = (M1 == iv + 1).astype(float)
        
        Im1 = (M2 > threshold_otsu(M2)).astype(float)
        Im2 = thin(Im1, 10).astype(float)
        Im3 = A1 * Im2
        
        
        # plt.figure()
        # plt.imshow(Im2, cmap='gray')
        # plt.title('Im2')
        # plt.show()
        
        
        
        # plt.figure()
        # plt.imshow(Im3, cmap='gray')
        # plt.title('Im3')
        # plt.show()
        
        
        pix2 = np.unique(A1[Im3 != 0])
        
        if pix2.size == 2:  # captures mature mating
            r = np.sum(Im3 == pix2[0]) / np.sum(Im3 == pix2[1])
            if (2/8) <= r <= (8/2):  # 4/6 to 9/6
                gamete[0, iv] = pix2[0]
                gamete[1, iv] = pix2[1]
                gamete[2, iv] = its + 1

for iv in range(int(mat['no_obj'][0, 0])):
    if gamete[0, iv] == 0 and gamete[1, iv] == 0:
        tp_mat_start = int(mat['cell_data'][iv, 0])  # First appearance of mating event "iv"
        M1 = MTrack[tp_mat_start-1]  # Mat Track at tp_mat_start
        for its in range(tp_mat_start - 2, int(mat['shock_period'][1, 0]), -1):  # Loop through time from 1 tp before mating to one time point after shock
            A1 = Mask3[its].astype(float)
            M2 = (M1 == iv + 1).astype(float)
            
            Im1 = (M2 > threshold_otsu(M2)).astype(float)
            Im2 = thin(Im1, 10).astype(float)
            Im3 = A1 * Im2
            
            # plt.figure()
            # plt.imshow(Im3, cmap='gray')
            # plt.title('Im3_2')
            # plt.show()
            
            pix2 = np.unique(A1[Im3 != 0])
            
            if pix2.size == 1:  # captures ascus mating
                gamete[0, iv] = pix2[0]
                gamete[2, iv] = its + 1

sio.savemat(os.path.join(sav_path, f'{pos}_gametes.mat'), {"gamete": gamete}, do_compression=True)