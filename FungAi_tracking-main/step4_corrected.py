# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:03:27 2024

@author: samar
"""
import h5py
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.morphology import binary_erosion, disk, erosion, square
from skimage.filters import threshold_otsu
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# Define paths and parameters
pos = 'Pos13_1_B'
path = f'/Users\oargell\Desktop/Pos13_1_B/'
sav_path = '/Users\oargell\Desktop/'
#shock_period = [122, 134]

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
    
# def resolve_h5py_reference(data, f):
#     if isinstance(data, h5py.Reference):
#         return f[data][()]
#     return data

def resize_image(image, target_shape):
    zoom_factors = [n / float(o) for n, o in zip(target_shape, image.shape)]
    return zoom(image, zoom_factors, order=0)

# Load ART masks
path_dir = [f for f in sorted(os.listdir(path)) if f.endswith('_ART_masks.tif')]
Art_MT = [imread(os.path.join(path, f)).astype(np.uint16) for f in path_dir]# plt.imshow(Art_MT[134])

# Load tracked SpoSeg masks
tet_track_path = os.path.join(sav_path, f'{pos}_TET_Track_py.mat')

if os.path.exists(tet_track_path):
    
    tet = load_mat(tet_track_path)
    shock_period = tet['shock_period']-1 # why are we subtracting 1 here
    
       
    TETmasks = tet['TETmasks']# plt.imshow(TETmasks[0][162])
       
    
    
    TET_obj=tet['TET_obj']
    TET_obj= TET_obj[0][0]
   
    for iv in range(TET_obj):
        # iv = 0;
        if tet['TET_exists'][1][iv] >= shock_period[0][0]-1:#!  shock period corrected produce a single integer!!!
            tp_end = shock_period[0][1]
        else:
            tp_end = tet['TET_exists'][1][iv] 

        for its in range(tet['TET_exists'][0][iv], tp_end):# minus one added
            # its = 133;
            A1 = Art_MT[its].astype(np.double)
            plt.imshow(A1)
            if shock_period[0][0]-1 <= its <= shock_period[0][1]:#!!shock period corrected produce a single integer!!!!!!!!!!!!!!!!!!!!
                T1 = (TETmasks[0][shock_period[0][0]-1] == iv + 1).astype(np.double)#! TETmasks correctd and shock period corrected produce a single integer[]
                thresh = 0.6
            else:
                T1 = (TETmasks[0][its] == iv + 1).astype(np.double)#!!!!!!!!!!!!!!!!!!  
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
         if tet['TET_exists'][1][iv] > shock_period[0][1] and tet['TET_exists'][0][iv] < shock_period[0][0]:
            s1 = np.sum(TETmasks[0][shock_period[0][1]+1] == iv+1)#!!!!!!!!!!!!!!!!!!!!!!
            for its in range(shock_period[0][1]+1, tet['TET_exists'][1][iv]):#!!!!!!!!!!!!!!!!!!!!!!!!
                # its = 134;
                A1 = Art_MT[its].astype(np.double)
                # plt.imshow(A1)
                T1 = (TETmasks[0][its] == iv + 1).astype(np.double)
                # plt.imshow(T1)
                
                

                s2 = np.sum(TETmasks[0][its] == iv + 1)
                if its == tet['TET_exists'][1][iv]:
                    s3 = np.sum(TETmasks[0][its] == iv + 1)
                else:
                    s3 = np.sum(TETmasks[0][its + 1] == iv + 1)

                if s2 < s1 - 0.1 * s1:
                    if s3 > s2 + 0.1 * s2:
                        T1 = (TETmasks[0][its - 1] == iv + 1).astype(np.double)
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
                    
                    
            
    MAT1=np.transpose(Art_MT,(1,2,0));
    sio.savemat(os.path.join(sav_path, f'{pos}_ART_Masks_py.mat'), {"Art_MT": MAT1, "shock_period": shock_period})
else:
    MAT1=np.transpose(Art_MT,(1,2,0));
    sio.savemat(os.path.join(sav_path, f'{pos}_ART_Mask_py.mat'), {"Art_MT": MAT1, "shock_period": shock_period})
