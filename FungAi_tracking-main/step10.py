import os
import numpy as np
import h5py
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage import img_as_ubyte
from skimage.io import imshow
from glob import glob

# Initialize variables as needed
pos = 'Pos0_2'
path = '/Users/samarth/Documents/MATLAB/Full_Life_Cycle_tracking/tracks/'  # Path to segment SpoSeg masks
sav_path = '/Users/samarth/Documents/MATLAB/Full_Life_Cycle_tracking/saved_res/py_res/'  # Path to save Tracks

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

# Load ART and TET tracks
art_file_list = sorted(glob(os.path.join(path, '*_ART_Track_DS*')))
art = load_mat(os.path.join(path, art_file_list[0]))  # load art track

tet_file_list = sorted(glob(os.path.join(path, '*_TET_Track_DS*')))
tet = load_mat(os.path.join(path, tet_file_list[0]))  # load tet track

print("Keys in ART:", art.keys())
print("Keys in TET:", tet.keys())

# Extract necessary variables from loaded data
Mask3 = []
with h5py.File(os.path.join(path, art_file_list[0]), 'r') as f:
    for i in range(len(f['Mask3'])):
        art_masks_refs = f['Mask3'][i]
        for ref in art_masks_refs:
            mask = resolve_h5py_reference(ref, f)
            Mask3.append(mask)
            
shock_period = tet['shock_period']
TET_obj = int(tet['TET_obj'][0, 0])
TET_exists = tet['TET_exists']

TETmasks = []
with h5py.File(os.path.join(path, tet_file_list[0]), 'r') as f:
    for i in range(len(f['TETmasks'])):
        tet_masks_refs = f['TETmasks'][i]
        for ref in tet_masks_refs:
            mask = resolve_h5py_reference(ref, f)
            TETmasks.append(mask)

TET_ID = np.zeros((1, TET_obj))

print(shock_period[1,0])

for iv in range(TET_obj):
    its = int(TET_exists[iv, 0]) - 1 #
    
    if its > shock_period[1, 0]:
        TET_ID[0, iv] = -1
    else:
        A = Mask3[its].astype(float).T

        T = resize(TETmasks[its].T, A.shape, order=0, preserve_range=True).astype(float) 
        
        T1 = (T == iv + 1).astype(float)

        Im1 = (T1 > threshold_otsu(T1)).astype(float)

        Im2 = binary_erosion(Im1, np.ones((9, 9))).astype(float)

        Im3 = A * Im2

        pix1 = np.unique(A[Im3 != 0])
        TET_ID[0, iv] = pix1[0] if pix1.size > 0 else -1

name1 = os.path.join(sav_path, f'{pos}_TET_ID_art_track.mat')
sio.savemat(name1, {'TET_ID': TET_ID})
