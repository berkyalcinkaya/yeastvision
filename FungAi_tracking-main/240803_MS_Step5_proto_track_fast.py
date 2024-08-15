# -*- coding: utf-8 -*-
"""
Created on Sat Aug 03 09:12:38 2024

@author: Monish Shah
"""
import numpy as np
import os, sys
import statistics
import time
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import thin
import scipy.io as sio
from scipy.stats import mode

begin = time.time()

try:
    import cupy as cp
except ModuleNotFoundError as e:
    print(f"{e} not installed, proceeding with normal Numpy CPU processing")
finally:
    pass

try:
    from functions.OAM_230919_remove_artif import remove_artif
    from functions.OAM_231216_bina import binar
except ModuleNotFoundError as e:
    print(f"Module not found: {e}")
    print("")
    print("Without quotation marks,")
    func_path = input("Enter absolute path to functions directory: ")
    sys.path.append(func_path)
    from OAM_230919_remove_artif import remove_artif
    from OAM_231216_bina import binar
finally:
    pass


#Global Variables
art_mask_path = "/Users/mshah/Desktop/New_Step4_Results/Py/_ART_Masks.mat" #absolute path to ART_Masks file
input_images = "/Users/mshah/Desktop/Pos13_1/"
sav_path = "/Users/mshah/Desktop/New_Step5_Results/Py/"
pos = ''

#Helper Functions
def OAM_23121_tp3(M, cel, no_obj1, A):
    tp3 = np.array(M)  # Ensure M is a numpy array
    tp3[tp3 == cel] = no_obj1 + A
    return tp3
    
def export(variable, name):
    sio.savemat(sav_path + f'{name}_py.mat', { f'{name}_py': variable})
    

## Obtain and correctly load Masks3 as tensor
mat = sio.loadmat(art_mask_path)
shock_period = mat['shock_period']
Masks3 = mat['Art_MT'] 
Masks3 = np.transpose(Masks3, (1, 2, 0))

im_no1 = 0
im_no = 50 #Masks3.shape[2]
mm = range(im_no) # time points to track


"""
Load the first mask that begins the indexing for all the cells; IS1 is updated to most recently processed tracked mask at the end of it0
"""
IS1 = np.copy(Masks3[:,:,im_no1]).astype('uint16') # start tracking at first time point # plt.imshow(IS1)
IS1 = remove_artif(IS1) # remove artifacts and start tracking at first time point # plt.imshow(IS1)
masks = np.zeros((IS1.shape[0], IS1.shape[1], im_no)) # contains the re-labeled masks according to labels in the last tp mask
masks[:,:,im_no1] = IS1.copy() # first time point defines indexing; IS1 is first segmentation output

"""
Allocate a mask for cells where there's a gap in the segmentation; IblankG is updated within the loops it1 and itG
"""
IblankG = np.zeros(IS1.shape, dtype="uint16")
for it0 in mm: # notice IS1 will be updated in the loop
    print(f'it0={it0}')
    # Load the future cellpose mask, IS2: IS2 is the current image being re-indexed for tracking
    IS2 = np.copy(Masks3[:,:,it0]).astype('uint16') # plt.imshow(IS2)
    IS2 = remove_artif(IS2, disk_size=5) # set disk_size as needed # 5 is ideal to match MATLAB's disk_size=6
    
    IS2C = np.copy(IS2) # plt.imshow(IS2C) # <--- a copy of IS2, gets updated in it1
    IS1B = binar(IS1)
    
    IS3 = IS1B.astype('uint16') * IS2 # past superimposed to future; updated in it1
    tr_cells = np.unique(IS1[IS1 != 0]) # the tracked cells present in the present mask, IS1
    
    gap_cells = np.unique(IblankG[IblankG != 0]) # the tracked cells that had a gap in their segmentation; were not detected in IS1
    cells_tr = np.concatenate((tr_cells, gap_cells)) # all the cells that have been tracked up to this tp for this position

    
    # Allocate space for the re-indexed IS2 according to tracking
    Iblank0 = np.zeros_like(IS1)
    
    # Go to the previously tracked cells and find corresponding index in current tp being processed, IS2 -> Iblank0: mask of previously tracked cells with new position in IS2
    
    if cells_tr.sum() != 0: # this is required in case the mask goes blank because cells mate immediately during germination
        for it1 in np.sort(cells_tr): # cells are processed in order according to birth/appearance
            IS5 = (IS1 == it1).copy() # go to past mask, IS1, to look for the cell
            IS6A = np.uint16(thin(IS5, max_num_iter=1)) * IS3

            if IS5.sum() == 0: # if the cell was missing in the past mask; look at the gaps in segmentation, otherwise, continue to look at the past mask
                IS5 = (IblankG == it1).copy()
                IS6A = np.uint16(thin(IS5, max_num_iter=1)) * IS2C
                IblankG[IblankG == it1] = 0 # remove the cell from the segmentation gap mask - it'll be updated in the past mask for next round of processing

            # Find the tracked cell's corresponding index in IS2, update IS3 and IS2C to avoid overwriting cells 
            if IS6A.sum() != 0:
                IS2ind = 0 if not IS6A[IS6A != 0].any() else mode(IS6A[IS6A != 0])[0]
                Iblank0[IS2 == IS2ind] = it1
                IS3[IS3 == IS2ind] = 0
                IS2C[IS2 == IS2ind] = 0

        # Define cells with segmentation gap, update IblankG, the segmentation gap mask
        seg_gap = np.setdiff1d(tr_cells, np.unique(Iblank0)) # cells in the past mask (IS1), that were not found in IS2 

        if seg_gap.size > 0:
            for itG in seg_gap:
                IblankG[IS1 == itG] = itG

        # Define cells that were not relabelled in IS2; these are the buds and new cells entering the frame
        Iblank0B = Iblank0.copy()
        Iblank0B[Iblank0 != 0] = 1
        ISB = IS2 * np.uint16(1 - Iblank0B)
        
        # Add new cells to the mask with a new index Iblank0->Iblank, Iblank0 with new cells added
        newcells = np.unique(ISB[ISB != 0])
        Iblank = Iblank0.copy()
        A = 1

        if newcells.size > 0:
            for it2 in newcells:
                Iblank[IS2 == it2] = np.max(cells_tr) + A # create new index that hasn't been present in tracking
                A += 1

        masks[:, :, it0] = np.uint16(Iblank).copy() #<---convert tracked mask to uint16 and store
        IS1 = masks[:, :, it0].copy() # IS1, past mask, is updated for next iteration of it0

    else:
        masks[:, :, it0] = IS2.copy()
        IS1 = IS2.copy()


"""
Tracks as tensor
"""
im_no = masks.shape[2] # last time point to segment
ccell2 = np.unique(masks[masks != 0]) 

Mask2 = np.zeros_like(masks)

try:
    
    import cupy as cp
    mak = cp.asarray(masks)
    # re-assign ID
    for itt3 in range(ccell2.shape[0]): # cells
        pix3 = cp.where(mak == ccell2[itt3])
        Mask2[pix3.get()] = itt3 + 1
        print(itt3)
        
except ModuleNotFoundError as e:
    
    print(f"{e} not installed, proceeding with normal Numpy CPU processing")
    #re-assign ID
    for itt3 in range(ccell2.shape[0]):
        pix3 = np.where(masks == ccell2[itt3])
        Mask2[pix3] = itt3 + 1
        print(itt3)
        
finally:
    
    pass


"""
Get cell presence
"""
Mask3 = Mask2.copy()
numbM = im_no
obj = np.uint16((np.unique(Mask3))) 
no_obj1 = int(np.max(obj))
A = 1

tic = time.time()

tp_im = np.zeros((no_obj1, numbM), dtype=np.float64)

try:
    
    import cupy as cp
    for cel in range(1, np.max(obj) + 1):
        Ma = (Mask3 == cel)
        mak1 = cp.asarray(Ma)
        for ih in range(numbM):
            if cp.sum(mak1[:, :, ih]) != 0:
                tp_im[cel - 1, ih] = 1
        print(cel)
    
except ModuleNotFoundError as e:
    
    print(f"{e} not installed, proceeding with normal Numpy CPU processing")
    for cel in range(1, no_obj1 + 1):
        Ma = (Mask3 == cel)
        for ih in range(numbM):
            if np.sum(Ma[:, :, ih]) != 0:
                tp_im[cel - 1, ih] = 1
    
finally:
    
    pass


print(f"Elapsed time is {time.time() - tic} seconds.")

"""
Split interrupted time series
"""
tic = time.time()

for cel in range(1, np.max(obj) + 1):
    print(cel)
    tp_im2 = np.diff(tp_im[cel - 1, :])
    
    tp1 = np.where(tp_im2 == 1)[0]
    tp2 = np.where(tp_im2 == -1)[0]

    maxp = np.sum(Mask3[:, :, numbM - 1] == cel)
    
    if len(tp1) == 1 and len(tp2) == 1 and maxp != 0: # has one interruption
        for itx in range(tp1[0], numbM):
            tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
            Mask3[:, :, itx] = tp3
        no_obj1 += A

    elif len(tp1) == 1 and len(tp2) == 1 and maxp == 0: # has one interruption
        pass

    elif len(tp1) == len(tp2) + 1 and maxp != 0:
        tp2 = np.append(tp2, numbM - 1)
        for itb in range(1, len(tp1)): # starts at 1 because the first cell index remains unchanged
            for itx in range(tp1[itb] + 1, tp2[itb] + 1):
                tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                Mask3[:, :, itx] = tp3
            no_obj1 += A

    elif not tp2.size or not tp1.size: # its a normal cell, its born and stays until the end
        pass

    elif len(tp1) == len(tp2): # one interruption
        if tp1[0] > tp2[0]:
            tp2 = np.append(tp2, numbM - 1)
            for itb in range(1, len(tp1)): # starts at 1 because the first cell index remains unchanged
                for itx in range(tp1[itb] + 1, tp2[itb] + 1):
                    tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                    Mask3[:, :, itx] = tp3
                no_obj1 += A
        elif tp1[0] < tp2[0]:
            for itb in range(1, len(tp1)):
                for itx in range(tp1[itb] + 1, tp2[itb] + 1):
                    tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                    Mask3[:, :, itx] = tp3
                no_obj1 += A
        elif len(tp2) > 1: # if it has multiple
            for itb in range(1, len(tp1)): # starts at 2 because the first cell index remains unchanged
                for itx in range(tp1[itb] + 1, tp2[itb] + 1):
                    tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                    Mask3[:, :, itx] = tp3
                no_obj1 += A

print(f"Elapsed time is {time.time() - tic} seconds.")

"""
Calculate size
"""
no_obj2 = np.max(np.unique(Mask3)).astype('uint16')
all_ob = np.zeros((no_obj2, im_no))

tic = time.time()


try:
    
    import cupy as cp
    for ccell in range(1, no_obj2 + 1):
        print(ccell)
        Maa = (Mask3 == ccell)
        mak2 = cp.asarray(Maa)
        for io in range(im_no):
            pix = cp.sum(mak2[:, :, io])
            all_ob[ccell - 1, io] = pix
    
except ModuleNotFoundError as e:
    
    print(f"{e} not installed, proceeding with normal Numpy CPU processing")
    for ccell in range(1, no_obj2 + 1):
        print(ccell)
        Maa = (Mask3 == ccell)
        for io in range(im_no):
            pix = np.sum(Maa[:, :, io])
            all_ob[ccell - 1, io] = pix
            
finally:
    
    pass

print(f"Elapsed time is {time.time() - tic} seconds.")

# Cell existing birth and disappear times
cell_exists = np.zeros((2, all_ob.shape[0]))

for itt2 in range(all_ob.shape[0]):
    cell_exists[0, itt2] = np.argmax(all_ob[itt2, :] != 0)
    cell_exists[1, itt2] = len(all_ob[itt2, :]) - np.argmax(all_ob[itt2, ::-1] != 0) - 1
    
no_obj = len(cell_exists[0])

Mask3 = [Mask3[:, :, its] for its in range(Mask3.shape[2])] # Creates a list of 2D slices from 3D tensor Mask3
Mask3 = np.array(Mask3).transpose((1,2,0))

# Save the results
savemat(f"{sav_path}_ART_Track.mat", {
    'all_ob': all_ob,
    'Mask3': Mask3,
    'no_obj': no_obj2,
    'im_no': im_no,
    'ccell2': ccell2,
    'cell_exists': cell_exists,
    'shock_period': shock_period,
})

print(f"Total run time is {time.time() - begin} seconds.")

    







