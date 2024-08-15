import os
import math
import numpy as np
import re
import scipy.io as sio
from skimage.io import imread
from skimage.morphology import skeletonize
from scipy.stats import mode
from functions.SR_240222_cal_allob import cal_allob
from functions.OAM_231216_bina import binar
from functions.SR_240222_cal_celldata import cal_celldata
import matplotlib.pyplot as plt

# Define thresholds and parameters
thresh_percent = 0.015
thresh_remove_last_mask = 10
thresh_next_cell = 400
thresh = 80
shock_period = [122, 134]

pos = 'Pos0_2'
path = f'/Users/samarth/Documents/MATLAB/Full_Life_Cycle_tracking/Pos13_1_B/'
sav_path = '/Users/samarth/Documents/MATLAB/Full_Life_Cycle_tracking/saved_res/py_res'

# Load image file names
file_names = [f for f in os.listdir(path) if f.endswith('_Ph3_000_TET_masks.tif')]
file_numbers = np.zeros(len(file_names), dtype=int)

# Extract file numbers
for i, name in enumerate(file_names):
    base_name = os.path.splitext(name)[0]
    print(base_name)
    pattern = r"img_(\d+)_Ph3"
    
    # Use re.search to find the pattern in the input string
    match = re.search(pattern, base_name)
    
    if match:
        # Extract the matched group (the number) and convert it to an integer
        number = int(match.group(1))
    else:
        raise ValueError("The pattern 'img_<number>_Ph3' was not found in the input string.")
    file_numbers[i] = int(number)

# Sort file names by the extracted numbers
sorted_indices = np.argsort(file_numbers)
sorted_numbers = file_numbers[sorted_indices]

tet_masks_path = [os.path.join(path, file_names[i]) for i in sorted_indices]

# Load images
tet_masks = [None] * (sorted_numbers[-1] + 1)
for i, img_path in enumerate(tet_masks_path):
    tet_masks[sorted_numbers[i]] = imread(img_path)

# Initialize tet_masks for valid SpoSeg masks
for i in range(min(sorted_numbers), len(tet_masks)):
    if tet_masks[i] is None:
        tet_masks[i] = np.zeros_like(tet_masks[min(sorted_numbers)], dtype=np.uint16)

# Remove shock-induced timepoints
for start, end in [shock_period]:
    for i in range(start-1, end):
        tet_masks[i] = None

start = -1
for its in range(len(tet_masks)):
    if tet_masks[its] is not None and np.sum(tet_masks[its]) > 0:
        start = its
        break

# Tracking all detections
if start != -1:
    rang = range(start, len(tet_masks))
    I2 = tet_masks[start]
    A = np.zeros_like(tet_masks[start])
else:
    rang = range(len(tet_masks))
    I2 = tet_masks[0]
    A = np.zeros_like(tet_masks[0])

IS6 = np.zeros_like(I2)
TETC = [None] * 2
TETC[0] = [None] * len(tet_masks)
TETC[1] = [None] * len(tet_masks)
xx = start
rang2 = rang
ccel = 1

while xx != -1:
    k = 0
    for im_no in rang2:
        # im_no = 72
        I2 = tet_masks[im_no] if ccel == 1 else TETC[1][im_no]
        if I2 is None or I2.size == 0:
            continue
        if im_no == min(rang2):
            ind1 = np.unique(I2)[1:]  # Exclude background
            I3 = (I2 == ind1[0])
            I3A = I3.copy()
        else:
            I3A = IS6.copy()

        I3A = skeletonize(binar(I3A))
        I2A = I2.copy()
        I3B = I3A.astype(np.uint16) * I2A.astype(np.uint16)
        ind = mode(I3B[I3B != 0])[0]

        if (ind == 0 or math.isnan(ind)) and ccel == 1:
            k += 1
            if k > thresh_next_cell:
                for im_no_1 in range(im_no, rang[-1] + 1):
                    if tet_masks[im_no_1] is not None:
                        TETC[0][im_no_1] = np.zeros_like(tet_masks[start])
                    TETC[1][im_no_1] = tet_masks[im_no_1]
                break
            else:
                TETC[0][im_no] = I3B.copy()
                TETC[1][im_no] = I2A.copy()
                continue
        elif (ind == 0 or math.isnan(ind)) and ccel != 1:
            k += 1
            if k > thresh_next_cell:
                break
            else:
                continue

        k = 0
        pix = np.where(I2A == ind)
        pix0 = np.where(I2A != ind)
        
        # pix = np.flatnonzero(I2A == ind)
        # pix0 = np.flatnonzero(I2A != ind)
        
        I2A[pix] = ccel
        I2A[pix0] = 0
        
        IS6 = I2A.copy()
        I22 = np.zeros_like(I2)
        
        pix1 = np.where(IS6 == ccel)
        
        I2[pix1] = 0
        pix2 = np.unique(I2)
        pix2 = pix2[1:]  # Exclude background

        if ccel == 1:
            for ity, p2 in enumerate(pix2):
                pix4 = np.where(I2 == p2)
                I22[pix4] = ity + 1
            TETC[0][im_no] = IS6.copy()
        else:
            if len(pix2) > 0:
                for ity, p2 in enumerate(pix2):
                    pix4 = np.where(I2 == p2)
                    I22[pix4] = ity + 1
            else:
                I22 = I2.copy()
            IS61 = TETC[0][im_no]
            IS61[pix] = ccel
            TETC[0][im_no] = IS61.astype(np.uint16)

        TETC[1][im_no] = I22.copy()

    xx = -1
    for i in rang:
        if TETC[1][i] is not None and np.sum(TETC[1][i]) > 0:
            xx = i
            break
    ccel += 1
    rang2 = range(xx, len(tet_masks))
    print(xx + 1)


ccel -= 1  # number of cells tracked

# Removing the shock-induced points from rang
rang3 = list(rang)
for start, end in [shock_period]:
    for i in range(start-1, end):
        if i in rang3:
            rang3.remove(i)

# Removing artifacts - cells that appear once and cells that disappear thresh % of the time or more
def cal_allob1(ccel, TETC, rang):
    # Initialize the all_obj array with zeros
    all_obj = np.zeros((ccel, len(TETC[1])))

    for iv in range(ccel):  # Adjusted to 1-based index
        for its in rang:
            if TETC[0][its] is not None: #and np.sum(TETC[0][its]) > 0:  # Check if the array is not None and not empty
                all_obj[iv, its] = np.sum(TETC[0][its] == iv + 1)  # Adjusted for 1-based index logic
            else:
                all_obj[iv, its] = -1

    return all_obj

all_obj = cal_allob1(ccel, TETC, rang)
x_scale = 200
y_scale = 4
plt.imshow(all_obj, extent=[0, x_scale, 0, y_scale], aspect='auto')

# sio.savemat(os.path.join(sav_path, "art_py.mat"), {
#                 'all_ob_py': all_obj
#             })
cell_data = cal_celldata(all_obj, ccel) ## double check values

k = 1
cell_artifacts = []
for iv in range(ccel):
    if cell_data[iv, 2] < thresh_percent * len(rang3) or cell_data[iv, 4] > thresh:
        cell_artifacts.append(iv + 1)
        k += 1

all_ccel = list(range(1, ccel + 1))

if len(cell_artifacts) > 0:
    cell_artifacts = list(set(cell_artifacts))
    for iv in cell_artifacts:
        for its in rang3:
            pix = np.where(TETC[0][its] == iv)
            TETC[0][its][pix] = 0

# Retaining and relabeling the new cells
good_cells = sorted(set(all_ccel) - set(cell_artifacts))

for iv in range(len(good_cells)):
    for its in rang3:
        pix = np.where(TETC[0][its] == good_cells[iv])
        TETC[0][its][pix] = iv + 1

# Correcting the SpoSeg track masks or filling the empty spaces between the first and last appearance
# Removing artifacts
all_obj1 = cal_allob1(len(good_cells), TETC, rang)
# plt.imshow(all_obj1, extent=[0, x_scale, 0, y_scale], aspect='auto')
cell_data1 = cal_celldata(all_obj1, len(good_cells))

for iv in range(len(good_cells)):
    for its in range(int(cell_data1[iv, 0] + 1), int(cell_data1[iv, 1])):
        if all_obj1[iv, its] == 0:
            prev = np.where(all_obj1[iv, :its] > 0)[0][-1]
            all_obj1[iv, its] = all_obj1[iv, prev]
            pix = np.where(TETC[0][prev] == iv + 1)
            TETC[0][its][pix] = iv + 1

# Cell array that contains the fully tracked TetSeg masks
TETmasks = [TETC[0][i] for i in range(len(TETC[0]))]

# Calculate the size of tetrads
def cal_allob2(ccel, TETC, rang):
    # Initialize the all_obj array with zeros
    all_obj = np.zeros((ccel, len(TETC)))

    for iv in range(ccel):  # Adjusted to 1-based index
        for its in rang:
            if TETC[its] is not None: #and np.sum(TETC[its]) > 0:  # Check if the array is not None and not empty
                all_obj[iv, its] = np.sum(TETC[its] == iv + 1)  # Adjusted for 1-based index logic
            else:
                all_obj[iv, its] = -1

    return all_obj

TET_obj = len(good_cells)
all_obj_final = cal_allob2(TET_obj, TETmasks, list(range(len(TETmasks))))
plt.imshow(all_obj1, extent=[0, x_scale, 0, y_scale], aspect='auto')
TET_Size = all_obj_final.copy()

# Calculate first detection and last detection of tetrads
TET_exists = np.zeros((2, TET_obj), dtype=int)
for iv in range(TET_obj):
    TET_exists[0, iv] = np.where(TET_Size[iv, :] > 0)[0][0]  # 1st occurrence
    TET_exists[1, iv] = np.where(TET_Size[iv, :] > 0)[0][-1]  # last occurrence

tet_masks_exists_tp = rang3

def replace_none_with_empty_array(data):
    if isinstance(data, list):
        return [replace_none_with_empty_array(item) for item in data]
    elif data is None:
        return np.array([])
    else:
        return data
    
TETmasks = replace_none_with_empty_array(TETmasks)

# Save results
sio.savemat(sav_path + '_TET_Track_py.mat', {
    'start_py': start,
    'TET_Size_py': TET_Size,
    'TET_obj_py': TET_obj,
    'TET_exists_py': TET_exists,
    'TETmasks_py': TETmasks,
    'shock_period_py': shock_period,
    'thresh_py': thresh,
    'thresh_next_cell_py': thresh_next_cell,
    'thresh_percent_py': thresh_percent,
    'tet_masks_exists_tp_py': tet_masks_exists_tp
}, do_compression=True)
