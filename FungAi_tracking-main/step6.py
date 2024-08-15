import os
import numpy as np
import h5py
import scipy.io as sio
from skimage.transform import resize
from skimage.measure import regionprops, label
from skimage.morphology import thin
from scipy.ndimage import binary_fill_holes
from functions.SR_240222_cal_allob import cal_allob
from functions.SR_240222_cal_celldata import cal_celldata
import matplotlib.pyplot as plt

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


# Modified Cal Allob 
def cal_allob1(ccel, TETC, rang):
    # Initialize the all_obj array with zeros
    all_obj = np.zeros((ccel, len(TETC)))

    for iv in range(ccel):  # Adjusted to 1-based index
        for its in rang:
            if np.sum(TETC[its]) > 0:  # Check if the array is not None and not empty
                all_obj[iv, its] = np.sum(TETC[its] == iv + 1)  # Adjusted for 1-based index logic
            else:
                all_obj[iv, its] = -1

    return all_obj

# Define parameters
pos = 'Pos0_2'
path = '/Users/samarth/Documents/MATLAB/Full_Life_Cycle_tracking/Tracks2/'  # Path to segment SpoSeg masks
sav_path = '/Users/samarth/Documents/MATLAB/Full_Life_Cycle_tracking/saved_res/'  # Path to save Track

# Load MAT Track data
file_list = [f for f in os.listdir(path) if '_MAT_16_18_Track' in f]
file_list = sorted(file_list)
mat = load_mat(os.path.join(path, file_list[0]))

Matmasks = []
with h5py.File(os.path.join(path, file_list[0]), 'r') as f:
    for i in range(len(f['Matmasks'])):
        tet_masks_refs = f['Matmasks'][i]
        for ref in tet_masks_refs:
            mask = resolve_h5py_reference(ref, f)
            Matmasks.append(mask)

# Extract variables from loaded data
no_obj = int(mat['no_obj'][0])
if no_obj != 0:
    shock_period = mat['shock_period']
    MTrack = Matmasks
    cell_data = mat['cell_data']

    # Load ART Track data
    file_list = [f for f in os.listdir(path) if '_ART_Track' in f]
    file_list = sorted(file_list)
    art = load_mat(os.path.join(path, file_list[0]))
    art_masks = []
    with h5py.File(os.path.join(path, file_list[0]), 'r') as f:
        for i in range(len(f['Mask3'])):
            masks_refs = f['Mask3'][i]
            for ref in masks_refs:
                mask = resolve_h5py_reference(ref, f)
                art_masks.append(mask)
    mat_artifacts = []

    # Resize MTrack to match ART masks
    for its in range(len(MTrack)):
        if MTrack[its].size > 2:
            MTrack[its] = resize(MTrack[its], art_masks[its].shape, order=0, preserve_range=True, anti_aliasing=False)

    tp_end = len(art_masks)
    if len(MTrack) != tp_end:
        for its in range(len(MTrack[its]), tp_end):
            MTrack.append(np.zeros_like(MTrack[int(min(cell_data[:, 0])) - 1], dtype=np.uint16))

    # Correcting mating tracks
    cor_data = np.zeros((3, no_obj))
    size_cell = np.zeros((no_obj, len(MTrack)))
    morph_data = np.zeros((no_obj, len(MTrack)))
    outlier_tps = [None] * no_obj
    good_tps = [None] * no_obj
    

    for iv in range(no_obj):
        # iv = 0;
        int_range = range(int(cell_data[0, iv]) - 1, int(cell_data[1, iv]))  # Adjusting for 0-based indexing
        for its in int_range:
            # its = 240;
            M = np.uint16(MTrack[its] == iv + 1).T
            
            
# =============================================================================
#             plt.figure()
#             plt.imshow(np.uint16(M), cmap='gray')
#             plt.title('M')
#             plt.show()
# =============================================================================
            
            
            size_cell[iv, its] = np.sum(M)
            props = regionprops(M)
            morph_data[iv, its] = props[0].eccentricity if props else 0
        cor_data[0, iv] = np.mean(size_cell[iv, int_range])
        cor_data[1, iv] = np.std(size_cell[iv, int_range])
        cor_data[2, iv] = 1 * cor_data[1, iv]
        outlier_tps[iv] = [t for t in int_range if abs(size_cell[iv, t] - cor_data[0, iv]) > cor_data[2, iv]]
        good_tps[iv] = np.setdiff1d(int_range, outlier_tps[iv])

    for iv in range(no_obj):
        # iv = 0
        int_range = range(int(cell_data[0, iv]) - 1, int(cell_data[1, iv]))
        if np.var(morph_data[iv, int_range]) > 0.02:
            mat_artifacts.append(iv)

    for iv in range(no_obj):
        outlier = sorted(outlier_tps[iv])
        good = sorted(good_tps[iv])
        int_range = range(int(cell_data[0, iv]) - 1, int(cell_data[1, iv]))
        while outlier:
            its = min(outlier)
            gtp = max([g for g in good if g < its], default=min([g for g in good if g > its], default=its))
            A = art_masks[its].T
            
# =============================================================================
#             plt.figure()
#             plt.imshow(np.uint16(M), cmap='gray')
#             plt.title('M')
#             plt.show()
# =============================================================================
            
            M1 = (MTrack[gtp] == (iv + 1)).T
            M2 = thin(M1, 30)
            M3 = A * M2
            
            # plt.figure()
            # plt.imshow(np.uint16(M3), cmap='gray')
            # plt.title('M3')
            # plt.show()
            
            indx = np.unique(A[M3 != 0])
            if indx.size > 0:
                X1 = np.zeros_like(MTrack[its]).T
                for itt2 in indx:
                    if np.sum(M3 == itt2) > 5:
                        X1[A == itt2] = 1
                X1 = binary_fill_holes(X1)
                # plt.imshow(X1)
                X2 = label(X1)
                if np.max(X2) <= 1 and abs(np.sum(X1) - cor_data[0, iv]) <= 2 * cor_data[1, iv]:
                    MTrack[its][MTrack[its] == (iv + 1)] = 0
                    (MTrack[its].T)[X1 == 1] = iv + 1
                else:
                    MTrack[its][MTrack[its] == (iv + 1)] = 0
                    MTrack[its][MTrack[gtp] == (iv + 1)] = iv + 1
            outlier = [o for o in outlier if o != its]
            good.append(its)
            good = sorted(good)

    for iv in range(no_obj):
        if cell_data[1, iv] != tp_end:
            count = 0
            for its in range(int(cell_data[1, iv]), tp_end):
                A = art_masks[its]
                M1 = (MTrack[its - 1] == (iv + 1)).T
                M2 = thin(M1, 30)
                M3 = A * M2
                indx = np.unique(A[M3 != 0])
                if indx.size > 0:
                    X1 = np.zeros_like(MTrack[its])
                    for itt2 in indx:
                        if np.sum(M3 == itt2) > 5:
                            X1[A == itt2] = 1
                    if abs(np.sum(X1) - cor_data[0, iv]) > 2 * cor_data[1, iv]:
                        count += 1
                        MTrack[its][MTrack[its - 1] == (iv + 1)] = iv + 1
                    else:
                        MTrack[its][X1 == 1] = iv + 1
                else:
                    count += 1
                    MTrack[its][MTrack[its - 1] == (iv + 1)] = iv + 1
            if count / (tp_end - cell_data[iv, 0]) > 0.8:
                mat_artifacts.append(iv + 1)

    # Remove cell artifacts and rename
    if mat_artifacts:
        all_ccel = list(range(1, no_obj + 1))
        mat_artifacts = sorted(set(mat_artifacts))
        for iv in mat_artifacts:
            for its in range(len(MTrack)):
                MTrack[its][MTrack[its] == iv] = 0
        good_cells = sorted(set(all_ccel) - set(mat_artifacts))
        for iv in range(len(good_cells)):
            for its in range(len(MTrack)):
                MTrack[its][MTrack[its] == good_cells[iv]] = iv + 1
        no_obj = len(good_cells)
        
        

    # Recalculating MAT Data
    
    all_obj_new = cal_allob1(no_obj, MTrack, list(range(len(MTrack))))
    
    x_scale = 777
    y_scale = 2
    aspect_ratio = (777 / all_obj_new.shape[1]) / (2 / all_obj_new.shape[0])

    # Display the image with the adjusted scales
    plt.imshow(all_obj_new, extent=[0, x_scale, 0, y_scale], aspect='auto')
    plt.colorbar()
    
    # Optional: Set the ticks to reflect the scales accurately
    plt.xticks(np.linspace(0, x_scale, num=5))
    plt.yticks(np.linspace(0, y_scale, num=5))
    
    plt.show()
    
    cell_data_new = cal_celldata(all_obj_new, no_obj)

    cell_data = cell_data_new
    all_obj = all_obj_new
    Matmasks = MTrack

    sio.savemat(f'{sav_path}{pos}_MAT_16_18_Track1_py.mat', {
        "Matmasks_py": Matmasks,
        "all_obj_py": all_obj,
        "cell_data_py": cell_data,
        "no_obj_py": no_obj,
        "shock_period_py": shock_period,
        "mat_artifacts_py": mat_artifacts
    }, do_compression=True)
