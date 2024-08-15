import os
import numpy as np
import h5py
import scipy.io as sio
from skimage.morphology import dilation, disk, erosion, binary_dilation, binary_erosion, remove_small_objects, thin
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label
from scipy.ndimage import distance_transform_edt as bwdist
from scipy.ndimage import watershed_ift as watershed
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from skimage.segmentation import relabel_sequential
# Initialize variables
pos = 'Pos0_2'
path = '/Users/samarth/Documents/MATLAB/Full_Life_Cycle_tracking/tracks/'  # Path to segment SpoSeg masks
sav_path = '/Users/samarth/Documents/MirandaLabs/tracking_algo/FungAi_tracking/Tracking_toydata_Tracks'  # Path to save Tracks

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

# Load TET Tracks
tet_track_path = path
file_list = [f for f in os.listdir(tet_track_path) if '_TET_Track_DS' in f]
tet = load_mat(os.path.join(path, file_list[0]))

file_list = [f for f in os.listdir(tet_track_path) if '_TET_ID' in f]
tet_ids = load_mat(os.path.join(path, file_list[0]))  # Load TET IDs

# Load ART Tracks
art_track_path = path
file_list = [f for f in os.listdir(art_track_path) if '_ART_Track_DS' in f]
art = load_mat(os.path.join(path, file_list[0]))  # Load ART track

masks = tet['TETmasks']
ART = art['Mask3']
shock_period = tet['shock_period']

# Handle h5py references
with h5py.File(os.path.join(path, file_list[0]), 'r') as f:
    masks = [resolve_h5py_reference(mask, f) for mask in masks]
    ART = [resolve_h5py_reference(art, f) for art in ART]
    shock_period = resolve_h5py_reference(shock_period, f)

for i in range(len(masks)):
    if masks[i].size > 0:
        masks[i] = resize(masks[i].astype(np.float32), ART[i].shape, order=0, preserve_range=True).astype(np.uint16)

start = int(shock_period[1, 0]) + 1
tp_end = len(ART)
int_range = range(start, tp_end)



# =============================================================================
# # Initialize variables
# TET_obj = tet_ids['TET_ID'].shape[1]
# size_var_tet = np.zeros(TET_obj)
# dead_tets = np.zeros(TET_obj)
# all_ob = art["all_ob"]
# 
# for iv in range(TET_obj):
#     if tet_ids['TET_ID'][0, iv] != -1:
#         A = all_ob.T
#         A = A[tet_ids['TET_ID'][0, iv], start:].T  # Size is calculated only from shock end +1 to the end
#         result = seasonal_decompose(A, model='additive', period=5)  # Adjust period as needed
#         ST = result.seasonal
#         size_var_tet[iv] = np.var(ST)
#         if np.var(ST) < 0.9:  # Threshold
#             dead_tets[iv] = 1
#         else:
#             dead_tets[iv] = 0
#             
#         # Plotting the seasonal component
#         plt.figure()
#         plt.plot(ST)
#         plt.title(f'Spyder Graph {tet_ids["TET_ID"][0, iv]}')
#         plt.xlabel('Time')
#         plt.ylabel('Seasonal Component')
#         plt.show()
#     else:
#         size_var_tet[iv] = -10**5  # If they have a TET_ID of -1
#         dead_tets[iv] = -10**5  # If they have a TET_ID of -1
# =============================================================================
        

# Initialize variables
TET_obj = tet_ids['TET_ID'].shape[1]
size_var_tet = np.zeros(TET_obj)
dead_tets = np.zeros(TET_obj)
all_ob = art["all_ob"]

for iv in range(TET_obj):
    if tet_ids['TET_ID'][0, iv] != -1:
        A = all_ob.T
        A = A[tet_ids['TET_ID'][0, iv]-1, start-1:].T  # Size is calculated only from shock end +1 to the end
        
        # Plot original data for verification
        # plt.figure()
        # plt.plot(A)
        # plt.title(f'Original Data for TET_ID {tet_ids["TET_ID"][0, iv]}')
        # plt.xlabel('Time')
        # plt.ylabel('Value')
        # plt.show()
        
        # Adjust period as needed
        period = 40  # Change this to match MATLAB behavior
        result = seasonal_decompose(A, model='additive', period=period)
        
        LT = result.trend
        ST = result.seasonal
        R = result.resid
        
        size_var_tet[iv] = np.var(ST)
        if np.var(ST) < 0.9:  # Threshold
            dead_tets[iv] = 1
        else:
            dead_tets[iv] = 0
        
        # Plotting the seasonal component
        # plt.figure()
        # plt.plot(ST)
        # plt.title(f'Seasonal Component for TET_ID {tet_ids["TET_ID"][0, iv]}')
        # plt.xlabel('Time')
        # plt.ylabel('Seasonal Component')
        # plt.show()

        # # Plotting the entire decomposition for comparison
        # plt.figure(figsize=(12, 8))
        
        # plt.subplot(4, 1, 1)
        # plt.plot(A)
        # plt.title('Original Data')
        
        # plt.subplot(4, 1, 2)
        # plt.plot(LT)
        # plt.title('Trend Component')
        
        # plt.subplot(4, 1, 3)
        # plt.plot(ST)
        # plt.title('Seasonal Component')
        
        # plt.subplot(4, 1, 4)
        # plt.plot(R)
        # plt.title('Residual Component')
        
        # plt.tight_layout()
        # plt.show()

    else:
        size_var_tet[iv] = -10**5  # If they have a TET_ID of -1
        dead_tets[iv] = -10**5  # If they have a TET_ID of -1


# Finding the first time when TETs germinate and new cells begin to show up

# Initialize the beginning time variable
begin = 0

# Finding the first time when tets germinate and new cells begin to show up
for its in int_range:
    if its == max(int_range):
        begin = 0
        break

    A1 = ART[its]
    A2 = ART[its + 1]

    # To visualize the images, you can use matplotlib (optional)
    
    # plt.figure()
    # plt.imshow(A2, cmap='gray')
    # plt.title('A2')
    # plt.show()

    A3 = (A1.astype(bool)).astype(np.uint16) * A2.astype(np.uint16)
    indx_ori = np.unique(A1[A1 != 0])  # previous mask
    indx_new = np.unique(A2[A2 != 0])  # present mask
    vals = np.setdiff1d(indx_new, indx_ori)

    if len(vals) > 0:
        begin = its + 1  # where new cells begin to emerge
        break




# Dividing the FOV into regions that belong to certain tetrads using the watershed algorithm
TET_exists = tet["TET_exists"]
if begin != 0:
    int1 = range(begin, len(ART))
    new_indx = [None] * len(ART)
    
    for its in int1:
        A1 = ART[its - 1]
        A2 = ART[its]
        indx_ori = np.unique(A1[A1 != 0])  # previous mask
        indx_new = np.unique(A2[A2 != 0])  # present mask
        vals = np.setdiff1d(indx_new, indx_ori)
        new_indx[its] = vals
    
    kka = 0
    new_indx_new = []
    for ii1 in range(len(new_indx)):
        if new_indx[ii1] is not None and len(new_indx[ii1]) > 0:
            kka += 1
            new_indx_new.append(new_indx[ii1])
    
    new_born = np.unique(np.concatenate(new_indx_new))
    
    I2 = np.zeros(ART[start].shape, dtype=np.uint16)
    for ccell in range(TET_obj):
        if tet_ids['TET_ID'][ccell, 0] != -1:
            if dead_tets[ccell] == 0:
                if TET_exists[ccell, 1] >= shock_period[1, 0] + 1:
                    mask = masks[int(shock_period[1, 0]) + 1] == ccell
                else:
                    mask = masks[int(TET_exists[1, ccell])] == ccell
                stats = regionprops(mask.astype(np.uint16))
                if len(stats) > 0:
                    cent = np.round(stats[0].centroid).astype(int)  # [y, x]
                    I2[cent[0], cent[1]] = 1
    
    I21 = dilation(I2, disk(4))  # equivalent to bwmorph(I2,'thicken',ones(9,9)) in MATLAB
    I4 = bwdist(I21)
    
    
    # Visualization
    # plt.figure(figsize=(10, 5))

    # plt.subplot(1, 2, 1)
    # plt.title('Distance Transform I4')
    # plt.imshow(I4, cmap='gray')
    # plt.colorbar()
    
    # Magnify a specific region, for example, (x_start, x_end, y_start, y_end)
    x_start, x_end = 60, 85  # Adjust these values as needed
    y_start, y_end = 115, 135  # Adjust these values as needed
    
    # plt.subplot(1, 2, 2)
    # plt.title('Magnified Region of I4')
    # plt.imshow(I4[y_start:y_end, x_start:x_end], cmap='gray')
    # plt.colorbar()
    
    # plt.show()
    
    I4 = I4.astype(np.uint16)
    # Ensure the markers are labeled and are of appropriate type
    markers = label(I2)  # Ensure markers are uint16
    
    # Perform watershed
    I3 = watershed(-I4, markers)

# =============================================================================
#     # Visualization
#     plt.figure()
#     plt.title('Watershed Segmentation I3')
#     plt.imshow(I3, cmap='jet', interpolation='nearest')
#     plt.colorbar()
#     plt.show()
# =============================================================================



# Checking which cell from ART tracks belongs to which region created using watershed
region = []
amt = []
k = 0
for iv in range(int(art['no_obj'][0, 0])):
    I12 = np.zeros(ART[start].shape, dtype=np.uint16)
    kx = 0
    for its in int1:
        I11 = (ART[its] == iv + 1).astype(np.uint16)
        if I11.sum() > 0:
            kx += 1
            if 1 <= kx <= 2:
                I11 *= 1000
        I12 += I11
    I13 = (I12 > 0).astype(np.uint16) * I3.astype(np.uint16)
    pix = np.unique(I13[I13 != 0])
    if pix.size > 0:
        for p in pix:
            amt.append([iv, p, (I13 == p).sum()])
        k += 1
        region.append([iv, pix[np.argmax([a[2] for a in amt if a[0] == iv])]])

unique_regions = np.unique([r[0] for r in region])
cell_arrays = [np.array([r[0] for r in region if r[0] == ur], dtype=np.uint16) for ur in unique_regions]

# Saving the TET ID, TET regions, and possible descendants
descendants = [set(ci) for ci in cell_arrays]
for iv in range(int(tet['TET_obj'][0, 0])):
    if tet_ids['TET_ID'][0, iv] != -1:
        if dead_tets[iv] == 0:
            T1 = masks[int(tet['TET_exists'][iv, 0])] == iv + 1
            T2 = (I3.astype(np.uint16) * T1.astype(np.uint16))
            
            # plt.figure()
            # plt.imshow(T1, cmap='gray')
            # plt.title('T1')
            # plt.show()

            pix = np.unique(T2[T2 != 0])
            if pix.size > 0:
                amt1 = [(iv, p, (T2 == p).sum()) for p in pix]
                tet_region = pix[np.argmax([a[2] for a in amt1])]
                common_indices = np.intersect1d(new_born, cell_arrays[tet_region - 1])
                common_indices = np.append(common_indices, tet_ids['TET_ID'][0, iv])
            else:
                common_indices = np.array([tet_ids['TET_ID'][0, iv]])
        else:
            common_indices = np.array([tet_ids['TET_ID'][0, iv]])
    else:
        common_indices = np.array([tet_ids['TET_ID'][0, iv]])

    descendants[iv] = set(common_indices)

alive_tets = [iv for iv in range(int(tet['TET_obj'][0, 0])) if dead_tets[iv] == 0]

# Identifying incorrectly associated descendants and reassigning
need_remov = []
works = []
for iv in alive_tets:
    common_indices1 = list(descendants[iv])
    for ittx1 in common_indices1:
        #if ittx1 in tet_ids['TET_ID']:
           # continue
        its = art['cell_exists'][0, ittx1 - 1]
        M = ART[int(its)]
        
        # plt.figure()
        # plt.imshow(M, cmap='gray')
        # plt.title('M')
        # plt.show()
        
        I_s0 = np.zeros_like(M, dtype=np.uint16)
        I_s2 = np.zeros_like(M, dtype=np.uint16)
        for rem in need_remov:
            common_indices1 = list(set(common_indices1) - set(rem))
        for it in common_indices1:
            I_s2 = (M == it).astype(int)
            I_s2 = binary_dilation(I_s2, disk(5)).astype(np.uint16)
            I_s0 += I_s2
        IA1 = (I_s0 > 0).astype(np.uint16)
        IA2 = label(IA1)
        if IA2.max() > 1:
            out = np.unique(IA2)
            sizes_occup = [(itt, (IA2 == itt).sum()) for itt in out if itt != 0]
            xx1 = max(sizes_occup, key=lambda x: x[1])[0]
            AAB = 0
            for itt2 in out:
                if itt2 != 0:
                    AB1 = IA2.copy()
                    AB2 = thin(M == ittx1, 5).astype(np.uint16)
                    AB3 = AB1 * AB2
                    pixab = np.unique(AB3)
                    if pixab.size > 1 and pixab[1] == xx1:
                        continue
                    elif pixab.size == 1:
                        continue
                    else:
                        AAB = 1
            if AAB == 1:
                for itx in alive_tets:
                    if itx != iv:
                        M = ART[its]
                        I_s0 = np.zeros_like(M, dtype=np.uint16)
                        I_s2 = np.zeros_like(M, dtype=np.uint16)
                        for it in common_indices1 + [ittx1]:
                            I_s2 = (M == it).astype(np.uint16)
                            I_s2 = binary_dilation(I_s2, disk(3)).astype(np.uint16)
                            I_s0 += I_s2
                        IA11 = (I_s0 > 0).astype(np.uint16)
                        IA21 = label(IA11)
                        if IA21.max() == 1:
                            works.append([ittx1, iv, itx])
                        else:
                            need_remov.append([ittx1, iv])

# Updating descendants based on reassignment
for work in works:
    ittx1, old_iv, new_iv = work
    descendants[old_iv].remove(ittx1)
    descendants[new_iv].add(ittx1)
for rem in need_remov:
    ittx1, old_iv = rem
    descendants[old_iv].remove(ittx1)

descendants_data = []
for iv in range(int(tet['TET_obj'][0, 0])):
    if tet_ids['TET_ID'][0, iv] == -1:
        descendants_data.append([iv, tet_ids['TET_ID'][0, iv], -1])
    else:
        descendants_data.append([iv, tet_ids['TET_ID'][0, iv], list(descendants[iv] - {tet_ids['TET_ID'][0, iv]})])


# Convert each set to a list
list_of_lists = [list(s) for s in descendants]
# Convert the list of lists to a numpy array
descendants = np.array(list_of_lists)


sio.savemat(os.path.join(sav_path, f'{pos}_descendants_new_art.mat'), {
    "I3": I3, "descendants_data": descendants_data, "descendants": descendants, 
    "alive_tets": alive_tets, "common_indices": common_indices, "cell_arrays": cell_arrays, 
    "TET_obj": tet['TET_obj']
}, do_compression=True)
