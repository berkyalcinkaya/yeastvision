from yeastvision.track.fiest.utils import cal_allob, cal_celldata, cal_allob1, binar, resize_image
import numpy as np
from skimage.transform import resize
from skimage.morphology import  erosion, square
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label
from scipy.ndimage import binary_fill_holes
from skimage.morphology import thin

# step 4: correct proSeg masks using tracked tetrad masks from step2 (tetrad.py)
def correct_proSeg_with_tetrads(art_masks: np.ndarray, tet: dict):
    Art_MT = [art_masks[i] for i in range(len(art_masks))]

    shock_period = tet['shock_period']-1 # why are we subtracting 1 here       
    TETmasks = tet['TETmasks']# plt.imshow(TETmasks[0][162])
    TET_obj=tet['TET_obj']
    TET_obj= TET_obj
   
    for iv in range(TET_obj):
        # iv = 0;
        if tet['TET_exists'][1][iv] >= shock_period[0][0]-1:#!  shock period corrected produce a single integer!!!
            tp_end = shock_period[0][1]
        else:
            tp_end = tet['TET_exists'][1][iv] 

        for its in range(tet['TET_exists'][0][iv], tp_end):# minus one added
            # its = 133;
            A1 = Art_MT[its].astype(np.double)
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
    MAT1=np.transpose(Art_MT,(1,2,0))   # transposed for step5, treat as a tensor 
    return {"Art_MT": MAT1, "shock_period": shock_period}

# step 6: remove mating from proSeg
def correct_mating(mat, art):
    Matmasks = mat['Matmasks']

    # Extract variables from loaded data
    no_obj = int(mat['no_obj'][0])
    if no_obj != 0:
        shock_period = mat['shock_period']
        MTrack = Matmasks
        cell_data = mat['cell_data']

        art_masks = art["Mask3"]
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
                M1 = (MTrack[gtp] == (iv + 1)).T
                M2 = thin(M1, 30)
                M3 = A * M2
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
        cell_data_new = cal_celldata(all_obj_new, no_obj)
        cell_data = cell_data_new
        all_obj = all_obj_new
        Matmasks = MTrack

        return {
        "Matmasks_py": Matmasks,
        "all_obj_py": all_obj,
        "cell_data_py": cell_data,
        "no_obj_py": no_obj,
        "shock_period_py": shock_period,
        "mat_artifacts_py": mat_artifacts
        }



# step 7: remove matinng events from tet masks
# art is output of step5
def correct_proSeg_with_mating(mat, art):
    all_obj = mat['all_obj']
    #Matmasks = mat['Matmasks']
    cell_data = mat['cell_data']
    Matmasks = mat["Matmasks"]

    Mask3 = art["Mask3"]


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
            M0 = (M == iv+1).astype(np.uint16)         
            A = Mask3[its].T
            M1 = binar(M0)
            M2 = thin(M1, 30)
            M3 = A * M2
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
        #print(iv)
    
    shock_period = mat['shock_period']
    no_obj = art['no_obj']
    ccell2 = art['ccell2']
    cell_exists = art['cell_exists']
    im_no = art['im_no']

    return {
        "no_obj": no_obj,
        "shock_period": shock_period,
        "Mask3": Mask3,
        "im_no": im_no,
        "ccell2": ccell2,
        "cell_exists": cell_exists
    }

