from yeastvision.track.fiest.utils import cal_allob, cal_celldata, cal_allob2, binar, resize_image
import numpy as np
from skimage.transform import resize
from skimage.morphology import  erosion, square
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label
from scipy.ndimage import binary_fill_holes
from skimage.morphology import thin

# step 4: correct proSeg masks using tracked tetrad masks from step2 (tetrad.py)
def correct_proSeg_with_tetrads(art_masks: np.ndarray, shock_period, TET_obj, TET_exists, TETmasks):
    Art_MT = [art_masks[i] for i in range(len(art_masks))]
    for iv in range(TET_obj):# TET_obj COMES FROM STEP 2
        # iv = 0;
        if TET_exists[1][iv] >= shock_period[0]-1:#!  shock period corrected produce a single integer!!!  # TET_exists COMES FROM STEP 2
            tp_end = shock_period[0]
        else:
            tp_end = TET_exists[1][iv] 

        for its in range(TET_exists[0][iv], tp_end):# minus one added
            # its = 42 ;
            A1 = Art_MT[its].astype(np.double)
            if shock_period[0]-1 <= its <= shock_period[1]:#!!shock period corrected produce a single integer!!!!!!!!!!!!!!!!!!!!
                T1 = (TETmasks[shock_period[0]-1] == iv + 1).astype(np.double)#! TETmasks correct and shock period corrected produce a single integer[] # TETmasks COMES FROM STEP 2 
                thresh = 0.6
            else:
                T1 = (TETmasks[its] == iv + 1).astype(np.double)#!!!!!!!!!!!!!!!!!!  
                thresh = 0.95 # plt.imshow(T1)

            if T1.shape != A1.shape:   
                T1 = resize_image(T1, A1.shape,).astype(np.float64)
            else:
                T1 = T1.astype(np.float32)
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

    for iv in range(TET_obj):  # TET_OBJ COMES FROM STEP 2
        # iv = 0
         if TET_exists[1][iv] > shock_period[1] and TET_exists[0][iv] < shock_period[0]:
            s1 = np.sum(TETmasks[shock_period[1]+1] == iv+1)#!!!!!!!!!!!!!!!!!!!!!!
            for its in range(shock_period[1]+1, TET_exists[1][iv]):#!!!!!!!!!!!!!!!!!!!!!!!!
                # its = 134;
                A1 = Art_MT[its].astype(np.double)
                # plt.imshow(A1)
                T1 = (TETmasks[its] == iv + 1).astype(np.double)
                # plt.imshow(T1)
                
                

                s2 = np.sum(TETmasks[its] == iv + 1)
                if its == TET_exists[1][iv]:
                    s3 = np.sum(TETmasks[its] == iv + 1)
                else:
                    s3 = np.sum(TETmasks[its + 1] == iv + 1)

                if s2 < s1 - 0.1 * s1:
                    if s3 > s2 + 0.1 * s2:
                        T1 = (TETmasks[its - 1] == iv + 1).astype(np.double)
                    else:
                        break

                s1 = s2
                #T1 = resize(T1, A1.shape, order=0, preserve_range=True)
                
                if T1.shape != A1.shape:
                    T1 = resize_image(T1, A1.shape,).astype(np.float64)
                else:
                    T1 = T1.astype(np.float64)
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
   
    # Art_MT CONTAINS THE SPORE MASKS REPLACED THE PROSEG MASS FOR SPORES
    return Art_MT 

# step 6: remove mating from proSeg
def correct_mating(Matmasks, Mask7, mat_no_obj, Mat_cell_data, cell_data):

    if mat_no_obj  != 0: # mat_no_obj COMES FROM STEP 3
        MTrack = Matmasks # Matmasks COMES FROM STEP 3
        art_masks = Mask7 # this is a tensor do not transpose # Mask7 COMES FROM STEP 5
    
        mat_artifacts = []

        # Resize MTrack to match ART masks
        for its in range(len(MTrack)):
            if MTrack[its].size > 2:
                if MTrack[its].shape != art_masks[:,:,its].shape:
                    MTrack[its] = resize(MTrack[its], art_masks[:,:,its].shape, order=0, preserve_range=True, anti_aliasing=False)# art_masks[:,:,its] added to correct for tensor to list format

        tp_end = len(art_masks[0][0])# tensor correction, gets last time point
        if len(MTrack) != tp_end: # loop for adding zeros matrixes 
            for its in range(len(MTrack[its]), tp_end):
                MTrack.append(np.zeros_like(MTrack[int(min(Mat_cell_data[:, 0])) - 1], dtype=np.uint16)) # final_mat_cell_data COMES FROM STEP 3, 

        # Correcting mating tracks
        cor_data = np.zeros((3, mat_no_obj ))
        size_cell = np.zeros((mat_no_obj , len(MTrack)))
        morph_data = np.zeros((mat_no_obj , len(MTrack)))
        outlier_tps = [None] * mat_no_obj 
        good_tps = [None] * mat_no_obj 
        

        for iv in range(mat_no_obj ):
            # iv = 0;
            int_range = range(int(Mat_cell_data[iv,0]), int(Mat_cell_data[iv,1]))  # Adjusting for 0-based indexing
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



        for iv in range(mat_no_obj ):
            # iv = 0
            int_range = range(int(Mat_cell_data[iv,0]), int(Mat_cell_data[iv,1]))
            if np.var(morph_data[iv, int_range]) > 0.02:
                mat_artifacts.append(iv)




        for iv in range(mat_no_obj ):
            outlier = sorted(outlier_tps[iv])
            good = sorted(good_tps[iv])
            int_range = range(int(Mat_cell_data[iv,0]), int(Mat_cell_data[iv,1]))
            while outlier:
                its = min(outlier)
                gtp = max([g for g in good if g < its], default=min([g for g in good if g > its], default=its))
                A = art_masks[:,:,its]
                
    # =============================================================================
    #             plt.figure()
    #             plt.imshow(np.uint16(M), cmap='gray')
    #             plt.title('M')
    #             plt.show()
    # =============================================================================
                
                M1 = (MTrack[gtp] == (iv + 1))
                M2 = thin(M1, 30)
                M3 = A * M2
                
                # plt.figure()
                # plt.imshow(np.uint16(M3), cmap='gray')
                # plt.title('M3')
                # plt.show()
                
                indx = np.unique(A[M3 != 0])
                if indx.size > 0:
                    X1 = np.zeros_like(MTrack[its])
                    for itt2 in indx:
                        if np.sum(M3 == itt2) > 5:
                            X1[A == itt2] = 1
                    X1 = binary_fill_holes(X1)
                    # plt.imshow(X1)
                    X2 = label(X1)
                    if np.max(X2) <= 1 and abs(np.sum(X1) - cor_data[0, iv]) <= 2 * cor_data[1, iv]:
                        MTrack[its][MTrack[its] == (iv + 1)] = 0
                        (MTrack[its])[X1 == 1] = iv + 1
                    else:
                        MTrack[its][MTrack[its] == (iv + 1)] = 0
                        MTrack[its][MTrack[gtp] == (iv + 1)] = iv + 1
                outlier = [o for o in outlier if o != its]
                good.append(its)
                good = sorted(good)




        for iv in range(mat_no_obj ):
            if cell_data[iv,1] != tp_end:
                count = 0
                for its in range(int(Mat_cell_data[iv,1])+1, tp_end): # its=156 check plus one is needed!!!!!!!!!!!!!
                    A = art_masks[:,:,its]
                    M1 = (MTrack[its - 1] == (iv + 1))
                    M2 = thin(M1, 30)
                    M3 = A * M2
                    indx = np.unique(A[M3 != 0])
                    if indx.size > 0:
                        X1 = np.zeros_like(MTrack[its])
                        for itt2 in indx:
                            if np.sum(M3 == itt2) > 5:
                                X1[A == itt2] = 1 # plt.imshow(X1)
                        if abs(np.sum(X1) - cor_data[0, iv]) > 2 * cor_data[1, iv]:
                            count += 1
                            MTrack[its][MTrack[its - 1] == (iv + 1)] = iv + 1
                        else:
                            MTrack[its][X1 == 1] = iv + 1
                    else:
                        count += 1
                        MTrack[its][MTrack[its - 1] == (iv + 1)] = iv + 1
                if count / (tp_end -Mat_cell_data[iv, 0]) > 0.8:
                    mat_artifacts.append(iv + 1)


        # Remove cell artifacts and rename
        if mat_artifacts:
            all_ccel = list(range(1, mat_no_obj  + 1))
            mat_artifacts = sorted(set(mat_artifacts))
            for iv in mat_artifacts:
                for its in range(len(MTrack)):
                    MTrack[its][MTrack[its] == iv] = 0
            good_cells = sorted(set(all_ccel) - set(mat_artifacts))
            for iv in range(len(good_cells)):
                for its in range(len(MTrack)):
                    MTrack[its][MTrack[its] == good_cells[iv]] = iv + 1
            mat_no_obj = len(good_cells)
            
            

        # Recalculating MAT Data

        all_obj_new = cal_allob2(mat_no_obj, MTrack, list(range(len(MTrack))))
        cell_data_new = cal_celldata(all_obj_new, mat_no_obj)# should mat_no_obj

        # THIS VARIABLES WILL BE USED IN STEP 7
        final_mat_cell_data = cell_data_new 
        mat_all_obj = all_obj_new
        Matmasks = MTrack

        # sio.savemat(f'{sav_path}{pos}_MAT_16_18_Track1_py.mat', {
        #     "Matmasks_py": Matmasks,
        #     "all_obj_py": all_obj,
        #     "cell_data_py": cell_data,
        #     "no_obj_py": no_obj,
        #     "shock_period_py": shock_period,
        #     "mat_artifacts_py": mat_artifacts
        # }, do_compression=True)


    return Matmasks, final_mat_cell_data, mat_no_obj



# step 7: Removes the mating events from sposeg tracks based on the overlapping tracked indices 
def correct_proSeg_with_mating(Matmasks, Mask7, art_cell_exists, mat_no_obj, final_mat_cell_data):
    for iv in range(int(mat_no_obj)): # mat_no_obj COMES FROM STEP 6  
        # iv = 3
        indx_remov = []
        final_indx_remov = []

        for its in range(int(final_mat_cell_data[iv,0]), int(final_mat_cell_data[iv,1])):# final_mat_cell_data COMES FROM STEP 6  
            # its = 167
            M = Matmasks[its]#  Matmasks COMES FROM STEP 6  
            
            # sio.savemat(os.path.join(sav_path, 'M2_py.mat'), {
            #     "M2_py": M2
            # })
            
# =============================================================================
#                 plt.figure()
#                 plt.imshow(M, cmap='gray')
#                 plt.title('M')
#                 plt.show()
# =============================================================================
            
            M0 = (M == iv+1).astype(np.uint16)
            
# =============================================================================
#                 plt.figure()
#                 plt.imshow(M0, cmap='gray')
#                 plt.title('M0')
#                 plt.show()
# =============================================================================
            
            A = Mask7[:,:,its]
            
            # plt.figure()
            # plt.imshow(A, cmap='gray')
            # plt.title('A')
            # plt.show()
            
            
            M1 = binar(M0)
            
# =============================================================================
#                 plt.figure()
#                 plt.imshow(M1, cmap='gray')
#                 plt.title('M1')
#                 plt.show()
# =============================================================================
            
            M2 = thin(M1, 30)
            #M2 = skeletonize(M1)
            
            # plt.figure()
            # plt.imshow(M2, cmap='gray')
            # plt.title('M2')
            # plt.show()
            
            M3 = A * M2
            
            # plt.figure()
            # plt.imshow(M3, cmap='gray')
            # plt.title('M3')
            # plt.show()
            
        
            
            indx = np.unique(A[M3 != 0])
            
            
            if indx.size > 0:
                for itt2 in indx:
                    if np.sum(M3 == itt2) > 5:
                        indx_remov.append(itt2)
        
        
        
    
        if len(indx_remov) > 0:
            indx_remov_inter = np.unique(indx_remov)
            final_indx_remov = np.unique(indx_remov)
            for itt1 in indx_remov_inter:
                # itt1 = 6
                dist_data = -1 * np.ones(len(Mask7[0][0]))
                for its1 in range(int(final_mat_cell_data[iv,0]), int(art_cell_exists[1,int(itt1)-1])): # final_mat_cell_data AND art_cell_exists COMES FROM STEPS 6 AND 5                  
                    # its1 = 141
                    if its1 >= art_cell_exists[0,int(itt1)-1]:
                        M6 = (Mask7[:,:,its1] == itt1)  #  plt.imshow(M6, cmap='gray')# Mask7 COMES FROM STEP 5    
                        M7 = (Matmasks[its1] == iv + 1) #  plt.imshow(M7, cmap='gray')# Matmasks COMES FROM STEP 6   
                        dist_data[its1] = np.sum(M6 * M7) / np.sum(M6)
                
                if np.any(dist_data != -1):
                    first_ov = np.where(dist_data != -1)[0][0]
                    last_ov = np.where(dist_data != -1)[0][-1]
                    val_avg = np.median(dist_data[first_ov:last_ov])
                    if val_avg <= 0.4:
                        final_indx_remov = np.setdiff1d(final_indx_remov, itt1)
            
            for its in range(int(final_mat_cell_data[iv,0]), len(Mask7[0][0])):# final_mat_cell_data AND Mask7 COMES FROM STEPS 6 AND 5   
                for itt in final_indx_remov:
                    Mask7[:,:,its][Mask7[:,:,its] == itt] = 0
    
    return Mask7