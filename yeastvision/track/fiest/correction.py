import numpy as np
from skimage.transform import resize
from skimage.morphology import  erosion, square
from skimage.filters import threshold_otsu
    
def correct_proSeg_with_tetrads(art_masks: np.ndarray, tet: dict):
    Art_MT = [art_masks[i] for i in range(len(art_masks))]

    shock_period = tet['shock_period']
    TETmasks = []
    for i in range(len(tet['TETmasks'])):
        tet_masks_refs = tet['TETmasks'][i]
        for ref in tet_masks_refs:
            mask = ref
            TETmasks.append(mask)
    
    for iv in range(int(tet['TET_obj'][0][0])):
        # iv = 0;
        if tet['TET_exists'][iv, 1] >= shock_period[0] - 1:
            tp_end = shock_period[1]
        else:
            tp_end = tet['TET_exists'][iv, 1]

        for its in range(int(tet['TET_exists'][iv, 0]) - 1, int(tp_end + 1)):
            # its = 72;
            A1 = Art_MT[its].astype(np.double)
            if shock_period and (shock_period[0] <= its <= shock_period[1]):
                T1 = (TETmasks[int(shock_period[0]) - 1] == iv + 1).astype(np.double)
                thresh = 0.6
            else:
                T1 = (TETmasks[its] == iv + 1).astype(np.double)
                thresh = 0.95

            # T1 = resize(T1, A1.shape, order=0, preserve_range=True)
            T1 = resize(T1.T, (A1).shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.float64)
            # T1 = resize_image(T1, A1.shape).astype(np.float64)
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

    for iv in range(int(tet['TET_obj'][0][0])):
        # iv = 0
        if shock_period and (tet['TET_exists'][iv, 1] > shock_period[1] and tet['TET_exists'][iv, 0] < shock_period[0]):
            s1 = np.sum(TETmasks[int(shock_period[1])] == iv + 1)
            for its in range(int(shock_period[1]), int(tet['TET_exists'][iv, 1]) - 1):
                # its = 134;
                A1 = Art_MT[its].astype(np.double)
                # plt.imshow(A1)
                T1 = (TETmasks[its] == iv + 1).astype(np.double).T
                # plt.imshow(T1)
                
                s2 = np.sum(TETmasks[its] == iv + 1)
                if its == tet['TET_exists'][iv, 1]:
                    s3 = np.sum(TETmasks[its] == iv + 1)
                else:
                    s3 = np.sum(TETmasks[its + 1] == iv + 1)

                if s2 < s1 - 0.1 * s1:
                    if s3 > s2 + 0.1 * s2:
                        T1 = (TETmasks[its - 1] == iv + 1).astype(np.double)
                    else:
                        break

                s1 = s2
                T1 = resize(T1, A1.shape, order=0, preserve_range=True)
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

