import cv2
import numpy as np
import yeastvision.track.utils as f
from yeastvision.utils import showCellNums
import statistics as stats
import matplotlib.pyplot as plt
from skimage.morphology import thin, skeletonize, opening, dilation, erosion, square
from skimage.measure import regionprops, label
from tqdm import tqdm
from yeastvision.track.cell import getBirthFrame

def get_mating_data(mating, cells):
    mating_tracks = correct_mat_tracks(track_mating(mating))
    return merge(cells, mating_tracks)


def track_mating(mating_masks, visualize = False):
    cell_margin = 20
    numbM = len(mating_masks)
    MATC = [[[] for _ in range(numbM)] for _ in range(2)]
    
    # First loop, for leading cell number one
    print("Tracking mating cells")
    print("\tLoop 1/2")
    for im_no, im in tqdm(enumerate(mating_masks)):
        I2 = im.copy()
        IS6 = np.zeros(I2.shape)
        
        if im_no == 0:
            ccel = 1
            I3 = (I2 == ccel)
            [x_cn,y_cn]=f.get_wind_coord1(I3.astype(np.uint8),cell_margin)
            I3A = I3[np.ix_(y_cn, x_cn)]
        else:
            I3A = MATC[0][im_no-1][np.ix_(y_cn, x_cn)]
            
        I3A = skeletonize((I3A>0).astype(np.uint8), method="zhang")
        
        I2A = I2[np.ix_(y_cn,x_cn)]
        
        if np.sum(I2A) == 0:
                I2A = MATC[0][im_no-1][np.ix_(y_cn, x_cn)]
                
        I3B = np.multiply(I3A.astype(np.uint16), I2A.astype(np.uint16))
        ind = stats.mode(I3B[I3B != 0])
        pix = np.nonzero(I2A==ind)
        pix0 = np.nonzero(I2A!=ind)
        I2A[pix]=1
        I2A[pix0]=0
        
        IS6[np.ix_(y_cn, x_cn)] = I2A
        
        x_cn, y_cn = f.get_wind_coord1(IS6.astype(np.uint8), cell_margin)
        
        I22 = np.zeros_like(I2)
        pix1 = np.nonzero(IS6 == 1)
        I2[pix1] = 0
        
        pix2 = np.unique(I2)
        pix2 = pix2[1:]
        
        for ity in range(0,len(pix2)):
            pix4 = np.nonzero(I2 == pix2[ity])
            I22[pix4] = ity+1
            
        MATC[0][im_no] = IS6
        MATC[1][im_no] = I22

    # Extra loop for python code
    for i in range(len(MATC)):
        for j in range(len(MATC[i])):
            if len(MATC[i][j])==0:  # check if the list is empty
                MATC[i][j] = np.zeros(I2.shape)
    
    # visualzation
    """for i in range(initial_point,len(MATC[0])):
        plt.imshow(MATC[1][i],cmap='jet',aspect='auto')
        plt.title(str(i))
        plt.show()"""
        
    # Correct for spurious segmentation
    fil = np.zeros((1,numbM))
    for it0 in range(0,numbM):
        if sum(sum(MATC[1][it0])) !=0:
            fil[0,it0] = 1
    
    if len(np.argwhere(fil == 0))>0:
        s_seg = np.argwhere(fil == 0)[-1][1]
        
        if s_seg > 0:
            for it1 in range(0,s_seg):
                MATC[1][it1] = np.zeros(IS6.shape)
                
    else:
        s_seg = -1

    # Second loop, for sequential labelling of all other cells
    ATC = np.ones(1,)
    print("Loop 2/2")
    while sum(ATC) != 0:
        
        A1 = np.zeros((1,numbM))
        A4 = 1
        A5 = 1
        rang2 = range(s_seg+1,numbM-1)
        for im_no in rang2:
            
            A2 = opening(MATC[1][im_no], square(3))

            if np.sum(A2) == 0 and A5 != 0:
                A1[0, im_no] = 0
                A5 = 1
            else:
                A1[0, im_no] = 1
                A5 = 0
            
                if im_no == 0:
                    ccel = 1
                    I3 = (MATC[1][im_no] == ccel)
                    x_cn, y_cn = f.get_wind_coord1(I3.astype(np.uint8), cell_margin)
                    I3A = I3[np.ix_(y_cn, x_cn)]              
                elif A1[0, im_no] == 1 and A1[0,im_no-1]==0 and A4 != 0:
                    ccel = min(A2[A2 != 0])
                    I3 = (MATC[1][im_no] == ccel)
                    x_cn, y_cn = f.get_wind_coord1(I3.astype(np.uint8), cell_margin)        
                    I3A = I3[np.ix_(y_cn, x_cn)]
                    A4 = 0
                else:
                    I3A = MATC[0][im_no-1][np.ix_(y_cn, x_cn)]
                    I3A[I3A != np.max(I3A)] = 0
                
                
                I2 = MATC[1][im_no+1]  
                                
                I3A = skeletonize((I3A>0).astype(np.uint8), method="zhang")
                I2A = I2[np.ix_(y_cn, x_cn)]
                
                
                if np.sum(I2A) == 0:
                    I2A = MATC[0][im_no-1][np.ix_(y_cn, x_cn)]
                    
                    pC1 = round(I2A.shape[0]/2)
                    pC2 = round(I2A.shape[1]/2)
                    
                    I2AA = np.zeros(I2A.shape)
                    I2AA[pC1, pC2] = 1
                    
                    I2AA = dilation(I2AA, np.ones((9, 9)))
                    
                    I2AA1 = np.multiply(I2AA,I2A)
                    
                    pixR = np.nonzero(I2A != np.max(I2AA1))
                    I2A[pixR] = 0
                                
                
                I3B = np.multiply(I3A.astype(np.uint16), I2A.astype(np.uint16))
                
               
                if np.sum(I3B) == 0:
                    I2A = MATC[0][im_no - 1][np.ix_(y_cn, x_cn)] 
                    pC1 = round(I2A.shape[0] / 2)
                    pC2 = round(I2A.shape[1] / 2)
                
                    I2AA = np.zeros(I2A.shape)
                    I2AA[pC1, pC2] = 1
                    I2AA = dilation(I2AA, square(9))
                    I2AA1 = np.multiply(I2AA,I2A)
                    pixR = np.nonzero(I2A != np.max(I2AA1))
                    I2A[pixR] = 0
                
                
                I3B = np.multiply(I3A.astype(np.uint16), I2A.astype(np.uint16))
                
                                
                if np.sum(I3B) == 0:
                    I2A = MATC[1][im_no][np.ix_(y_cn, x_cn)]
                    I2A = I2A.astype('float64')
                    
                    pC1 = round(I2A.shape[0] / 2)
                    pC2 = round(I2A.shape[1] / 2)
                
                    I2AA = np.zeros(I2A.shape)
                    I2AA[pC1, pC2] = 1
                    I2AA = dilation(I2AA, square(9))
                    I2AA1 = np.multiply(I2AA,I2A)
                    pixR = np.nonzero(I2A != np.max(I2AA1))
                    I2A[pixR] = 0
                    
                   
                I3B = np.multiply(I3A.astype(np.uint16), I2A.astype(np.uint16))
                
                if np.sum(I3B) == 0:
                    I2A = MATC[1][im_no][np.ix_(y_cn, x_cn)]
                    I3A = skeletonize((I2A>0).astype(np.uint8), method="zhang")
                    I3B = np.multiply(I3A.astype(np.uint16), I2A.astype(np.uint16))
                
                if len(I3B[I3B != 0]) > 0:
                    ind = stats.mode(I3B[I3B != 0])
                else:
                    ind = None
                    
                pix = np.nonzero(I2A==ind)
                pixE = np.nonzero(I2A!=ind)
                I2A[pixE]=0
                                 
                I2A[pix] = np.max(MATC[0][numbM-3]) + 1
                
                IS6B = np.zeros(I2.shape)
                IS6B[np.ix_(y_cn, x_cn)] = I2A
                               
                [x_cn,y_cn]=f.get_wind_coord1(IS6B.astype(np.uint8),cell_margin)
                                
                pixF = np.nonzero(IS6B!=0)
                IS6C = MATC[0][im_no]
                
                IS6C[pixF] = np.max(MATC[0][numbM-3])+1
                MATC[0][im_no] = IS6C
                                
                I2D = MATC[1][im_no]
                _, I2E = cv2.threshold(IS6C.astype(np.uint8), 128, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                I2E = I2E.astype(bool)
                I2D1 = np.multiply(I2D,I2E)
                I2D1 = erosion(I2D1, square(3))
                pix2 = np.nonzero(I2D1!=0)
                
                if len(I2D[pix2]) > 0:
                    cell_E = stats.mode(I2D[pix2])
                else:
                    cell_E = None
                  
                I2D[I2D == cell_E] = 0           
                I22 = np.zeros(I2D.shape)        
                pix2 = np.unique(I2D)
                pix2 = pix2[1:]
                
                for ity in range(0,len(pix2)):
                    
                    BBB = erosion(I2D == pix2[ity], square(3))
                    
                    if np.mean(BBB) > 0:
                        pix4 = np.nonzero(I2D == pix2[ity])
                        I22[pix4] = ity+1
                    else:
                        print()
                         
                MATC[1][im_no] = I22   
        ATC = np.unique(MATC[1][im_no])
        
    # cell array that contains the fully tracked MatSeg masks
    Matmasks = [[[] for _ in range(im_no+1)] for _ in range(1)]
    for ita in range(im_no+1):
        Matmasks[0][ita] = MATC[0][ita]
    
    # Save the tracks
    #sav_path = r'C:\Users\shrey\OneDrive\Desktop\RA_Orlando_Lab\MAT_2_Py_transation\Shreya_Translated\Saved Data\\'
    #name1 = sav_path + exp_name + '_MatSeg_2_Python_v2'
    #np.save(name1, Matmasks)
    
    # visualization of single labels in tracked MatSeg Masks
    """for iv in range(1, 22):
        for its in range(0, im_no+1):
            #its = im_no
            plt.imshow(Matmasks[0][its] == iv)
            plt.title(str(its+1) + 'c =' + str(iv))
            plt.show()"""
    
    # visualization of full tracked MatSeg Masks
    if visualize:
        for its in range(0, numbM-1):
            plt.imshow(Matmasks[0][its],cmap='jet',aspect='auto')
            showCellNums(Matmasks[0][its])
            plt.title(str(its+1))
            plt.show()
    return np.array(Matmasks, dtype = np.uint16)


def correct_mat_tracks(Matmasks):
    # correction of single labels in tracked MatSeg Masks
    nu = Matmasks[0].shape[0]
    MATrack = [[[] for _ in range(nu)] for _ in range(1)]
    Mrack = [[[] for _ in range(nu)] for _ in range(1)]
    cell_margin = 20
    no_obj = np.max(Matmasks[0, nu-1, :, :])
    
    min_size, avg_size, max_size = f.avg_cell_size(Matmasks, nu, no_obj)
    print("correcting tracks")
    print("\t Loop 1/1")
    for iv in tqdm(range(1, no_obj.astype(int)+1)):
        for its in range(nu):
            canvs  = np.zeros(Matmasks[0][0].shape)
            present_cell = Matmasks[0][its]==iv
            
            if sum(sum(present_cell))==0:
                pass
            else:
                [x_cn,y_cn]=f.get_wind_coord1(present_cell.astype(np.uint8),cell_margin)
                I33 = present_cell[np.ix_(y_cn,x_cn)]
            
                if its == 0:
                    past_cell = Matmasks[0][its]==iv
                elif iv==1 and its!=0:
                    past_cell = Mrack[0][its-1]==iv
                elif its==0 and iv!=1:
                    past_cell = Matmasks[0][its]==iv
                elif its!=0 and iv!=1:
                    past_cell = MATrack[0][its-1]==iv
                
                si1 = sum(sum(present_cell))
                si2 = sum(sum(past_cell))
                
                if si1 <= si2-0.1*si2 and np.average(past_cell) > avg_size - 0.2*avg_size:
                    if iv == 1:
                        Mrack[0][its] = past_cell
                    else:
                        pass
                    [x_cn,y_cn]=f.get_wind_coord1(past_cell.astype(np.uint8),cell_margin)
                    I3A = past_cell[np.ix_(y_cn,x_cn)]
                    
                #we can use other morphological features for adaptive thresholding
                elif si1 >= si2+0.13*si2 and si2!=0 and np.average(past_cell) > avg_size - 0.2*avg_size:
                    if iv == 1:
                        Mrack[0][its] = past_cell
                    else:
                        pass
                    [x_cn,y_cn]=f.get_wind_coord1(past_cell.astype(np.uint8),cell_margin)
                    I3A = past_cell[np.ix_(y_cn,x_cn)]
                    
                elif si2 == 0:
                    I3A = present_cell[np.ix_(y_cn,x_cn)]
                    
                else:
                    Mrack[0][its] = present_cell
                    [x_cn,y_cn]=f.get_wind_coord1(present_cell.astype(np.uint8),cell_margin)
                    I3A = present_cell[np.ix_(y_cn,x_cn)]
                    
                if si1 > si2+0.045*si2 and si2 !=0 and np.average(past_cell) > avg_size - 0.2*avg_size:
                    IA = I3A.astype(np.float32) - past_cell[np.ix_(y_cn,x_cn)].astype(np.float32)
                    IA[IA<0] = 0
                    IA = opening(IA, square(3))
                    I3A1 = opening(I3A-IA, square(3))
                    I3A = I3A1
                    
                canvs1 = np.zeros(Matmasks[0][its].shape)
                canvs1[np.ix_(y_cn, x_cn)] = I3A
                pixo = np.nonzero(canvs1 != 0)
                
                if iv == 1:
                    canvs[pixo] = iv
                    MATrack[0][its] = canvs
                else:
                    canvs2 = MATrack[0][its]
                    canvs2[pixo] = iv
                    MATrack[0][its] = canvs2
            
    # visualization of previously tracked MatSeg Masks
    """for iv in range(1,no_obj.astype(int)+1):
        for its1 in range(nu):
            plt.imshow(Matmasks[0][its1]==iv)
            plt.title(str(its1+1) + ' c = ' + str(iv) + ' Before correction')
            plt.show()
            plt.imshow(MATrack[0][its1]==iv)
            plt.title(str(its1+1) + ' c = ' + str(iv) + ' After correction')
            plt.show()"""
            
    # # Save the corrected masks
    # sav_path = r'C:\Users\shrey\OneDrive\Desktop\RA_Orlando_Lab\MAT_2_Py_transation\Correction Code\Saved Data\\'
    # name1 = sav_path + 'Corrected_MatSeg_2_Python'
    # np.save(name1, MATrack)
    return np.array(MATrack, dtype = np.uint16)

def merge(full_masks, Matmasks, visualize = False):
    #Corrected MATracking masks
    numbM = Matmasks[0].shape[0]
    no_obj = np.max(Matmasks[0, numbM-1, :, :])
    MATC = [[[] for _ in range(numbM)] for _ in range(1)]
    new_tracks = np.zeros_like(Matmasks)
    print("Merging mating cells into cell mask")
    print("\t Loop 1/1")
    for cxell in tqdm(range(1,no_obj.astype(int)+1)):
        for im_no, mask in enumerate(full_masks[0:numbM]):
            if cxell == 1:
                I2 = mask.copy()
            else:
                I2 = MATC[0][im_no]
                
            I3 = Matmasks[0][im_no]
            I3 = (I3==cxell)
            
            if np.sum(I3[:])==0:
                pass
            else:
                s1 = np.sum(np.sum(I3))
                I3 = I3.astype(np.uint16)
                
                I2B1 = np.multiply(I2,I3)
                I2BB = skeletonize((I2B1>0), method="zhang")
                I2B1 = np.multiply(I2B1,I2BB)
                
                objs = np.unique(I2B1).reshape(1, -1)
                
                size = [[] for _ in range(2)]
                for i in objs[0][1:]:
                    s = regionprops(label(I2B1==i))[0].area
                    size[0].append(i)
                    size[1].append(s)
                    
                    
                if (len(size[1])==0 or len(size[1])==1):
                    I2A = np.zeros(I2.shape)
                    I2A[I2==size[0][0]] = 1
                    props_Imat = regionprops(label(I3))[0]
                    props_Ifull = regionprops(label(I2A))[0]
                    m1 = np.abs(props_Imat.extent-props_Ifull.extent)/props_Imat.extent
                    m2 = np.abs(props_Imat.area-props_Ifull.area)/props_Imat.area
                    
                    if len(regionprops(label(I3)))>1:
                        bbox = props_Ifull.bbox
                        IZ = I2A[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                    
                    else:
                        if m1 > 0.01 or m2 > 0.09:
                            bbox = props_Imat.bbox
                            IZ = I3[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                        else:
                            bbox = props_Ifull.bbox
                            IZ = I2A[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                                            
                else:
                    size_ind = np.array(size[0])
                    size_con = np.array(size[1])*100/np.sum(np.array(size[1]))
                    m = size_con >= 10
                    size_con_filt = size_con[m]
                    size_ind_filt = size_ind[m]
                    
                    I2A = np.zeros(I2.shape)
                    for i in size_ind_filt:
                        I2A[I2==i] = 1
                        
                    props_Imat = regionprops(label(I3))[0]
                    props_Ifull = regionprops(label(I2A))[0]
                    m1 = np.abs(props_Imat.extent-props_Ifull.extent)/props_Imat.extent
                    m2 = np.abs(props_Imat.area-props_Ifull.area)/props_Imat.area
                                      
                    if len(regionprops(label(I3)))>1:
                        bbox = props_Ifull.bbox
                        IZ = I2A[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                    
                    else:
                        if m1 > 0.03 or m2 > 0.09:
                            bbox = props_Imat.bbox
                            IZ = I3[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                        else:
                            bbox = props_Ifull.bbox
                            IZ = I2A[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                
                I2 = I2.astype(np.uint16)
                I44 = np.zeros(I2.shape)
                I44[bbox[0]:bbox[2], bbox[1]:bbox[3]] = IZ
                pix = np.nonzero(I44)
                new_num = np.max(I2) + 1
                I2[pix] = new_num
                MATC[0][im_no] = I2
                new_tracks[im_no][pix] = new_num
                
                if visualize:
                    plt.imshow(I2==np.max(I2))
                    plt.title('cxell: ' + str(cxell) + ', im_no: ' + str(im_no))
                    plt.show()
    MATC[0].append(MATC[0][-1])
    new_tracks = list(new_tracks)
    new_tracks.append(new_tracks[-1])
    return np.array(MATC[0], dtype = np.uint16), np.array(new_tracks, dtype = np.uint16)

