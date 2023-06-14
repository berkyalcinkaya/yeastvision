import numpy as np
from skimage.measure import regionprops
from skimage.morphology import thin, skeletonize, opening, dilation, erosion
import statistics
from yeastvision.track.utils import get_bbox_coords
import statistics as stats
from yeastvision.track.cell import getCellData
import matplotlib.pyplot as plt
from tqdm import tqdm

def track_to_cell(objs, cells):
    tracked_mask = np.zeros_like(objs)
    im_idx = 0 
    for obj_mask, cell_mask in zip(objs, cells):
        for obj_num in np.unique(obj_mask):
            if obj_num > 0:
                vals, counts  = np.unique(cell_mask[obj_mask==obj_num], return_counts=True)

                if np.any(vals==0):
                    vals, counts = vals[1:], counts[1:]

                tracked_mask[im_idx][obj_mask==obj_num] = vals[np.argmax(counts)]
        im_idx += 1
    return tracked_mask

def track_obj(ims, cp_masks, masks, do_skeletonize):
    MATC = [[],[]]
    for i in range(len(ims)-1):
        I1 = ims[i].copy()
        I2 = masks[i+1].copy()
        IS6 = np.zeros_like(I2)

        if i==0:
            ccel = 1
            I3=(I2==ccel)
            r_slice, c_slice = get_bbox_coords(I3)
            I3A = I3.copy()[r_slice, c_slice]
        else:
            I3A = MATC[0][i-1].copy()[r_slice, c_slice]
        
        if do_skeletonize:
            I3A = skeletonize((I3A>0).astype(np.uint8), method="zhang")
        else:
            r_centroid,c_centroid = regionprops((I3A>0).asytpe(np.uint8))[0].centroid
            r_centroid, c_centroid =int(r_centroid), int(c_centroid)
            I3A = np.zeros_like(I3A, dtype = np.uint8)
            I3A[r_centroid,c_centroid] = 1
            I3A = dilation(I3A, np.ones((3,3)))
        I2A = I2.copy()[r_slice, c_slice]

        if np.sum(I2A.astype(np.uint8))==0:
            I2A = MATC[0][i-1].copy()[r_slice, c_slice]

        
        I3B = np.multiply(I3A.astype(np.uint16),(I2A.astype(np.uint16)))
        ind = stats.mode(I3B[I3B!=0])
        pix = np.nonzero(I2A==ind)

        I2A[pix] = 1

        IS6[r_slice, c_slice] = I2A.copy()
        r_slice, c_slice = get_bbox_coords(IS6)

        I22 = np.zeros_like(I2)
        pix1 = np.nonzero(IS6!=0)
        I2[pix1] = 0

        pix2 = np.unique(I2)
        pix2 = pix2[1:]
        for ity in range(0, len(pix2)):
            pix4 = np.nonzero(I2==pix2[ity])
            I22[pix4] = ity+1

        MATC[1].append(I22)
        MATC[0].append(IS6)
    
    plt.imshow(MATC[0][-1])
    plt.show()


    # loop 2
    ATC = 1
    while np.sum(ATC)!=0:
        A1 = np.zeros((1,len(ims)))
        A4 = 1
        A5 = 1
        #for i in range(len(ims)-2):
        for i in [0]:
            A2 = opening(MATC[1][i], footprint = np.ones((3,3)))

            if A2.sum() == 0 and A5!=0:
                A1[0,i]=0
                A5 = 1
            else:
                A1[0,i]=1
                A5 = 0

                if i==0:
                    ccel=1
                    I3 = MATC[1][i]==ccel
                    r_slice, c_slice = get_bbox_coords(I3)
                    I3A = I3[r_slice, c_slice]
                elif A1[0,i]==1 and A1[0, i-1]==0 and A4!=0:
                    ccel = np.min(A2[A2!=0])
                    I3 = MATC[1][i]==ccel
                    r_slice, c_slice = get_bbox_coords(I3)
                    I3A = I3[r_slice, c_slice]
                    A4 = 0
                else:
                    I3A = MATC[0][i-1].copy()[r_slice, c_slice]
                    I3A[I3A!=np.max(I3A)] = 0
                
                I2 = MATC[1][i+1].copy()

                if skeletonize:
                    I3A = skeletonize((I3A>0).astype(np.uint8), method="zhang")
                else:
                    r_centroid,c_centroid = regionprops((I3A>0).asytpe(np.uint8))[0].centroid
                    r_centroid, c_centroid =int(r_centroid), int(c_centroid)
                    I3A = np.zeros_like(I3A, dtype = np.uint8)
                    I3A[r_centroid,c_centroid] = 1
                    I3A = dilation(I3A, np.ones((3,3)))

                I2A = I2.copy()[r_slice, c_slice]

                if I2A.sum()==0:
                    I2A = MATC[0][i-1].copy()[r_slice, c_slice]
                    #print("I2A shape", I2A.shape)
                    pC1 = round(I2A.shape[0]/2)
                    pC2 = round(I2A.shape[1]/2)

                    I2AA = np.zeros_like(I2A)
                    I2AA[pC1,pC2]=1
                    I2AA = dilation(I2AA, footprint = np.ones((9,9)))
                    I2AA1 = np.multiply(I2AA, I2A)
                    pixR = np.nonzero(I2A!=np.max(I2AA1))
                    I2A[pixR] = 0
                
                I3B = np.multiply(I3A.astype(np.uint16),I2A.astype(np.uint16))

                if I3B.sum()==0:
                    I2A = MATC[0][i-1].copy()[r_slice, c_slice]
                    pC1 = round(I2A.shape[0]/2)
                    pC2 = round(I2A.shape[1]/2)

                    I2AA = np.zeros_like(I2A)
                    I2AA[pC1, pC2]=1
                    I2AA = dilation(I2AA,footprint = np.ones((9,9)))
                    I2AA1 = np.multiply(I2AA, I2A)
                    pixR = np.nonzero(I2A!=np.max(I2AA1))
                    I2A[pixR] = 0
                
                I3B = np.multiply(I3A.astype(np.uint16),I2A.astype(np.uint16))
                
                if I3B.sum()==0:
                    I2A = MATC[1][i].copy()[r_slice, c_slice]
                    pC1 = round(I2A.shape[0]/2)
                    pC2 = round(I2A.shape[1]/2)

                    I2AA = np.zeros_like(I2AA)
                    I2AA[pC1, pC2]=1
                    I2AA = dilation(I2AA,footprint = np.ones((9,9)))
                    I2AA1 = np.multiply(I2AA, I2A)
                    pixR = np.nonzero(I2A!=np.max(I2AA1))
                    I2A[pixR] = 0
                
                I3B = np.multiply(I3A.astype(np.uint16),I2A.astype(np.uint16))

                if I3B.sum()==0:
                    I2A = MATC[1][i].copy()[r_slice, c_slice] 
                    I3A = skeletonize((I2A>0).astype(np.uint8), method = "zhang") 
                    I3B = np.multiply(I3A.astype(np.uint16),I2A.astype(np.uint16))
                
                ind = stats.mode(I3B[I3B!=0])
                pix = np.nonzero(I2A==ind)
                pixE = np.nonzero(I2A!=ind)
                I2A[pixE]=0

                I2A[pix] = np.max(MATC[0][-2])+1

                IS6B = np.zeros_like(I2)
                IS6B[r_slice, c_slice] = I2A.copy()
                r_slice, c_slice = get_bbox_coords(IS6B>0)
                
                pixF = np.nonzero(IS6B!=0)
                IS6C = MATC[0][i].copy()
                IS6C[pixF] = np.max(MATC[0][-2])+1
                MATC[0][i] = IS6C.copy()

                I2D = MATC[1][i]
                I2E = (IS6C>0).astype(np.uint8)
                plt.imshow(I2E)
                plt.show()
                I2D1 = np.multiply(I2D, I2E)
                I2D1 = erosion(I2D1, footprint = np.ones((3,3)))
                pix2 = np.nonzero(I2D1!=0)
                cell_E = stats.mode(I2D[pix2])
                I2D[I2D==cell_E] = 0

                I22 = np.zeros_like(I2D)

                pix2 = np.unique(I2D)
                pix2 = pix2[1:]
                for ity in range(0,len(pix2)):
                    BBB = erosion((I2D == pix2[ity]).astype(np.uint8), 
                                footprint = np.ones((3,3)))

                    if np.mean(BBB) > 0:
                        pix4 = np.nonzero(I2D==pix2[ity])
                        I22[pix4] = ity
                    
                
                MATC[1][i]=I22.copy()

            ATC = np.unique(MATC[1][i])
    Matmasks = [0]*(len(ims)-2)
    for i in range(len(ims)-2):
        Matmasks[i] = MATC[0][i]

    return Matmasks, add_obj_tracks_to_cell(ims, cp_masks, Matmasks, MATC, do_skeletonize)

def add_obj_tracks_to_cell(ims, cp_masks, Matmasks, MATC, do_skeleton):
    no_objs = np.max(Matmasks[-1]) 
    # replace cellpose uisng matSeg skeletons

    for cxell in range(1,no_objs+1):
        for i in range(len(Matmasks)):
            I1 = ims[i]
            if cxell==1:
                I2 = cp_masks[i]
            else:
                I2 = MATC[0][i]
            I3 = Matmasks[i]
            if i == 0:
                I4 = Matmasks[i]
            else:
                I4 = Matmasks[i-1]
            
            I3 = I3==cxell

            if np.sum(I3)!=0:
                siz1 = np.count_nonzero((I3==cxell).astype(np.uint8))
                siz2 = np.count_nonzero((I4==cxell).astype(np.uint8))
                I3 = I3.astype(np.uint16)

                if siz1>siz2 + 0.2*siz2:
                    r_slice, c_slice = get_bbox_coords(I4)
                    I3A = (I4[r_slice, c_slice]>0).astype(np.uint8)
                else:
                    r_slice, c_slice = get_bbox_coords(I3)
                    I3A = (I3[r_slice, c_slice]>0).astype(np.uint8)
                
                I2A = I2[r_slice, c_slice]
                I2B1 = np.multiply(I2A, I3A.astype(np.uint16))
                I2B = thin(I2B1, max_num_iter=2)
                I2B = opening(I2B, np.ones((3,3)))

                if do_skeleton:
                    I2BB = skeletonize((I2B>0).astype(np.uint8), method = "zhang")
                else:
                    r_centroid,c_centroid =  regionprops((I2B>0).astype(np.uint8))[0].centroid
                    r_centroid, c_centroid = int(r_centroid), int(c_centroid)
                    I2BB = np.zeros_like(I2B)
                    I2BB[r_centroid, c_centroid] = 1
                    I2BB = dilation(I2BB, np.zeros((3,3)))

                I2B1 = np.multiply(I2B1, I2BB.astype(np.uint16))
                objs = np.unique(I2B1)
                size1 = np.zeros((I2B1.shape[0], 2))
                for it1 in objs[1:]:
                    A = objs[1:]==it1
                    A = np.nonzero(A)[0]
                    size1[:,0][A]=it1
                    size1[:,1][A] = np.count_nonzero((I2B1==it1).astype(np.uint8))

                ab1 = size1.copy()

                if ab1.shape[0]!=1:
                    ab1 = size1[np.argsort(size1[:,1])]
                    s1 = np.count_nonzero((I2A == ab1[1,0]).astype(np.uint8))
                    s2 = np.count_nonzero((I2A == ab1[0,0]).astype(np.uint8))
                    if s1/s2 <= 0.25:
                        indx = ab1[0,0]
                    else:
                        indx = ab1[:2,1]
                else:
                    indx = ab1[0,0]
                
                indx = np.array(indx)
                
                if np.all(indx == 0):
                    I2C = I3A.copy()
                elif indx.size==1:
                    I2C = I2A.copy() == indx
                elif indx.size == 0:
                    I2C = I3A.copy()
                else:
                    I2C = np.logical_or((I2A==indx[0]), I2A==indx[1])

                I2C2 = I2A.copy()

                for ind in indx:
                    I2C2[I2C2==ind]=0
                
                I2D = np.subtract(I2C, I3A)
                I2D[I2D!=0] = 1
                r1 = np.sum(I2D)/np.sum(I2C)

                if r1 <= 0.21:
                    I2E = I2D + I2C
                    I2E = (I2E>0).astype(np.uint8)
                elif r1 == 0:
                    I2E=I2C
                elif r1 > 0.21:
                    I2E = I2D+I3A
                    I2E = (I2E>0).astype(np.uint8)
                
                I44 = np.zeros_like(I2)
                I44[r_slice, c_slice] = I2E
                pix = np.nonzero(I44!=0)
                I2[pix] = np.max(I2)+1
                MATC[0][i] = I2.copy()
    result = [im for im in MATC[0][:-1]]
    return np.array(result, dtype = np.uint16)


def trackYeasts(ims):
    print("--TRACKING: LOOP 1/4")
    IS1 = ims[0,:,:]
    IblankG = np.zeros(IS1.shape, dtype="uint16") # allocate
    masks = np.zeros((IS1.shape[0], IS1.shape[1], ims.shape[0])) # allocate
    masks[:,:,0]= IS1   

    for it0 in tqdm(range(1,ims.shape[0])):
        m0b = ims[it0]
        IS2 = m0b.copy()
        IS2C = IS2 
        IS1B = IS1 
        IS1B= (IS1 > 0.5).astype(np.uint16) # binarize # plt.imshow(IS1B)
        
        IS3 = np.multiply(IS1B,IS2) # plt.imshow(IS3)
        
        tr_cells=(np.unique(IS1.astype(np.int0)))
        tr_cells=tr_cells[1:len(tr_cells)]
        gap_cells=(np.unique(IblankG.astype(np.int0)))
        gap_cells=gap_cells[1:len(gap_cells)]
        
        # gap_cells=np.array((10,122)).astype(np.uint64) 
        # cells_tr =  cells_tr + gap_cells
        if gap_cells.size==0 :
            cells_tr = tr_cells;
        else:
        # cells_tr = np.array([tr_cells, gap_cells]) # !
        # cells_tr =  (tr_cells) + (gap_cells)
            cells_tr = np.concatenate((tr_cells,gap_cells),axis=0) # right concatenation
            cells_tr = np.sort(cells_tr)
            
        Iblank0=np.zeros(IS1.shape, dtype="uint16") # plt.imshow(Iblank0)
    
        for it1 in cells_tr: # ascending, default
            # it1=1
            IS5=(IS1==it1) # plt.imshow(IS5) # sum(sum(IS5))
            IS5=IS5.astype(np.uint16) # plt.imshow(IS5)
            IS56 = thin(IS5,1) # plt.imshow(IS56)  #  sum(sum(IS56))
            IS6A = np.multiply(IS56,IS3) # plt.imshow(IS6A)  #   sum(sum(IS6A))
        
            if sum(sum(IS5))==0 :
                IS5=(IblankG==it1)
                IS6A = np.multiply(IS56,IS2C)
                IblankG[IblankG==it1]=0
                # IblankG=np.where(IblankG==it1,IblankG,0)
            
            if sum(sum(IS6A))>0 :
                IS2ind=(statistics.mode(IS6A[np.nonzero(IS6A)])) # nonzero gives the indexes
                Iblank0[IS2==IS2ind]=it1
                IS3[IS3==IS2ind]=0;
                IS2C[IS2==IS2ind]=0;
                


        bl_cells = np.unique(Iblank0) # plt.imshow(Iblank0)
        bl_cells = bl_cells[1:len(bl_cells)]
        seg_gap = np.setdiff1d(tr_cells,bl_cells) 
        
                
        if seg_gap.size>0 :
            for itG in seg_gap:
                IblankG[IS1==itG]=itG;
        
        
        Iblank0B = np.copy(Iblank0)      # plt.imshow(Iblank0)
        Iblank0B[np.nonzero(Iblank0B)] = 1;  
        Iblank0B = (Iblank0B < 0.5).astype(np.uint16) # plt.imshow(Iblank0B) # invert Iblank0B
        
        ISB = np.multiply(IS2,Iblank0B) # plt.imshow(ISB)
        
        newcells=np.unique(ISB)
        newcells = newcells[1:len(newcells)]
        Iblank=Iblank0  # plt.imshow(Iblank)
        A=1;
        
        if newcells.size>0 :
            for it2 in newcells :
                Iblank[IS2==it2]=max(cells_tr)+A; 
                A=A+1;      
                
        masks[:,:,it0]=Iblank # plt.imshow(masks[:,:,it0-1])
        IS1=masks[:,:,it0] # plt.imshow(IS1)


    ccell2=np.unique(masks[:,:,len(masks[0][0])-1]);
    ccell2 = ccell2[1:len(ccell2)] # remove zero since its background
    Mask2 = np.zeros((masks.shape[0], masks.shape[1], len(masks[0][0])))

    print("TRACKING: LOOP 2/4")
    for itt4 in tqdm(range(0,len(masks[0][0]))) : # masks
        mask3=np.zeros((masks.shape[0], masks.shape[1]))
                
                
        for itt3 in range(0,len(ccell2)) : # % itt3=23
            pix2 = np.nonzero(masks[:,:,itt4]==ccell2[itt3])
            mask3[pix2]=itt3 + 1 # zero correction    
                
        Mask2[:,:,itt4] = mask3 

    no_obj=len(ccell2);
    all_ob=np.zeros((no_obj,len(Mask2[0][0])))
    print("TRACKING: LOOP 3/4")
    for ccell in tqdm(range(0,no_obj)): # ccell=0
            
            for itt in range(0,len(masks[0][0])): # itt = 0
                Mas=Mask2[:,:,itt]==ccell + 1 # % imagesc(Mask2)
                pix=sum(sum(Mas))
                all_ob[ccell,itt]=pix #  figure;imagesc(all_ob)

    cell_exists=np.zeros((1,len(all_ob[:,0]))).astype("uint16")
    print("TRACKING: LOOP 4/4")
    for itt2 in tqdm(range(0,len(all_ob[:,0]))) : # itt2 = 20
        etime = sum(all_ob[itt2,:]==0) #
        if etime == 0 :
            cell_exists[0,itt2]=1
        else:
            cell_exists[0,itt2]=np.max(np.nonzero(all_ob[itt2,:]==0)) + 1

    Mask2 = np.array([Mask2[:,:,i] for i in range(Mask2.shape[-1])], dtype = np.uint16)

    return Mask2



    

