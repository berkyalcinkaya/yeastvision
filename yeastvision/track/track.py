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



    

