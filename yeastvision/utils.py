import numpy as np
from skimage.morphology import disk, binary_erosion, binary_dilation
import torch
import tensorflow as tf
from skimage.measure import label
import cv2
import skimage
import matplotlib.pyplot as plt
import cv2
from cv2 import resize

def showCellNums(mask):
    "annotates the current plt figure with cell numbers"
    #cell_vals = np.unique(mask[mask!=0]).astype(int)
    cell_vals = np.unique(mask).astype(int)
    cell_vals = np.delete(cell_vals, np.where(cell_vals == 0))

    for val in cell_vals:
        # print("cell val: " + str(val)) #test
        x, y = getCenter(mask, val)
        plt.annotate(str(val), xy=(x, y), ha='center', va='center')

def getCenter(mask, cell_val):
    '''
    takes random points within a cell to estimate its center. Used to find the x,y coordinates where the cell number
    annotation text should be displated
    '''
    y, x = (mask == cell_val).nonzero()
    sample = np.random.choice(len(x), size=20, replace=True)
    meanx = np.mean(x[sample])
    meany = np.mean(y[sample])

    return int(round(meanx)), int(round(meany))

def rescaleByFactor(factor, ims):
    row, col = ims[0].shape
    newrow, newcol = int(row*factor), int(col*factor)
    print(newrow, newcol)
    return rescaleBySize((newrow, newcol), ims)

def rescaleBySize(newshape, ims):
    row, col = newshape
    return [resize(im, (col,row), interpolation=cv2.INTER_CUBIC) for im in ims]

def binarize_relabel(mask):
    return label(binarize(mask))

def binarize(mask):
    return (mask > 0).astype(np.uint8)

def check_torch_gpu(gpu_number = 0):
    try:
        device = torch.device('cuda:' + str(gpu_number))
        _ = torch.zeros([1, 2, 3]).to(device)
        print('** TORCH CUDA version installed and working. **')
        return True
    except:
        print('TORCH CUDA version not installed/working.')
        return False

def check_tf_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print("** TENSORFLOW installed and working. **")
    else:
        print("TENSORFLOW not installed/working")
    return bool(gpus)

def count_objects(labeledMask):
    return len(np.unique(labeledMask))-1

def capitalize(string):
    first_letter = string[0]
    rest_of_string = string[1:]
    return first_letter.capitalize()+rest_of_string

def get_mask_contour(mask):
    out_mask = np.zeros_like(mask)
    for cellnum in np.unique(mask):
        if cellnum>0:
            cell_contour = get_cell_contour(mask, cellnum)
            out_mask[cell_contour==1]=cellnum
    return out_mask

def get_cell_contour(mask, cellnum):
    out_mask = np.zeros_like(mask)
    mask_binary = (mask==cellnum).astype(np.uint8)
    in_mask = shrink_bud(mask_binary, kernel_size = 2)
    mask_binary[in_mask]=0
    return mask_binary


def shrink_bud(bud_mask, footprint = None, kernel_size = 2):
    if not footprint:
        footprint = disk(kernel_size)
    return binary_erosion(bud_mask, footprint)


def enlarge_bud(bud_mask, footprint = None, kernel_size = 4):
    if not footprint:
        footprint = disk(kernel_size)
    return binary_dilation(bud_mask, footprint)

def normalize_im(im_o, clip = True):
    """
    Normalizes a given image such that the values range between 0 and 1.     
    
    Parameters
    ---------- 
    im : 2d-array
        Image to be normalized.
    clip: boolean
        Whether or not to set im values <0 to 0
        
    Returns
    -------
    im_norm: 2d-array
        Normalized image ranging from 0.0 to 1.0. Note that this is now
        a floating point image and not an unsigned integer array. 
    """ 
    im = np.nan_to_num(im_o)
    if clip:
        im[im<0] = 0
    if im.max()==0:
        return im.astype(np.float32)
    im_norm = (im - im.min()) / (im.max() - im.min())
    im_norm[np.isnan(im_o)] = np.nan
    return im_norm.astype(np.float32)

def convertGreyToRGBA(im):
    image = skimage.util.img_as_ubyte(normalize_im(im))
    image_3D = cv2.merge((image,image,image))
    return image_3D
    # rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
    # rgba[:, :, 3] = 1
    # return rgba

def get_img_from_fig(fig):
    '''
    input: matplotlib pyplot figure object
    output: a 3D numpy array object corresponding to the image shown by that figure
    '''
    import io as i
    buf = i.BytesIO()
    fig.savefig(buf, format="tif", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return (np.array(img)).astype("uint8")


def overlay(im, mask, color = [255,0,0], get_fig = False, image_only = True):
    '''
    Overlays an image and with its binary mask for easily visualizing
    
    
    Parameters
    --------
    image: grayscale ndarray of shape mxn
    mask: binary ndaraay of shape mxn
    get_fig (bool): return the matplotlib figure as a np array
    color (list): color given to mask in [R,G,B] form
    image_only (bool): return only the merged images as a RGB ndarray 
    '''
    image = skimage.util.img_as_ubyte(normalize_im(im))
    image_3D = cv2.merge((image,image,image))
    mask_plot = image_3D.copy()
    mask_plot[mask==1] = color # set thresholded pixels to red
    
    if image_only:
        return mask_plot
    
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(image_3D)
    plt.imshow(mask_plot)
    plt.show()
    
    if get_fig:
        return get_img_from_fig(fig)
    else:
        return mask_plot