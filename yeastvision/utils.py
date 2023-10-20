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
import logging
import pathlib
import sys
import os
from PIL import Image


def colorize_mask(mask, rgba_color):
    """Apply an RGBA color to non-zero pixels in a binary mask.

    Parameters:
    - mask: 2D numpy array representing a binary mask.
    - rgba_color: Tuple of 4 values representing the desired RGBA color.

    Returns:
    - 3D numpy array with shape (height, width, 4) representing an RGBA image.
    """
    # Initialize an empty RGBA image of the same size as the mask
    rgba_image = np.zeros((*mask.shape, 4), dtype=np.uint8)

    # Assign the specified RGBA color to non-zero pixels in the mask
    rgba_image[mask > 0] = rgba_color

    return rgba_image

def overlay_masks_on_image(phase_img, masks, is_contour, alpha=100):
    # Predefined colors with transparency
    colors = [
        [255, 0, 0, alpha],
        [0, 0, 255, alpha],
        [0, 255, 0, alpha],
        [255, 255, 0, alpha],
        [0, 255, 255, alpha],
        [255, 218,0, alpha]
    ]

    # Convert the 2D numpy array to a PIL Image
    phase_image_pil = Image.fromarray(convertGreyToRGB(phase_img)).convert("RGBA")

    for color_idx, mask in enumerate(masks):
        color = colors[color_idx].copy()

        if is_contour[color_idx]:
            color[-1]=255
        # Convert binary mask to a PIL Image
        mask_binary = binarize(mask)*255
        colored_mask = Image.fromarray(colorize_mask(mask_binary, color))
        phase_image_pil.paste(colored_mask, (0, 0), Image.fromarray(mask_binary))

    result_array = np.asarray(phase_image_pil, dtype=np.uint8)
    return result_array

def write_images_to_dir(path, ims, extension = ".tif"):
    dir = os.path.dirname(path)
    if os.path.exists(path):
        return
    os.mkdir(path)
    if "." not in extension:
        extension = "." + extension
    for i, im in enumerate(ims):
        name = f"im_{i}{extension}"
        fname = os.path.join(path, name)
        skimage.io.imsave(fname, im)

def save_image(path, im, extension  = ".tif"):
    pass


def overlay_images(phase_img, mask_img, colorized_mask):
    # Convert numpy arrays to PIL images
    mask = mask_img.copy()
    mask[mask>0] = 255
    microscopy_pil = Image.fromarray(convertGreyToRGB(phase_img))
    mask_pil = Image.fromarray(mask)
    colorized_pil = Image.fromarray(colorized_mask)

    # Convert images to 'RGBA' mode
    microscopy_pil = microscopy_pil.convert("RGBA")
    colorized_pil = colorized_pil.convert("RGBA")
    mask_pil = mask_pil.convert("L")

    # Overlay images
    microscopy_pil.paste(colorized_pil, (0, 0), mask_pil)

    # Convert PIL image back to numpy array
    result = np.array(microscopy_pil)

    return result


def get_frames_with_no_cells(masks):
    no_cell_frames = []
    for i, mask in enumerate(masks):
        if np.all(mask==0):
            no_cell_frames.append(i)
    return no_cell_frames

def get_end_of_cells(masks):
    for i in reversed(range(masks.shape[0])):
        if np.any(masks[i]!=0):
            return i



def get_filename(path):
    _, name_with_extension = os.path.split(path)
    return os.path.splitext(path)[0]

def logger_setup():
    cp_dir = pathlib.Path.home().joinpath('.yeastvision')
    cp_dir.mkdir(exist_ok=True)
    log_file = cp_dir.joinpath('run.log')
    try:
        log_file.unlink()
    except:
        print('creating new log file')
    logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler(sys.stdout)
                    ]
                )
    logger = logging.getLogger(__name__)
    logger.info(f'WRITING LOG OUTPUT TO {log_file}')
    return logger, log_file

def configure_tf_memory_growth():
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


def check_gpu(do_tf = True, do_torch = True):
    if do_tf:
        print(f"Tensorflow\n__________:\n{tf.config.experimental.get_memory_info('GPU:0')}")
    if do_torch:
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        d = {"total": t,
             "reserved": r,
             "allocated": a}
        print(f"Pytorch\n_________\n{d}")
        # free inside reserved

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

def convertGreyToRGB(im):
    image = skimage.util.img_as_ubyte(normalize_im(im))
    image_3D = cv2.merge((image,image,image))
    #return cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
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