import numpy as np
from scipy.ndimage import zoom
import numpy as np
from skimage.morphology import disk, binary_erosion, binary_dilation
import torch
from skimage.measure import label
import cv2
import skimage
import matplotlib.pyplot as plt
from cv2 import resize
import logging
import pathlib
import sys
import os
from PIL import Image
import pandas as pd
import math

YV_DIR = pathlib.Path.home().joinpath(".yeastvision")

def divide_and_round_up(input_tuple: tuple) -> tuple:
    """
    Divides each element of the input tuple by 2 and rounds up.

    Parameters:
    input_tuple (tuple): A tuple containing two integers.

    Returns:
    tuple: A tuple with each element divided by 2 and rounded up.
    """
    return (math.ceil(input_tuple[0] / 2), math.ceil(input_tuple[1] / 2))


def resize_image_OAM(image, target_shape):
    zoom_factors = [n / float(o) for n, o in zip(target_shape, image.shape)]
    return zoom(image, zoom_factors, order=0)

def resize_image_scipy(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Resizes an image by a decimal factor using scipy.ndimage.zoom.

    Parameters:
    image (np.ndarray): The input image to resize.
    factor (float): The scaling factor. Values less than 1 reduce the image size, 
                    while values greater than 1 increase the size.

    Returns:
    np.ndarray: The resized image.
    """
    if factor <= 0:
        raise ValueError("Factor must be greater than 0.")
    
    # Use zoom to resize the image
    resized_image = zoom(image, zoom=factor, order=3)  # order=3 for cubic interpolation
    
    return resized_image

def get_longest_interval(intervals):
        longest = (0, 0)
        max_length = 0
        for interval in intervals:
            start, end = map(int, interval.split('-'))
            length = end - start  # End is exclusive
            if length > max_length:
                max_length = length
                longest = (start, end)
        return longest

def nonzero_intervals(images, max_blanks = 1):
    """
    Finds all non-zero image intervals allowing for up to max_blanks blank images within the sequence.
    
    :param images: A NumPy array of shape (n, r, c) where n is the number of images
    :param max_blanks: Maximum number of blank images allowed within the sequence
    :return: List of strings representing intervals in the format "{start}-{end},
    where end is exlusive and start is inclusive"
    """
    num_images = images.shape[0]
    non_empty = np.any(images > 0, axis=(1, 2))
    intervals = []
    df = pd.DataFrame({"Value": non_empty})
    df['tag'] = df['Value'] > 0
    fst = df.index[df['tag'] & ~ df['tag'].shift(1).fillna(False)]
    lst = df.index[df['tag'] & ~ df['tag'].shift(-1).fillna(False)]
    pr = [(i, j) for i, j in zip(fst, lst)]
    for i, j in pr:
        intervals.append(f"{i}-{j+1}")
    return intervals

def cleanup_npz_files_from_sample_movie(script_file):
    # Get the current script path
    current_script_path = os.path.abspath(script_file)
    
    # Go back one directory
    parent_dir = os.path.dirname(os.path.dirname(current_script_path))
    
    # Define the target directory
    target_dir = os.path.join(parent_dir, 'data', 'sample_movie_1')
    
    # Check if the target directory exists
    if not os.path.exists(target_dir):
        print(f"The directory {target_dir} does not exist.")
        return

    # Initialize a list to store the names of deleted files
    deleted_files = []

    # Iterate over files in the target directory
    for filename in os.listdir(target_dir):
        if filename.endswith('.npz'):
            file_path = os.path.join(target_dir, filename)
            try:
                os.remove(file_path)
                deleted_files.append(filename)
            except Exception as e:
                print(f"Failed to delete {filename}: {e}")

    # Print the names of the deleted files
    if deleted_files:
        print("Deleted files:")
        for file in deleted_files:
            print(file)
    else:
        print("No .npz files found to delete.")

def logger_setup():
    YV_DIR.mkdir(exist_ok=True)
    log_file = YV_DIR.joinpath("run.log")
    try:
        log_file.unlink()
    except:
        print("creating new log file")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file),
                  logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(__name__)
    logger.info(f"WRITING LOG OUTPUT TO {log_file}")
    return logger


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

def write_images_to_dir(path, ims, flags = None, annotation = None, extension = ".tif", ):
    dir = os.path.dirname(path)
    if os.path.exists(path):
        return
    os.mkdir(path)
    if "." not in extension:
        extension = "." + extension
    num_digits = len(str(len(ims)))
    for i, im in enumerate(ims):
        base_name = f"{str(i+1).zfill(num_digits)}"
        if flags is not None and flags[i] and annotation is not None:
            base_name += f"_{annotation}"
        new_file_name = f"{base_name}{os.path.splitext(im)[1]}"
        new_file_path = os.path.join(path, new_file_name)
        skimage.io.imsave(new_file_path, im)

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

def check_gpu(do_torch = True):
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

def rescaleBySize(newshape, ims, interpolation=cv2.INTER_CUBIC):
    row, col = newshape
    return [resize(im, (col,row), interpolation=interpolation) for im in ims]

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

def count_objects(labeledMask):
    return len(np.unique(labeledMask))-1

def capitalize_first_letter(string):
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