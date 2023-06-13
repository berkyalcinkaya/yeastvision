from skimage.measure import regionprops
import numpy as np

def normalize_dict_by_sum(dict):
    sum = 0
    for val in dict.values():
        sum+=val
    new_dict = {k:v/sum for k,v in dict.items()}

    return new_dict

def get_cell_bbox(array, cellval, desired_shape):
    # Get the center coordinates
    centroid = regionprops((array==cellval).astype(np.uint8))[0].centroid
    r,c = centroid
    r,c = round(r), round(c)
    center_row, center_col = r,c

    # Get the desired shape dimensions
    desired_height, desired_width = desired_shape

    # Calculate the starting and ending row indices
    start_row = center_row - desired_height // 2
    end_row = start_row + desired_height

    # Calculate the starting and ending column indices
    start_col = center_col - desired_width // 2
    end_col = start_col + desired_width

    # Check if the indices are out of bounds
    if start_row < 0:
        start_row = 0
        end_row = min(start_row + desired_height, array.shape[0])
    elif end_row > array.shape[0]:
        end_row = array.shape[0]
        start_row = max(end_row - desired_height, 0)

    if start_col < 0:
        start_col = 0
        end_col = min(start_col + desired_width, array.shape[1])
    elif end_col > array.shape[1]:
        end_col = array.shape[1]
        start_col = max(end_col - desired_width, 0)

    # Get the actual cropped region from the array
    cropped_array = array[start_row:end_row, start_col:end_col]

    return cropped_array

def get_bbox_coords(bool_im, cell_margin = 20):
    centroid = regionprops(bool_im.astype(np.uint8))[0].centroid
    r_size, c_size = bool_im.shape
    r,c = centroid
    r,c = round(r), round(c)
    bbox = slice(max([0, r-cell_margin]), min([r_size, r+cell_margin])), slice(max([0, c-cell_margin]), min([c_size, c+cell_margin]))
    return bbox

def normalize_im(im, clip = True):
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
    assert not np.any(np.isnan(im))
    if clip:
        im[im<0] = 0
    if im.max()==0:
        return im.astype(np.float32)
    im_norm = (im - im.min()) / (im.max() - im.min())
    return im_norm.astype(np.float32)