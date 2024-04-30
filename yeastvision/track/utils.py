from skimage.measure import regionprops
import numpy as np
import cv2

def normalize_dict_by_sum(data):
    total = sum(data.values())  # Use the built-in sum() function for efficiency
    if total == 0:
        return {k: 0 for k in data}  # Avoid division by zero by returning zero for all keys
    return {k: v / total for k, v in data.items()}  # Use a dictionary comprehension for normalization


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

def avg_cell_size(Matmasks,nu,no_obj):
    avg = [[[] for _ in range(nu)] for _ in range(no_obj.astype(int)+1)]
    avg_cell = [0]*no_obj.astype(int)

    for iv in range(1,no_obj.astype(int)+1):
        for its in range(nu):
            avg[iv-1][its] = np.average(Matmasks[0][its]==iv)
        
    for iv in range(1,no_obj.astype(int)+1):
        count = 0
        for its in range(nu):
            if avg[iv-1][its] != 0:
                count = count + 1
        avg_cell[iv-1] = np.sum(avg[iv-1])/count
        
    return np.min(avg_cell), np.average(avg_cell), np.max(avg_cell)

def get_wind_coord1(I3, cell_margin=20):
    
    # Get the dimensions of the input image
    x_size, y_size = I3.shape
    # If the input image contains non-zero pixels, find the bounding box of the wind object
    if np.sum(I3) > 0:
        # Threshold the image to create a binary image
        _, binary_image = cv2.threshold(I3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find the contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Loop over each contour to compute the centroid and bounding box
        s_f = []
        for contour in contours:
            # Compute the centroid
            M = cv2.moments(contour)
            cx = int(M['m10'] / (M['m00'] + 1e-6))  # Add small epsilon to avoid division by zero
            cy = int(M['m01'] / (M['m00'] + 1e-6)) 
            # Get the bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)
            # Save the centroid and bounding box information
            s_f.append({'Centroid': (cx, cy), 'BoundingBox': (x, y, w, h)})
        # If there is at least one wind object, get the bounding box coordinates and margins
        if s_f:
            bbox = np.round(s_f[0]['BoundingBox'])
            lower_x_limit = max(1, bbox[0]-cell_margin)
            upper_x_limit = min(y_size, bbox[0]+bbox[2]+cell_margin)
            lower_y_limit = max(1, bbox[1]-cell_margin)
            upper_y_limit = min(x_size, bbox[1]+bbox[3]+cell_margin)
            if upper_x_limit >= I3.shape[1]:
                upper_x_limit = I3.shape[1]-1
            if upper_y_limit >= I3.shape[0]:
                upper_y_limit = I3.shape[0]-1
            x_cn = range(lower_x_limit, upper_x_limit+1)
            y_cn = range(lower_y_limit, upper_y_limit+1)
        # If no wind object is found, raise an error
        else:
            print('empty or multiple object given - error')
    # If the input image is all zero pixels, use the entire image as the bounding box
    else:
        x_cn = np.arange(1, y_size+1)
        y_cn = np.arange(1, x_size+1)

    return x_cn, y_cn