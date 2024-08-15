

# -*- coding: utf-8 -*-
"""
Created on Thu May  9 23:50:34 2024

@author: samar
"""

import numpy as np
import matplotlib.pyplot as plt
#import imageio
from PIL import Image

import numpy as np

def cal_celldata(all_obj, ccel):
    cell_data = np.zeros((ccel, 5))

    for iv in range(ccel):
        first_occurrence = np.argmax(all_obj[iv, :] > 0)
        last_occurrence = len(all_obj[iv, :]) - np.argmax((all_obj[iv, :][::-1] > 0)) - 1
        
        cell_data[iv, 0] = first_occurrence  # 1st occurrence
        cell_data[iv, 1] = last_occurrence   # Last occurrence
        
    for iv in range(ccel):
        cell_data[iv, 2] = cell_data[iv, 1] - cell_data[iv, 0] + 1  # Times the cell appears
        aa1 = all_obj[iv, :]
        aa2 = aa1[int(cell_data[iv, 0]):int(cell_data[iv, 1]) + 1]
        aa3 = np.where(aa2 == 0)[0]
        cell_data[iv, 3] = len(aa3)  # Number of times it disappears between 1st and last occurrence
        cell_data[iv, 4] = (cell_data[iv, 3] * 100) / cell_data[iv, 2]  # Percentage of times the cell disappears

    return cell_data


def plot_image(image, title):
    plt.figure()
    plt.imshow(image)  # Use the gray colormap for grayscale images
    plt.title(title)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

def main():
    # Load an image (should be in grayscale format)
    image_path = '../I2A.tif'  # Specify the path to your image
    image = Image.open(image_path)
    img_array = np.array(image)
    print(img_array)
    # Assuming the image is already in the correct format, if not, convert it

    # Apply the artifact removal function
    op = cal_celldata(img_array, 3);

    print(op)

if __name__ == '__main__':
    main()