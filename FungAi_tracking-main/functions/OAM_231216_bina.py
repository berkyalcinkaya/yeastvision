# -*- coding: utf-8 -*-
"""
Created on Thu May  9 23:41:54 2024

@author: samarth
"""
import numpy as np
import matplotlib.pyplot as plt
import imageio

def binar(IS1):
    # Copy IS1 to IS1B
    IS1B = np.copy(IS1)
    
    # Convert non-zero elements to 1
    IS1B[IS1 != 0] = 1

    
    return IS1B

def plot_image(image, title):
    plt.figure()
    plt.imshow(image)  # Use the gray colormap for grayscale images
    plt.title(title)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()


def main():
    # Load an image (should be in grayscale format)
    image_path = '../I2A.tif'  # Specify the path to your image
    image = imageio.imread(image_path)

    binary_image = binar(image)

    # Plot the original and cleaned images
    plot_image(image, 'Original Image')
    plot_image(binary_image, 'Binarized Image')

    # Save the cleaned image
    output_path = 'binary_image.tif'  # Specify the output path
    imageio.imwrite(output_path, binary_image)
    print(f'Binarized image saved to {output_path}')

if __name__ == '__main__':
    main()