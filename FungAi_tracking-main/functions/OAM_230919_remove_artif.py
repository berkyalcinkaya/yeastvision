import numpy as np
import imageio
import matplotlib.pyplot as plt
from skimage.morphology import disk, binary_opening, dilation
from skimage import img_as_uint

def remove_artif(I2A):
    # Applying logical operation and morphological opening
    I2B = binary_opening(I2A > 0, disk(6))  # Adjust the disk size as needed
    # Morphological dilation
    I2C = dilation(I2B, disk(6))  # Adjust the disk size as needed
    # Element-wise multiplication of I2A with I2C
    I3 = I2A * I2C

    # Extract unique objects
    objs = np.unique(I3)

    # Initialize an image of zeros with the same size as I2A
    I4 = np.zeros_like(I2A, dtype=np.uint16)

    # Mapping the original image values where they match the unique objects
    for obj in objs:
        I4[I2A == obj] = obj

    # Returning the final image
    
    return img_as_uint(I4)

def plot_image(image, title):
    plt.figure()
    plt.imshow(image)  # Use the gray colormap for grayscale images
    plt.title(title)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

def main():
    image_path = '../I2A.tif'
    image = imageio.imread(image_path)

    cleaned_image = remove_artif(image)

    # Plot the original and cleaned images
    plot_image(image, 'Original Image')
    plot_image(cleaned_image, 'Cleaned Image')

    # Save the cleaned image
    output_path = 'cleaned_image.tif'
    imageio.imwrite(output_path, cleaned_image)
    print(f'Cleaned image saved to {output_path}')

if __name__ == '__main__':
    main()
