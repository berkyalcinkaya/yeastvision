from skimage.io import imread, imsave
import os
import glob

def stacks_in_dir(dir)->bool:
    files = get_files_in_dir(dir)
    return check_im_dimensions(files[0])>2

def check_im_dimensions(path):
    return len(imread(path).shape)

def get_stacks_in_dir_info(dir):
    files = get_files_in_dir(dir)
    num_channels = check_im_dimensions(files[0])

    return num_channels, files



def get_files_in_dir(dir, extension = None):
    if extension:
        suffix = extension
    else:
        suffix = "*"
    
    return  glob.glob(os.path.join(dir,"*"))
