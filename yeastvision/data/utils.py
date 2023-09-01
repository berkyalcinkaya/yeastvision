import os
import glob
import numpy as np

image_extensions = ['.tif','.tiff',
                            '.jpg','.jpeg','.png','.bmp',
                            '.pbm','.pgm','.ppm','.pxm','.pnm','.jp2',
                            '.TIF','.TIFF',
                            '.JPG','.JPEG','.PNG','.BMP',
                            '.PBM','.PGM','.PPM','.PXM','.PNM','.JP2']

def get_extensions_in_dir(dir):
    extensions = []
    for path in os.listdir(dir):
        name, ext = os.path.splitext(path)
        if ext not in extensions:
            extensions.append(ext)
    return extensions

def has_label_npz(dir):
    for path in os.listdir(dir):
        name,ext = os.path.splitext(path)
        if name == "labels" and ext == ".npz":
            return True

def get_filetype(path):
    _, ext = os.path.splitext(path)
    return ext

def is_image(path):
    return get_filetype(path) in image_extensions

def all_files(dir):
    return glob.glob(os.path.join(dir, "*"))


def get_image_files(dir, remove_masks = False):
    files = all_files(dir)
    i = 0
    while not is_image(files[i]):
        i+=1
        if i == len(files):
            print("No images detected")
            return
    ext = get_filetype(files[i])
    all_ims = glob.glob(os.path.join(dir, f"*{ext}"))  
    if remove_masks:
        return [im for im in all_ims if "masks" not in im]
    else:
        return all_ims

def concatenate_arrays_by_dict(main_dict, new_dict):
    for key in main_dict.keys():
        new_data = new_dict[key]
        main_data = main_dict[key]
        main_dict[key] = np.concatenate(main_data, new_data, axis = 0)


def append_to_npz(npz_file, new_data):
    npz_dict = dict(np.load(npz_file))
    np.savez(npz_file, concatenate_arrays_by_dict(npz_dict, new_data))