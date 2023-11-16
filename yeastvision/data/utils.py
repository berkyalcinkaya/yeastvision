import os
import glob
import numpy as np

image_extensions = ['.tif','.tiff',
                            '.jpg','.jpeg','.png','.bmp',
                            '.pbm','.pgm','.ppm','.pxm','.pnm','.jp2',
                            '.TIF','.TIFF',
                            '.JPG','.JPEG','.PNG','.BMP',
                            '.PBM','.PGM','.PPM','.PXM','.PNM','.JP2']

def get_file_name(path):
    _,path = os.path.split(path)
    return path

def get_files(dir, extension):
    return sorted(glob.glob(os.path.join(dir, "*"+extension)))

def has_extension(dir, extension):
    return bool(get_files(dir, extension))

def get_im_mask_npzs(dir):
    npz_files = get_files(dir, ".npz")
    ims, masks = [], []
    if npz_files:
        for file in npz_files:
            if is_mask_npz(file):
                masks.append(file)
            elif is_image_npz(file):
                ims.append(file)
    return ims,masks

def is_mask_npz(file):
    return "labels" in np.load(file).files

def is_image_npz(file):
    return "ims" in np.load(file).files


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
    print("\t \t path", ext)
    return ext

def is_image(path):
    print("\t determining if", path, "is an image file")
    return get_filetype(path) in image_extensions

def all_files(dir):
    return glob.glob(os.path.join(dir, "*"))


def get_image_files(dir, remove_masks = False):
    files = all_files(dir)
    all_ims = [im for im in files if is_image(im)]
    if remove_masks:
        return [im for im in all_ims if "masks" not in im]
    else:
        return all_ims

def concatenate_arrays_by_dict(main_dict, new_dict):
    for key in main_dict.keys():
        new_data = new_dict[key]
        main_data = main_dict[key]
        print(main_data.shape, new_data.shape)
        main_dict[key] = np.concatenate((main_data, new_data), axis = 0)
    return main_dict


def append_to_npz(npz_file, new_data):
    npz_dict = dict(np.load(npz_file))
    np.savez(npz_file, **concatenate_arrays_by_dict(npz_dict, new_data))