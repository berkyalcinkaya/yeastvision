'''
Berk Yalcinkaya
8/29/30

A script track a full-lifecycle movie. Utilizes a novel tracking algorithm to track yeast colonies with sporulating and mating cells. 
Takes a directory as input and assumes that tetrad and mating segmentations are present for their correct intervals.
'''
from yeastvision.track.fiest.track import track_full_lifecycle
import os
from os.path import join
from glob import glob
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MASK_KEYWORD = "_masks"
ART_KEYWORD = "_ART_"
TET_KEYWORD = "_TET_"
MAT_KEYWORD = "_MAT16_18"

def sort_masks(mask_files):
    arts, mats, tets = [], [], []
    data_dict = {ART_KEYWORD: arts, MAT_KEYWORD: mats, TET_KEYWORD: tets}
    for mask_file in mask_files:
        for mask_type in data_dict:
            if mask_type in os.path.split(mask_file)[-1]:
                data_dict[mask_type].append(mask_file)
    return arts, mats, tets


def load_and_pad_timeseries(template, start, stop, files, do_resize=True):
    masks = np.zeros_like(template)
    
    if do_resize:
        new_shape = masks.shape[1:]
    
    file_counter = 0
    num_missing = 0
    for i in range(len(template)):
        if i>=start and i<stop:
            file_idx = i-start-num_missing
            if get_time_point_from_image(files[file_counter]) == i: # zeros are kept otherwise
                to_insert = imread(files[file_idx])
                
                if do_resize:
                    to_insert = resize(to_insert, new_shape)
                masks[i] = to_insert
                file_counter+=1
            else:
                num_missing+=1
    return masks
    

def get_time_point_from_image(fname, split_idx=1, split_delim="_"):
    return int((os.path.split(fname)[-1]).split(split_delim)[split_idx])

                
def main(dir):
    ext = os.listdir(dir)[0].split(".")[-1]
    all_files = sorted(glob(join(dir, f"*.{ext}")))
    all_masks, ims = [], []
    
    for file in all_files:
        if MASK_KEYWORD in os.path.split(file)[-1]:
            all_masks.append(file)
        else:
            ims.append(file)
    
    art_files, mat_files, tet_files = sort_masks(all_masks)

    tet_start, tet_stop = get_time_point_from_image(tet_files[0]), get_time_point_from_image(tet_files[-1])+1
    mat_start, mat_stop = get_time_point_from_image(mat_files[0]), get_time_point_from_image(mat_files[-1])+1
    
    arts = np.array([imread(im) for im in art_files])
    
    logging.info(f"Movie length: {len(ims)} ({arts.shape})")
    logging.info(f"Artilife: {len(arts)} ims")
    logging.info(f"Mating: {mat_start} - {mat_stop}, {len(mat_files)} images")
    logging.info(f"Tetrads: {tet_start} - {tet_stop}, {len(tet_files)} images")
    
    tets = load_and_pad_timeseries(arts, tet_start, tet_stop, tet_files)
    mats = load_and_pad_timeseries(arts, mat_start, mat_stop, mat_files)

    

    track_full_lifecycle(arts, mats, tets, [tet_start, tet_stop], [mat_start, mat_stop], len(ims), [0,0])

if __name__ == "__main__":
    logging.info("----FULL LIFECYCLE TRACKING (for pre-generated cell, mating, and tetrad masks)")
    parser = argparse.ArgumentParser(description="Track the full lifecycle of yeast colonies with sporulating and mating cells.")
    parser.add_argument('dir', type=str, help="Directory containing the movie files and segmentation masks.")
    args = parser.parse_args()    
    dir = args.dir
    if not os.path.exists(dir) or not os.path.isdir(dir):
        raise ValueError(f"{dir} does not exist or is not a directory")
    logging.info(f"Utilizing dir {dir} for masks")
    main(dir)
