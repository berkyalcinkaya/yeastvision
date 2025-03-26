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
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import time

MASK_KEYWORD = "_masks"
ART_KEYWORD = "_ART_"
TET_KEYWORD = "_TET_"
MAT_KEYWORD = "_MAT16"

def sort_masks(mask_files):
    arts, mats, tets = [], [], []
    data_dict = {ART_KEYWORD: arts, MAT_KEYWORD: mats, TET_KEYWORD: tets}
    for mask_file in mask_files:
        for mask_type in data_dict:
            if mask_type in os.path.split(mask_file)[-1]:
                data_dict[mask_type].append(mask_file)
    return arts, mats, tets


def load_and_pad_timeseries(t, start, stop, files, do_resize=True):
    im_shape = imread(files[0]).shape
    masks = np.zeros((t, im_shape[0], im_shape[1]))
    
    if do_resize:
        new_shape = masks.shape[1:]
    
    file_counter = 0
    num_missing = 0
    for i in range(t):
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

                
def main(dir, info_only):
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
    
    tet_shape = imread(tet_files[0]).shape
    mat_shape = imread(mat_files[0]).shape
    t = len(ims)
    
    logging.info(f"Movie length: {t} | Shape: {arts.shape}")
    logging.info(f"Artilife: {len(arts)} ims")
    logging.info(f"Mating (start - stop): {mat_start} - {mat_stop}, {len(mat_files)} x {mat_shape[0]} x {mat_shape[1]}")
    logging.info(f"Tetrads (start - stop): {tet_start} - {tet_stop}, {len(tet_files)} x {tet_shape[0]} x {tet_shape[1]}")
    
    if info_only:
        return
    
    tets = load_and_pad_timeseries(t, tet_start, tet_stop, tet_files, do_resize=False)
    mats = load_and_pad_timeseries(t, mat_start, mat_stop, mat_files, do_resize=False)
    track_full_lifecycle(arts, mats, tets, [tet_start, tet_stop], [mat_start, mat_stop], len(ims), None)

if __name__ == "__main__":
    # start timeer


    logging.info("----FULL LIFECYCLE TRACKING (for pre-generated cell, mating, and tetrad masks)")
    parser = argparse.ArgumentParser(description="Track the full lifecycle of yeast colonies with sporulating and mating cells.")
    parser.add_argument('dir', type=str, help="Directory containing the movie files and segmentation masks.")
    parser.add_argument('--mat_keyword', required=False, type=str, default="_MAT16")
    parser.add_argument('--info_only', action="store_true")
    args = parser.parse_args()    
    dir = args.dir
    MAT_KEYWORD = args.mat_keyword
    if not os.path.exists(dir) or not os.path.isdir(dir):
        raise ValueError(f"{dir} does not exist or is not a directory")
    logging.info(f"Utilizing dir {dir} for masks")
    start = time.time()
    main(dir, args.info_only)
    end = time.time()
    logging.info(f"Total time: {end-start}")
    logging.info("----END FULL LIFECYCLE TRACKING----")
