from typing import Optional, List, Tuple, Union
import torch
import time
import numpy as np
import logging
from .utils import _get_budSeg, _get_matSeg, _get_proSeg, _get_spoSeg, extend_seg_output, fill_empty_arrays, resize_image, save_images_to_directory, synchronize_indices
from .mating import track_mating
from .tetrads import track_tetrads
from .full_lifecycle_utils import track_general_masks
from .correction import correct_mating, correct_proSeg_with_mating, correct_proSeg_with_tetrads
from yeastvision.ims.interpolate import interpolate_intervals, get_interp_labels, deinterpolate
from yeastvision.track.track import track_proliferating
from yeastvision.track.lineage import LineageConstruction
from .FL import FL
from skimage.io import imsave

logger = logging.getLogger(__name__)

def fiest_basic(ims:np.ndarray, interp_intervals:Optional[List[dict]], proSeg_params:Optional[dict], proSeg_weights:Optional[str]=None)->Tuple[np.ndarray[np.uint16], np.ndarray[np.uint8], np.ndarray[np.uint8]]:
    '''
    Implementation 1 of Frame Interpolation Enhanced Single-cell Tracking (FIEST) from the paper 'Deep learning-driven 
    imaging of cell division and cell growth across an entire eukaryotic life cycle' for tracking of timeseries movies of
    yeasts that proliferate asexually. This novel algorithm is based on enhancing the overlap of single 
    cell masks in consecutive images through deep learning video frame interpolation. 
    
    Steps: (1) interpolate with RIFE, (2) segment with proSeg, (3) track, (4) deinterpolate
    
    Args:
        ims (np.ndarray): an (n x h x w) array with n images of h x w dimensions to be interpolated and segmented
        interp_intervals (optional, lists of dict): optional list specifying interpolation levels and intervals, each entry
                                                    of the list should be a dictionary with the keys (start, stop, interp)
                                                    where interp is an integer value in [1,2,3,4]. If none, no interpolation is
                                                    performed
        proSeg_params (optional dict): a dictionary specifying the parameters for segmentation with proSeg, should have the keys 
                                    'mean_diameter,' 'flow_threshold,' and 'cell_probability_threshold.' If none, defaults are used
        proSeg_weights (optional, path): path to weights to use. If none, loads default proSeg weights from the miranda lab. Assumes weights
                                        have been downloaded
                                    
    Returns:
        tuple:
            np.ndarray[uint16] - the tracked masks
            np.ndarray[uint8] - the cell probability masks
            np.ndarray[uint8] - the cell flows masks
    '''
    logger.info("---FIEST Tracking Basic---")
    to_segment = ims
    if interp_intervals:
        logger.info(f"Performing interpolation over intervals:\n {interp_intervals}")
        to_segment, new_intervals = interpolate_intervals(ims, interp_intervals)
        interp_locs = get_interp_labels(len(ims), interp_intervals)

    proSeg, proSeg_params, proSeg_weights = _get_proSeg(proSeg_params, proSeg_weights)
    
    masks, probs, flows = proSeg.run(to_segment, proSeg_params, proSeg_weights)
    masks = track_general_masks(masks, transpose_out=True)
    
    if interp_intervals:
        masks, probs, flows = [deinterpolate(mask_type, interp_locs) for mask_type in [masks, probs, flows]]
    
    torch.cuda.empty_cache()
    del proSeg
    
    return masks, probs, flows
    

def fiest_basic_with_lineage(ims:np.ndarray, interp_intervals:Optional[List[dict]], proSeg_params:Optional[dict], proSeg_weights:Optional[str]=None,
                            budSeg_params:Optional[dict]=None, budSeg_weights:Optional[str]=None)->dict:
    '''
    Implementation 2 of Frame Interpolation Enhanced Single-cell Tracking (FIEST) and lineage reconstruction alogirthm 
    from the paper 'Deep learning-driven imaging of cell division and cell growth across an entire eukaryotic life cycle' 
    for tracking cells and lineages of yeasts that proliferate asexually. This novel algorithm is based on enhancing the overlap of single 
    cell masks and mother-daughter pairs in consecutive images through deep learning video frame interpolation.
    
    Steps: (1) interpolate with RIFE, (2) segment with proSeg, (3) track, (4) construct lineages, (4) deinterpolate
    
    Args:
        ims (np.ndarray): an (n x h x w) array with n images of h x w dimensions to be interpolated and segmented
        interp_intervals (list, optional): optional list specifying interpolation levels and intervals, each entry
                                                    of the list should be a dictionary with the keys (start, stop, interp)
                                                    where interp is an integer value in [1,2,3,4]. If none, no interpolation is
                                                    performed
        proSeg_params (dict, optional): a dictionary specifying the parameters for segmentation with proSeg, should have the keys 
                                    'mean_diameter,' 'flow_threshold,' and 'cell_probability_threshold.' If none, defaults are used
        proSeg_weights (str, optional): path to weights to use. If none, loads default proSeg weights from the miranda lab. Assumes weights
                                        have been downloaded
        budSeg_params (dict, optional): budSeg parameters for budding cell segmentation. See proSeg_params
        budSeg_weights (str, optional): path to weights for budSeg predictions, see proSeg_weights. 
                                    
    Returns:
        dict: a dictionary with the segmentation outputs and lineage data, containing the following keys:
            'cells' (tuple): the output of the cytoplasm (proSeg) model with the following items
                np.ndarray[uint16] - the tracked masks
                np.ndarray[uint8] - the cell probability masks
                np.ndarray[uint8] - the cell flows masks
            'buds' (tuple): the output of the bud segmentation used for lineage constructions:
                np.ndarray[uint16] - the labeled buds
                np.ndarray[uint8] - the bud probability masks
                np.ndarray[uint8] - the bud flows masks
            'lineages' (tuple): the lineage information:
                np.ndarray[uint16] - cell daughters array
                pd.DataFrame - a DataFrame containing cell life data with columns 'birth', 'death', 'mother', and 'confidence'.
    '''
    logger.info("---FIEST Tracking With Lineage Reconstruction---")
    to_segment = ims
    if interp_intervals:
        logger.info(f"Performing interpolation over intervals:\n {interp_intervals}")
        to_segment, new_intervals = interpolate_intervals(ims, interp_intervals)
        interp_locs = get_interp_labels(len(ims), interp_intervals)

    proSeg, proSeg_params, proSeg_weights = _get_proSeg(proSeg_params, proSeg_weights)
    budSeg, budSeg_params, budSeg_weights = _get_budSeg(budSeg_params, budSeg_weights)
    
    masks, probs, flows = proSeg.run(to_segment, proSeg_params, proSeg_weights)
    masks = track_general_masks(masks, transpose_out=True)

    
    buds, bud_probs, bud_flows = budSeg.run(to_segment, budSeg_params, budSeg_weights)
    
    lineage = LineageConstruction(masks, buds, forwardskip=3)
    daughters, mothers = lineage.computeLineages()
    
    if interp_intervals:
        masks, probs, flows = [deinterpolate(mask_type, interp_locs) for mask_type in [masks, probs, flows]]
        buds, bud_probs, bud_flows = [deinterpolate(mask_type, interp_locs) for mask_type in [buds, bud_probs, bud_flows]]
    
    torch.cuda.empty_cache()
    del proSeg
    del budSeg
    
    return {"cells": (masks, probs, flows), 
            "buds": (buds, bud_probs, bud_flows),
            "lineages": (daughters, mothers)
    }


def fiest_full_lifecycle(ims: np.ndarray, interp_intervals:Optional[List[dict]]=None, proSeg_params:Optional[dict]=None, proSeg_weights:Optional[str]=None, 
                         matSeg_params:Optional[dict]=None, matSeg_weights:Optional[str]=None, spoSeg_params: Optional[dict]=None, spoSeg_weights:Optional[str]=None,
                         mat_start:Optional[int]=0, mat_stop:Optional[int]=None, spo_start:Optional[int]=0, spo_stop:Optional[int]=None, shock_period:Optional[List[int]]=None):
    '''
    Implementation of Frame Interpolation Enhanced Single-cell Tracking (FIEST) from the paper 'Deep learning-driven 
    imaging of cell division and cell growth across an entire eukaryotic life cycle' for tracking of timeseries movies of
    yeasts that mate, sporulate, and proliferate asexually. This novel algorithm is based on enhancing the overlap of single 
    cell masks in consecutive images through deep learning video frame interpolation. 
    
    Steps: 
    (1) segment ims with matSeg, proSeg, and spoSeg
    (2) track using a novel full-lifecycle track method based frame overlap and corrections that leverage all three outputs
    (3) de-interpolate all outputs
    
    Args:
        ims (np.ndarray): an (n x h x w) array with n images of h x w dimensions to be interpolated and segmented
        interp_intervals (list, optional): optional list specifying interpolation levels and intervals, each entry
                                                    of the list should be a dictionary with the keys (start, stop, interp)
                                                    where interp is an integer value in [1,2,3,4]. If none, no interpolation is
                                                    performed
        proSeg_params (dict, optional): a dictionary specifying the parameters for segmentation with proSeg, should have the keys 
                                    'mean_diameter,' 'flow_threshold,' and 'cell_probability_threshold.' If none, defaults are used
        proSeg_weights (str, optional): path to weights to use. If none, loads default proSeg weights from the Miranda lab. Assumes weights
                                        have been downloaded
        matSeg_params (dict, optional): matSeg parameters for mating cell segmentation. See proSeg_params
        matSeg_weights (str, optional): path to weights for matSeg predictions, see proSeg_weights.  
        spoSeg_params (dict, optional): spoSeg parameters for tetrad segmentation. See proSeg_params
        spoSeg_weights (str, optional): path to weights for spoSeg predictions, see proSeg_weights.  
        mat_start (int, optional): the first index (0-indexed) of mating cells. Default is 0
        mat_stop (int, optional): the last index (exclusive, 0-indexed) of mating cells in ims. If none, then mat_stop defaults to len(ims).
        spo_start (int, optional): the first index (0-indexed) of sporulating cells (tetrads) in ims. Default is 0.
        spo_stop (int, optional): the last index (exclusive, 0-indexed) of sporulating cells (tetrads) in ims. If none, defaults to len(ims).
        shock_period (list of integers, optional): a two number list specifying to python indeces (0-indexed, exclusive) of the shock_period in the experiments.
                                                    Leave as none if no shock_period applies to the movie.
        
        
                                    
    Returns:
        dict: a dictionary with tracked outputs, probability, and flow images. Please noe that probability and flow images have not been corrected.  
        This dictionary contains the following keys:
            'cells' (tuple): output of proSeg segmentation and tracking
                np.ndarray[uint16] - the tracked general (proSeg) masks with mating events removed
                np.ndarray[uint8] - the raw proSeg cell probability masks
                np.ndarray[uint8] - the raw proSeg cell flows masks 
            'mating' (tuple): output of matSeg segmentation and tracking
                np.ndarray[uint16] - the mating tracked masks after correction
                np.ndarray[uint8] - the raw matSeg mating probability masks
                np.ndarray[uint8] - the raw matSeg mating flows masks
            'spores' (tuple): output of spoSeg segmentation and tracking:
                np.ndarray[uint16] - the tracked tetrad masks after correction
                np.ndarray[uint8] - the raw spoSeg tetrad probability masks
                np.ndarray[uint8] - the raw spoSeg tetrad flows masks
    '''

    logger.info("---FIEST Full Lifecycle Tracking---")
    to_segment = ims
    if interp_intervals:
        logger.info(f"Performing interpolation over intervals:\n {interp_intervals}")
        to_segment, new_intervals = interpolate_intervals(ims, interp_intervals)
        interp_locs = get_interp_labels(len(ims), interp_intervals)

    
    #logger.info("Fetching models and weight files")
    proSeg, proSeg_params, proSeg_weights = _get_proSeg(proSeg_params, proSeg_weights)
    spoSeg, spoSeg_params, spoSeg_weights = _get_spoSeg(spoSeg_params, spoSeg_weights)
    matSeg, matSeg_params, matSeg_weights = _get_matSeg(matSeg_params, matSeg_weights)

    #logger.info("")
    masks, probs, flows = proSeg.run(to_segment, proSeg_params, proSeg_weights)

    if mat_stop is None:
        mat_stop = len(ims)
    if spo_stop is None:
        spo_stop = len(ims)
    
    #logger.info(f"Segment with matseg from")
    mat_out = matSeg.run(to_segment[mat_start:mat_stop], matSeg_params, matSeg_weights)
    mat_out_correct_len = extend_seg_output(to_segment, mat_out, mat_start, mat_stop)
    
    #logger.info()
    spo_out = spoSeg.run(to_segment[spo_start:spo_stop], spoSeg_params, spoSeg_weights)
    spo_out_correct_len = extend_seg_output(to_segment, spo_out, spo_start, spo_stop)

    proSeg_tracked, matSeg_tracked, spoSeg_tracked = track_full_lifecycle(masks, mat_out_correct_len[0], spo_out_correct_len[0],
                                                                          [spo_start, spo_stop], [mat_start, mat_stop], len(to_segment), 
                                                                          shock_period=shock_period)

    if interp_intervals:
        proSeg_tracked, probs, flows = [deinterpolate(mask_type, interp_locs) for mask_type in [proSeg_tracked, probs, flows]]
        matSeg_tracked, mat_probs, mat_flows = [deinterpolate(mask_type, interp_locs) for mask_type in [matSeg_tracked, mat_out_correct_len[1], mat_out_correct_len[2]]]
        spoSeg_tracked, spo_probs, spo_flows = [deinterpolate(mask_type, interp_locs) for mask_type in [spoSeg_tracked, spo_out_correct_len[1], spo_out_correct_len[2]]]
        
    else:
        mat_probs, mat_flows = mat_out_correct_len[0], mat_out_correct_len[1]
        spo_probs, spo_flows = spo_out_correct_len[0], spo_out_correct_len[1]
        

    torch.cuda.empty_cache()
    del proSeg
    del spoSeg
    del matSeg

    return {"cells": (proSeg_tracked, probs, flows),
            "mating": (matSeg_tracked, mat_probs, mat_flows),
            "spores": (spoSeg_tracked, spo_probs, spo_flows)
            }
    
def track_full_lifecycle(proSeg:Union[np.ndarray, List[np.ndarray]], mating: Union[np.ndarray, List[np.ndarray]], tetrads: Union[np.ndarray, List[np.ndarray]], 
                         tetrad_interval:List[int], mating_interval: List[int], movie_length:int, shock_period:Optional[List[int]]=None):
    '''
    Performs the full-lifecycle tracking of yeast cells using the general (proSeg) detections, mating (matSeg) detections, and tetrad (spoSeg) detections using a novel algorithm developed at the Miranda Lab
    at NCSU. The algorithm is as follows:
    
    1) track tetrads
    2) correct general masks with the tracked tetrads
    3) track the corrected general masks
    4) track mating events
    5) correct the general masks using the tracked mating events

    Args:
        proSeg (np.ndarray or list of np.ndarrays) - the general masks either as 3D array (time x height x width) or a list of numpy arrays (height x width)
        proSeg (np.ndarray or list of np.ndarrays) - the general masks either as 3D array (time x height x width) or a list of numpy arrays (height x width)
        proSeg (np.ndarray or list of np.ndarrays) - the general masks either as 3D array (time x height x width) or a list of numpy arrays (height x width)
    Returns
        tuple: a tuple of three 3D numpy arrays of the same shape. In the case that the tetrad/mating interval did not encompass the whole movie,
        the tetrad/mating masks will be padded with 0 arrays to reach the same length as the full-length movie. All arrays in the tuple will have the same shape (timepoints x rows x cols). The ordering of the tuple is as follows:
            np.ndarray[uint16] - the tracked general masks with spores correct
            np.ndarray[uint16] - tracked mating cells
            np.ndarray[uint16] - tracked sporulating cells (tetrads)
        '''
    if shock_period is None:
        shock_period=[0,0]
    
    logger.info("FULL LIFECYCLE TRACKING: 6 Steps")
    logger.info("STEP 1/6: Track Tetrads")
    TET_obj, TET_exists, TETmasks = track_tetrads(tetrads, tetrad_interval, movie_length, shock_period) # step2
    
    logger.info("STEP 2/6: Track Mating Cells")
    Matmasks, mat_no_obj, Mat_cell_data, cell_data = track_mating(mating, mating_interval, shock_period) # step3
    save_images_to_directory(Matmasks, "/home/berk/code/yeastvision/test/step3")
    
    
    logger.info("STEP 3: Correct General Masks")
    proSeg_corrected_tetrads = correct_proSeg_with_tetrads(proSeg, shock_period, TET_obj, TET_exists, TETmasks) # step4
    
    logger.info("STEP 4/6: Track General Masks ")
    Mask7, art_cell_exists = track_general_masks(proSeg_corrected_tetrads) # step 5
    
    logger.info("STEP 5/6: Correct Mating Cells")
    Matmasks, final_mat_cell_data, mat_no_obj = correct_mating(Matmasks, Mask7, mat_no_obj, Mat_cell_data, cell_data) # step6
    save_images_to_directory(Matmasks, "/home/berk/code/yeastvision/test/step6")
    
    logger.info("STEP 6/6: Correct General Masks again")
    final_art_masks = correct_proSeg_with_mating(Matmasks, Mask7, art_cell_exists, mat_no_obj, final_mat_cell_data) # step7
    
    logger.info("Post-Processing Outputs...padding, resizing, transposing")
    mating_tracks = np.array(fill_empty_arrays(Matmasks), dtype=np.uint16)
    
    tetrad_resized = [resize_image(image, final_art_masks[:,:,0].shape) for image in fill_empty_arrays(TETmasks)]
    tetrad_tracks = np.zeros_like(mating_tracks)
    tetrad_tracks[: tetrad_interval[1]] = np.array(tetrad_resized) 
    final_art_masks = np.transpose(final_art_masks, (2, 0, 1)).astype(np.uint16)  
    
    logger.info("Post-Processing Tetrad Masks...Synchronizing tetrad cell values with the general track") 
    tetrad_tracks = synchronize_indices(final_art_masks, tetrad_tracks)

    return final_art_masks, mating_tracks, tetrad_tracks