from typing import Optional, List, Tuple
import torch
import time
import numpy as np
import logging
from .utils import _get_budSeg, _get_matSeg, _get_proSeg, _get_spoSeg, extend_seg_output
from .mating import track_mating
from .tetrads import track_tetrads
from .full_lifecycle_utils import track_correct_artilife
from .correction import correct_mating, correct_proSeg_with_mating, correct_proSeg_with_tetrads
from yeastvision.ims.interpolate import interpolate_intervals, get_interp_labels, deinterpolate
from yeastvision.track.track import track_proliferating
from yeastvision.track.lineage import LineageConstruction

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
    masks = track_proliferating(masks)
    
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
    masks = track_proliferating(masks)
    
    buds, bud_probs, bud_flows = budSeg.run(to_segment, budSeg_params, budSeg_weights)
    
    lineage = LineageConstruction(masks, buds, forwardskip=3)
    daughters, mothers = lineage.computeLineages()
    
    if interp_intervals:
        masks, probs, flows = [deinterpolate(mask_type, interp_locs) for mask_type in [masks, probs, flows]]
        bud, bud_probs, bud_flows = [deinterpolate(mask_type, interp_locs) for mask_type in [buds, bud_probs, bud_flows]]
    
    torch.cuda.empty_cache()
    del proSeg
    del budSeg
    
    return {"cells": (masks, probs, flows), 
            "buds": (buds, bud_probs, bud_flows),
            "lineages": (daughters, mothers)
    }


def fiest_full_lifecycle(ims: np.ndarray, interp_intervals:Optional[List[dict]]=None, proSeg_params:Optional[dict]=None, proSeg_weights:Optional[str]=None, 
                         matSeg_params:Optional[dict]=None, matSeg_weights:Optional[str]=None, spoSeg_params: Optional[dict]=None, spoSeg_weights:Optional[str]=None,
                         shock_period=Optional[List[int]]):
    '''
    Implementation of Frame Interpolation Enhanced Single-cell Tracking (FIEST) from the paper 'Deep learning-driven 
    imaging of cell division and cell growth across an entire eukaryotic life cycle' for tracking of timeseries movies of
    yeasts that mate, sporulate, and proliferate asexually. This novel algorithm is based on enhancing the overlap of single 
    cell masks in consecutive images through deep learning video frame interpolation. 
    
    Steps: 
    (1) segment ims with matSeg, proSeg, and spoSeg
    (2) track using a novel full-lifecycle track method based frame overlap and corrections that leverage all three outputs
    (3) de-interpolate all outputs
    '''
    logger.info("---FIEST Full Lifecycle Tracking---")
    to_segment = ims
    if interp_intervals:
        logger.info(f"Performing interpolation over intervals:\n {interp_intervals}")
        to_segment, new_intervals = interpolate_intervals(ims, interp_intervals)
        interp_locs = get_interp_labels(len(ims), interp_intervals)

    proSeg, proSeg_params, proSeg_weights = _get_proSeg(proSeg_params, proSeg_weights)
    spoSeg, spoSeg_params, spoSeg_weights = _get_spoSeg(spoSeg_params, spoSeg_weights)
    matSeg, matSeg_params, matSeg_weights = _get_matSeg(matSeg_params, matSeg_weights)

    masks, probs, flows = proSeg.run(to_segment, proSeg_params, proSeg_weights)

    mat_start, mat_stop = matSeg_params["t_start"], matSeg_params["t_stop"]
    spo_start, spo_stop = spoSeg_params["t_start"], spoSeg_params["t_stop"]
    mat_out = matSeg.run(to_segment[mat_start:mat_stop], matSeg_params, matSeg_weights)
    mat_out_correct_len = extend_seg_output(to_segment, mat_out, mat_start, mat_stop)
    spo_out = spoSeg.run(to_segment[spo_start:spo_stop], spoSeg_params, spoSeg_weights)
    spo_out_correct_len = extend_seg_output(to_segment, spo_out, spo_start, spo_stop)

    proSeg_tracked, matSeg_tracked, spoSeg_tracked = track_full_lifecycle(masks, mat_out_correct_len[0], spo_out_correct_len[0],
                                                                          [spo_start, spo_stop], [mat_start, mat_stop], len(to_segment), 
                                                                          )

    if interp_intervals:
        tracked_cells, probs, flows = [deinterpolate(mask_type, interp_locs) for mask_type in [proSeg_tracked, probs, flows]]
        mat_masks, mat_probs, mat_flows = [deinterpolate(mask_type, interp_locs) for mask_type in [matSeg_tracked, mat_out[1], mat_out[2]]]
        spo_masks, spo_probs, spo_flows = [deinterpolate(mask_type, interp_locs) for mask_type in [spoSeg_tracked, spo_out[1], spo_out[2]]]

    torch.cuda.empty_cache()
    del proSeg
    del spoSeg
    del matSeg

    return {"cells": (tracked_cells)}
    
    
def track_full_lifecycle(proSeg, mating, tetrads, tetrad_interval, mating_interval, movie_length, shock_period):
    tracked_tet_dict = track_tetrads(tetrads, tetrad_interval, movie_length, shock_period) # step2
    tracked_mat_dict = track_mating(mating, mating_interval, shock_period) # step3

    proSeg_corrected_tetrads = correct_proSeg_with_tetrads(proSeg, tracked_tet_dict) # step4
    
    # step5 takes a tensor
    proSeg_tracked_dict = track_correct_artilife(proSeg_corrected_tetrads, shock_period=shock_period) # step5
    
    # TODO: transpose back into a list (y,x ) images for step6 and step7
    
    mating_corrected = correct_mating(tracked_mat_dict, proSeg_tracked_dict) # step6
    proSeg_corrected = correct_proSeg_with_mating(mating_corrected, proSeg_tracked_dict) # step7
    
    return proSeg_corrected["Mask3"], mating_corrected["Matmasks_py"], tracked_tet_dict["TETmasks_py"]