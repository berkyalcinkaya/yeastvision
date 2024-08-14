import numpy as np
from typing import Optional, List
from yeastvision.models.utils import produce_weight_path


def fiest_full_lifecycle(ims: np.ndarray, interpolation_intervals:Optional[List[dict]]=None, 
                         proSeg_params:Optional[dict]=None, matSeg_params:Optional[dict]=None, 
                         spoSeg_params: Optional[dict]=None):
    '''Implementation of Frame Interpolation Enhanced Single-cell Tracking (FIEST) from the paper 'Deep learning-driven 
    imaging of cell division and cell growth across an entire eukaryotic life cycle' for tracking of timeseries movies of
    yeasts that mate, sporulate, and proliferate asexually. This novel algorithm is based on enhancing the overlap of single 
    cell masks in consecutive images through deep learning video frame interpolation. 
    
    Steps:
    1) segment ims with matSeg, proSeg, and spoSeg
    '''
    return