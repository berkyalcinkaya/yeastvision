from yeastvision.models.matSeg.model import MatSeg

class SpoSeg(MatSeg):
    hyperparams  = { 
    "mean_diameter":30,
    "flow_threshold":0.4, 
    "cell_probability_threshold": 0.0,
    "half_image_size": True}
    types = [float, float, float, bool]
    def __init__(self, params, weights):
        super().__init__(params, weights)
