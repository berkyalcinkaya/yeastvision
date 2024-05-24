from yeastvision.models.cp import CustomCPWrapper

class FlouSeg(CustomCPWrapper):
    hyperparams  = { 
    "mean_diameter":30.0,
    "flow_threshold":0.9, 
    "cell_probability_threshold": 0.5}
    types = [float, float,float]

    def __init__(self, params, weights):
        super().__init__(params, weights)