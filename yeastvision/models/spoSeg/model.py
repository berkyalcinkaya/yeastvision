from yeastvision.models.cp import CustomCPWrapper

class SpoSeg(CustomCPWrapper):
    hyperparams  = { 
    "mean_diameter":30,
    "flow_threshold":0.4, 
    "cell_probability_threshold": 0.0}
    tpes = [float, float, float]
    def __init__(self, params, weights):
        super().__init__(params, weights)
