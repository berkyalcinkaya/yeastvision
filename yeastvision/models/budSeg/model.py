from yeastvision.models.cp import CustomCPWrapper

class BudSeg(CustomCPWrapper):
    hyperparams  = { 
    "flow_threshold":0.4, 
    "cell_probability_threshold": 0.0}
    types = [float,float]
    def __init__(self, params, weights):
        params["mean_diameter"] = 0 # BudSeg should do not scaling of image sizes
        super().__init__(params, weights)
    

