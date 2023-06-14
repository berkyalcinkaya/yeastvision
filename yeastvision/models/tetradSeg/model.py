from yeastvision.models.cp import CustomCPWrapper

class TetradSeg(CustomCPWrapper):
    def __init__(self, params, weights):
        super().__init__(params, weights)
