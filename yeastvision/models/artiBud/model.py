from yeastvision.models.cp import CustomCPWrapper

class ArtiBud(CustomCPWrapper):
    def __init__(self, params, weights):
        super().__init__(params, weights)