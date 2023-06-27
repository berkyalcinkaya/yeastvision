from cellpose import models
from yeastvision.models.model import Model as CustomModel
import cv2
import numpy as np
from cellpose.transforms import normalize99


class CustomCellpose(models.Cellpose):
    def __init__(self, gpu=True, model_type='cyto', pretrained_model = None, net_avg=False, device=None):
        super(models.Cellpose, self).__init__()
        self.torch = True
        
        # assign device (GPU or CPU)
        sdevice, gpu = models.assign_device(self.torch, gpu)
        self.device = device if device is not None else sdevice
        self.gpu = gpu
        
        model_type = 'cyto' if pretrained_model else None
        
        self.diam_mean = 30. #default for any cyto model 
        nuclear = 'nuclei' in model_type
        if nuclear:
            self.diam_mean = 17. 
        
        self.cp = models.CellposeModel(device=self.device, gpu=self.gpu,
                                pretrained_model= pretrained_model,
                                model_type=None,
                                diam_mean=self.diam_mean,
                                net_avg=net_avg)
        self.cp.model_type = model_type
        
        self.pretrained_size = models.size_model_path(model_type, self.torch)

        self.sz = models.SizeModel(device=self.device, pretrained_size=self.pretrained_size,
                            cp_model=self.cp)
        self.sz.model_type = model_type


class CustomCPWrapper(CustomModel):
    hyperparams  = {  
    "Mean Diameter":0, 
    "Flow Threshold":0.4, 
    "Cell Probability Threshold": 0,
    "Speed mode": False}
    types = [None, None,None,bool]
    
    loss = "categorical crossentropy"
    trainparams = {
                "learning_rate":0.1,
                "weight_decay": 0.0001,
                "n_epochs":100}

    def __init__(self, params, weights):
        super().__init__(params, weights)
        if self.params["Mean Diameter"] == 0:
            self.params["Mean Diameter"] = None
        
        self.model = CustomCellpose(pretrained_model=self.weights)
        self.cpAlone = self.model.cp
        
    
    def processProbability(self, rawProb):
        return (np.clip(normalize99(rawProb.copy()), 0, 1) * 255).astype(np.uint8)

    
    def train(self, ims, labels, params):
        # ims must be 3D for cellpose
        ims = [cv2.merge((im,im,im)) for im in ims]
        # call train the models.CellPoseModel
        self.model.cp.train(ims, labels, 
                                        channels=[0,0], 
                                        save_path=params["dir"], 
                                        nimg_per_epoch=8,
                                        learning_rate = params['learning_rate'], 
                                        weight_decay = params['weight_decay'], 
                                        n_epochs = int(params['n_epochs']),
                                        model_name = params["model_name"])

    @classmethod
    def run(cls, ims, params, weights):
        params = params if params else cls.hyperparams

        model = cls(params, weights)
        ims = [cv2.merge((im,im,im)) for im in ims]
        assert len(ims[0].shape)==3

        model.masks, flows, _, model.diams = model.model.eval(ims, 
                                                diameter = model.params["Mean Diameter"], 
                                                channels = [0, 0],
                                                cellprob_threshold = model.params["Flow Threshold"], 
                                                do_3D=False)
        model.cellprobs = [flow[2] for flow in flows]
        model.cellprobs = np.array((model.processProbability(model.cellprobs)), dtype = np.uint8)

        return np.array(model.masks, dtype = np.uint16), np.array(model.cellprobs, dtype = np.float32)