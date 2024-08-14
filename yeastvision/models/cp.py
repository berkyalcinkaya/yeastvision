from yeastvision.models.model import Model as CustomModel
import cv2
import numpy as np
from cellpose.transforms import normalize99
import torch
from cellpose.models import CellposeModel, size_model_path, SizeModel, Cellpose
from cellpose.core import assign_device

from yeastvision.utils import normalize_im


class CellposeAllowPreTrainedModel(Cellpose):
    """ main model which combines SizeModel and CellposeModel, and allows for a pretrained model to be loaded
    for cellpose model

    Parameters
    ----------

    gpu: bool (optional, default False)
        whether or not to use GPU, will check if GPU available

    model_type: str (optional, default 'cyto')
        'cyto'=cytoplasm model; 'nuclei'=nucleus model; 'cyto2'=cytoplasm model with additional user images

    net_avg: bool (optional, default False)
        loads the 4 built-in networks and averages them if True, loads one network if False

    device: torch device (optional, default None)
        device used for model running / training 
        (torch.device('cuda') or torch.device('cpu')), overrides gpu input,
        recommended if you want to use a specific GPU (e.g. torch.device('cuda:1'))
    
    pretrained_cp_model: path to custom cellpose weights (optional, default None)
        weights for a pretrained CellposeModel model

    """
    def __init__(self, gpu=False, model_type='cyto', net_avg=False, device=None, pretrained_model = None):
        super(Cellpose, self).__init__()
        self.torch = True
        
        # assign device (GPU or CPU)
        sdevice, gpu = assign_device(self.torch, gpu)
        self.device = device if device is not None else sdevice
        self.gpu = gpu
        
        model_type = 'cyto' if model_type is None else model_type
        
        self.diam_mean = 30. #default for any cyto model 
        nuclear = 'nuclei' in model_type
        if nuclear:
            self.diam_mean = 17. 
        
        self.cp = CellposeModel(device=self.device, gpu=self.gpu,
                                model_type=model_type,
                                diam_mean=self.diam_mean,
                                net_avg=net_avg,
                                pretrained_model=pretrained_model)
        self.cp.model_type = model_type
        
        # size model not used for bacterial model
        self.pretrained_size = size_model_path(model_type, self.torch)
        self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size,
                            cp_model=self.cp)
        self.sz.model_type = model_type


class CustomCPWrapper(CustomModel):
    hyperparams  = {  
    "mean_diameter":-1.0, 
    "flow_threshold":0.4, 
    "cell_probability_threshold": 0}
    types = [float, float,float]
    
    loss = "categorical_crossentropy"
    trainparams = {
                "learning_rate":0.1,
                "weight_decay": 0.0001,
                "n_epochs":100
                }

    def __init__(self, params, weights):
        super().__init__(params, weights)

        self.mean_diam = self.params["mean_diameter"]
        self.do_size_estimation = self.mean_diam == -1
        self.no_diam = self.mean_diam == 0
        self.use_size_model = self.do_size_estimation

        if self.use_size_model:
            if self.do_size_estimation:
                self.mean_diam = None
            self.model  = CellposeAllowPreTrainedModel(gpu=True, pretrained_model=self.weights)
        else:
            self.model = CellposeModel(gpu=True, pretrained_model=self.weights)
            if self.no_diam:
                self.mean_diam = self.model.diam_labels
    
    def eval_params(self, params):
        a = {"diameter":self.mean_diam,
                "channels": [0,0],
                "cellprob_threshold": params["cell_probability_threshold"],
                "flow_threshold": params["flow_threshold"],
                "do_3D": False,
                "min_size":-1}
        return a
    
    def process_probability(self, rawProb):
        return (np.clip(normalize99(rawProb), 0, 1) * 255).astype(np.uint8)

    def train(self, ims, labels, params, savepath):
        ims = [cv2.merge((im,im,im)) for im in ims]
        
        modelToTrain = self.model if isinstance(self.model, CellposeModel) else self.model.cp
        modelToTrain.train(ims, labels, 
                                        channels=[0,0], 
                                        save_path=params["dir"], 
                                        min_train_masks=1,
                                        nimg_per_epoch=8,
                                        learning_rate = params['learning_rate'], 
                                        weight_decay = params['weight_decay'], 
                                        n_epochs = int(params['n_epochs']),
                                        model_name = params["model_name"])
    def get_masks_and_flows(self, ims):
        if self.use_size_model:
            masks, flows, _, _ = self.model.eval(ims, **self.eval_params(self.params))
        else:
            masks, flows, _ = self.model.eval(ims, **self.eval_params(self.params))
        return masks,flows
    
    def process_flows(self, flow_list) -> np.ndarray[np.uint8]:
        '''Reduces cellpose 3D flows into 2D images by taking the magnitude of the flow vector'''
        # Calculate the magnitude of the vector along the last dimension
        flowsXY = np.array([normalize_im(np.linalg.norm(flow[0], axis=-1)) for flow in flow_list], dtype=np.float32) * 255
        return flowsXY.astype(np.uint8)

    @classmethod
    @torch.no_grad()
    def run(cls, ims, params, weights):
        '''Class method for running a built-in model type
        
        Returns
        - masks (np.ndarray[uint16]): the segmentation results where each pixel is given an index according to the object
        - cellprobs (np.ndarray[uint8]): the probability of each pixel being an object or background. Scaled such that 255 is 100% confidence
                                        and 0 is 0% confidence
        - flowsXY (np.ndarray[np.uint8]): each pixel gives the magnitude of the outputted flow vector at that location
        '''
        params = params if params else cls.hyperparams
        model = cls(params, weights)
        #ims3D = [cv2.merge((im,im,im)) for im in ims]
        if isinstance(ims, np.ndarray):
            ims = list(ims)
        masks, flows = model.get_masks_and_flows(ims)
        cellprobs = [flow[2] for flow in flows]
        flowsXY = model.process_flows(flows)
        cellprobs = np.array((model.process_probability(cellprobs)), dtype = np.uint8)
        masks = np.array(masks, dtype = np.uint16)
        del model
        return masks, cellprobs, flowsXY