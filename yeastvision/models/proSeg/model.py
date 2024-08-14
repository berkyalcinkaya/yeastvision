from yeastvision.models.cp import CustomCPWrapper

class ProSeg(CustomCPWrapper):
    hyperparams  = { 
    "mean_diameter":30.0,
    "flow_threshold":0.9, 
    "cell_probability_threshold": 0.5}
    types = [float, float,float]

    def __init__(self, params, weights):
        super().__init__(params, weights)

# # need to update here
# class ArtilifeFullLifeCycle(ProSeg):
#     hyperparams  = { 
#     "mean_diameter":30, 
#     "flow_threshold":0.9, 
#     "cell_probability_threshold": 0.5,
#     "has_mating": False,
#     "has_spores": False,
#     "is_time_series": True}
#     types = [None, None,None, bool, bool, bool, bool]

#     '''
#     params also include:
#         matStart - start index of mating cells
#         matStop - stop index of mating cells (inclusive)
#         matSeg - weights to use for matSeg 

#         sporeStart - start index of sporulating cells
#         sporeStop - stop index of sporulating cells (inclusive)
#         tetradSeg - weights to use for tetradSeg
#     these are hard coded via guiparts.ArtilifeParamDialog
#     '''
#     def __init__(self, params, weights):
#         super().__init__(params, weights)

#         self.matSeg, self.tetradSeg = None,None
#         self.matMasks, self.matprobs = None, None
#         self.tetraMasks, self.tetraprobs = None, None

#         if params["has_mating"]:
#             from models.matSeg.model import MatSeg
#             self.matSeg = MatSeg(self.params, produce_weight_path("matSeg", self.params["matSeg"]))

#         if params["has_spores"]:
#             from models.spoSeg.model import TetradSeg
#             self.tetradSeg = TetradSeg(self.params, produce_weight_path("tetradSeg", self.params["tetradSeg"]))

#     def addTetrads(self, ims):
#         tetraSlice = slice(int(self.params["sporeStart"]), int(self.params["sporeStop"])+1)
#         self.tetraMasks, tetraFlows, _, _ = self.tetradSeg.model.eval(ims[tetraSlice], 
#                                                                 diameter = self.params["Mean Diameter"], 
#                                                                 channels = [0,0],
#                                                                 cellprob_threshold = self.params["Flow Threshold"],
#                                                                 do_3D = False)
#         self.tetraprobs = [flow[2] for flow in tetraFlows]
#         self.tetraprobs = np.array(self.process_probability(self.tetraprobs), dtype = np.uint8)

#         if self.params["is_time_series"]:
#             newTetraMasks, newMasks = track_obj([im[:,:,0] for im in ims[tetraSlice]], self.masks[tetraSlice], self.tetraMasks[tetraSlice], False)
#             tetraSlice = slice(int(self.params["sporeStart"]), int(self.params["sporeStop"])-1)
#         else:
#             for tetraMask, cellMask in zip(self.tetraMasks, self.masks):
#                 if np.any(tetraMask>=1):
#                     newMask = addMasks(tetraMask, cellMask)
#                     newMasks.append(newMask)
#                 else:
#                     newMasks.append(cellMask)
#             newTetraMasks = self.tetraMasks

#         self.masks[tetraSlice] = newMasks
#         tetraMasks = np.zeros_like(self.masks)
#         tetraMasks[tetraSlice] = np.array(newTetraMasks, dtype = np.uint16)
#         self.tetraMasks = tetraMasks


#     def addMatingCells(self, ims):
#         matSlice = slice(int(self.params["matStart"]),int(self.params["matStop"]) + 1)
#         self.matMasks, matFlows, _, _ = self.matSeg.model.eval(ims[matSlice], 
#                                                                 diameter = self.params["mean_diameter"], 
#                                                                 channels = [0,0],
#                                                                 cellprob_threshold = self.params["flow_threshold"],
#                                                                 do_3D = False)
#         print("got mating cells")
#         self.matprobs = [flow[2] for flow in matFlows]
#         self.matprobs = np.array(self.process_probability(self.matprobs), dtype = np.uint8)
        
#         if self.params["is_time_series"]:
#             matSlice = slice(int(self.params["matStart"]), int(self.params["matStop"])-1)
#             newMatMasks, newMasks = get_mating_data(self.matMasks[matSlice], self.masks[matSlice])
#         else:
#             newMasks = []
#             for matMask, cellMask in zip(self.matMasks, self.masks):
#                 if np.any(matMask>=1):
#                     newMask = addMasks(matMask, cellMask)
#                     newMasks.append(newMask)
#                 else:
#                     newMasks.append(cellMask)
#             newMatMasks = self.matMasks

#         self.masks[matSlice] = newMasks
#         matMasks = np.zeros_like(self.masks)
#         matMasks[matSlice] = np.array(newMatMasks, dtype = np.uint16)
#         self.matMasks = matMasks

#     @classmethod
#     @torch.no_grad()
#     def run(cls, ims, params, weights):
#         params = params if params else cls.hyperparams
#         model = cls(params, weights)
#         ims3D = [cv2.merge((im,im,im)) for im in ims]
#         assert len(ims3D[0].shape)==3

#         if not params["Mean Diameter"]:
#             evaluator = model.cpAlone
#             model.masks, flows, _ = evaluator.eval(ims3D, 
#                                                     diameter = model.params["Mean Diameter"], 
#                                                     channels = [0, 0],
#                                                     cellprob_threshold = model.params["Flow Threshold"], 
#                                                     do_3D=False)
#         else:
#             evaluator = model.model
#             model.masks, flows, _, model.diams = evaluator.eval(ims3D, 
#                                                     diameter = model.params["Mean Diameter"], 
#                                                     channels = [0, 0],
#                                                     cellprob_threshold = model.params["Flow Threshold"], 
#                                                     do_3D=False)
#         print("process probability")
#         model.cellprobs = [flow[2] for flow in flows]
#         model.cellprobs = np.array((model.process_probability(model.cellprobs)), dtype = np.uint8)
        
#         print("formatting return")
#         model.masks = np.array(model.masks, dtype = np.uint16)

#         if model.matSeg:
#             print('add mating cell')
#             model.addMatingCells(ims3D)
#         if model.tetradSeg:
#             model.addTetrads(ims3D)
        
#         print("finished")
#         arti = (model.masks, model.cellprobs)
#         mating = (model.matMasks, model.matprobs)
#         tetra = (model.tetraMasks, model.tetraprobs)

#         del model

#         return {"artilife": arti,
#                 "mating": mating,
#                 "tetrads": tetra
#                 }


    
