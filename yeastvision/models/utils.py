import patchify
import json
import numpy as np
from patchify import patchify, unpatchify
import skimage
from skimage.morphology import disk, binary_erosion, binary_dilation
import yeastvision.models as models
from os.path import join
import os
from yeastvision.utils import capitalize_first_letter

CUSTOM_MODELS_FILE = "model_types.json"
MODEL_DIR = models.__path__[0]
with open(os.path.join(MODEL_DIR, CUSTOM_MODELS_FILE), "r") as file:
    CUSTOM_MODEL_TYPES = json.load(file)

def is_RGB(ims):
    return ims.shape[-1] == 3

def getBuiltInModelTypes():
    '''Retrieves the builtin model names'''
    models = [model for model in os.listdir(MODEL_DIR) if os.path.isdir(join(MODEL_DIR, model)) and model != "__pycache__"]
    return models

def getModelsByType(model_type):
    '''Returns all the weights that exist within model_type directory'''
    model_dir = os.path.join(MODEL_DIR, model_type)
    return [file for file in os.listdir(model_dir) if "." not in file and file != "__pycache__"]

def getModelLoadedStatus(model):
    '''Determines whether or not a model is loaded. Built for default models type
    Assumes that model name is identical to model type'''
    return os.path.exists(produce_weight_path(model, model))

def produce_weight_path(modeltype, modelname):
    return join(MODEL_DIR, modeltype, modelname)

def shrink_bud(bud_mask, footprint = None, kernel_size = 2):
    if not footprint:
        footprint = disk(kernel_size)
    return binary_erosion(bud_mask, footprint)

def enlarge_bud(bud_mask, footprint = None, kernel_size = 4):
    if not footprint:
        footprint = disk(kernel_size)
    return binary_dilation(bud_mask, footprint)

def normalizeIm(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im))

def addMasks(maskToAdd, origMask):
    newMask = origMask.copy()
    for i in np.unique(maskToAdd)[1:]:
        boolMask = maskToAdd==i
        valExists, maskValToReplace = getMaskOverlapVal(boolMask, origMask)
        
        if valExists:
            newMask[newMask==maskValToReplace]=0
            newMask[boolMask] = maskValToReplace
        else:
            newMask[boolMask] = maskValToReplace
    return newMask

def getMaskOverlapVal(boolMask, fullMask):
    vals, counts = np.unique(fullMask[boolMask], return_counts=True)
    if len(vals)>1:
        val = vals[np.argmax(counts)]
        if val==0:
            return False, fullMask.max()+1
        else:
            return True, val
    else:
        if vals[0]==0:
            return False, fullMask.max()+1
        else:
            return True, vals[0]
        
def test_time_aug(im, model):
    temp_test_img = im[0,:,:,0]
    p0 = model.predict(np.expand_dims(temp_test_img, axis=0), verbose = 0)[0][:, :, 0]

    p1 = model.predict(np.expand_dims(np.fliplr(temp_test_img), axis=0), verbose = 0)[0][:, :, 0]
    p1 = np.fliplr(p1)

    p2 = model.predict(np.expand_dims(np.flipud(temp_test_img), axis=0), verbose = 0)[0][:, :, 0]
    p2 = np.flipud(p2)

    p3 = model.predict(np.expand_dims(np.fliplr(np.flipud(temp_test_img)), axis=0), verbose =0 )[0][:, :, 0]
    p3 = np.fliplr(np.flipud(p3))

    p = ((p0 + p1 + p2 + p3)/4).astype(np.float32)
    return p

def prediction(models, image, input_shape = 256, buffer_size = 10,  double_patch = False, do_thresh = True, thresh = 0.50, pad_mode = "symmetric", preprocess = skimage.exposure.equalize_adapthist, do_aug = True):
    '''
    Params
    ---------
    Image (ndarray) -  2D image
    Model (tf.Model) - trained unet
    patch_shape (tuple) - size of 2D tiles after padding, should match model input reqs
    buffer_size (int) - size of additional paddding added
    
    Returns
    --------
    Ndarray uint8
    '''
    if type(models) != list:
        models = [models]

    patch_size = input_shape - 2*buffer_size
    if double_patch:
        step_size = patch_size//2    
    else:
        step_size = patch_size

    large_image = pad_for_patching(image, patch_size = patch_size)
    patches = patchify(large_image, (patch_size, patch_size), step=step_size) 
    predicted_patches = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i,j,:,:]

            # we pad our patch for prediction
            #single_patch = np.pad(single_patch, ((10,10), (10,10)), mode = pad_mode)
            single_patch = np.pad(single_patch, ((buffer_size,buffer_size), (buffer_size,buffer_size)), mode = pad_mode)
            # add a leading and trailing dimension 
            single_patch_norm = np.expand_dims(preprocess(np.array(single_patch)),2)
            single_patch_input=np.expand_dims(single_patch_norm, 0)
            
            avg_image = np.zeros((input_shape, input_shape), dtype = np.float32)
            for model in models:
                if do_aug:
                    single_patch_prediction = test_time_aug(single_patch_input, model)
                else:
                    single_patch_prediction = (model.predict(single_patch_input, verbose = 0)[0,:,:,0])
                avg_image = np.add(avg_image, single_patch_prediction) 
            avg_image = avg_image/len(models)
            
            if do_thresh:
                single_patch_prediction = (avg_image>thresh).astype(np.uint8)

            # crop additional padding, reducing image back to size patch_size
            single_patch_prediction = single_patch_prediction[buffer_size:(input_shape-buffer_size), buffer_size:(input_shape-buffer_size)]
            predicted_patches.append(single_patch_prediction)

    # unpatchify the list of images, using format required by patchify module
    # reshape 
    predicted_patches = np.array(predicted_patches)
    predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], patch_size,patch_size) )
    reconstructed_large_image = unpatchify(predicted_patches_reshaped, large_image.shape)

    # remove padding added to large image during pad_for_patching()
    reconstructed_image = reconstructed_large_image[0:image.shape[0], 0:image.shape[1]]
    
    return reconstructed_image

def pad_for_patching(image, patch_size = 256, pad_mode = "constant"):
    '''adds padding determined by get_pad_dim to image and returns a padded copy of same type '''

    def get_closest_int(num, target = 256):
        '''returns closest integer greater or equal to num divisible by target '''
        return (num + target) - (num % target)

    def get_pad_dim_for_patching(shape, patch_size = 256):
        '''returns the neccesary additions vertically and horizonatally for an image to be divisible
        evenly by the specified patch_size (patches must be symmetrical)'''
        r,c = shape
        r_pad = (get_closest_int(r, target = patch_size) - r)
        c_pad = (get_closest_int(c, target = patch_size) - c)
        return r_pad, c_pad

    
    r_pad, c_pad = get_pad_dim_for_patching(image.shape[:2], patch_size = patch_size) # 2D images only
    padded_image = np.pad(image, ((0, r_pad), (0, c_pad)), mode = pad_mode)
    return padded_image


def patchify_for_train(ims, input_shape = 256, double_patch = True):

    patch_size = input_shape 
    if double_patch:
        step_size = patch_size//2    
    else:
        step_size = patch_size

    out_patches = []
    for image in ims:
        large_image = pad_for_patching(image, patch_size = patch_size)
        patches = patchify(large_image, (patch_size, patch_size), step=step_size) 
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                out_patches.append(patches[i,j,:,:])
    return out_patches