import cv2
from torch.nn import functional as F
import torch
import numpy as np
from yeastvision.ims.rife_model.pytorch_msssim import ssim_matlab
from queue import Queue, Empty
from .rife_model.RIFE import Model
from yeastvision.ims import rife_model
import copy
import os

RIFE_DIR = rife_model.__path__[0]
RIFE_WEIGHTS_NAME = "flownet.pkl"
RIFE_WEIGHTS_PATH = os.path.join(RIFE_DIR, RIFE_WEIGHTS_NAME)

#def get_interp_mask(original_len, intervals, )

def get_interpolated_length(t, intervals):
    """
    Calculate the number of frames after interpolation.

    Parameters:
    t (int): Original number of frames in the movie.
    intervals (list): List of interval dictionaries with 'start', 'stop', and 'interp' keys.

    Returns:
    int: Total number of frames after interpolation.
    """
    total_frames = t

    for interval in intervals:
        start = interval['start']
        stop = interval['stop']
        interp_factor_log2 = int(interval['interp'])
        num_frames = stop - start
        additional_frames = (num_frames) * (2**interp_factor_log2 - 1)
        total_frames += additional_frames
    return int(total_frames)

def rife_weights_loaded():
    return os.path.exists(RIFE_WEIGHTS_PATH)

def copy_and_normalize(im):
    new_im = copy.deepcopy(im)
    return cv2.normalize(new_im, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    
    model = Model()
    model.load_model(RIFE_DIR, -1)
    model.eval()
    model.device()
    return model, device

def interpolate_intervals(ims, intervals):
    model, device = load_model()
    single_im_shape = ims[0].shape
    new_length = get_interpolated_length(len(ims), intervals)
    #print(intervals)
    #print((new_length, single_im_shape[0], single_im_shape[1]))
    template = np.zeros((new_length, single_im_shape[0], single_im_shape[1]), dtype=np.uint8)

    template_pos = 0  # This keeps track of where we are in the new template
    original_pos = 0
    interval_index = 0
    original_len = len(ims)
    new_intervals = []

    while original_pos < original_len:
        #print()
        #print("original pos", original_pos)
        #print("template pos", template_pos)
        #print("interval indnex", interval_index)

        if interval_index < len(intervals) and original_pos == intervals[interval_index]['start']:
            #print("original pos", original_pos, "aligns with start", intervals[interval_index]['start'])
            start, stop, interp = intervals[interval_index]["start"], intervals[interval_index]["stop"], int(intervals[interval_index]["interp"])
            interval_interp = interpolate(ims[start:stop+1], interp, device, model)
            #print("Interp args: ", start, stop, interp)
            #print("Interpolation output: ", interval_interp.shape)

            num_frames_added = interval_interp.shape[0]
            template[template_pos:template_pos+num_frames_added] = interval_interp

            new_intervals.append({"start": template_pos, "stop": template_pos+num_frames_added, 
                            "interp": int(interp)})


            template_pos += (num_frames_added) # point to last frame added
            original_pos = stop
            interval_index += 1
        else:
            #print("inserting image at original pos", original_pos, "into template pos", template_pos)
            template[template_pos] = copy_and_normalize(ims[original_pos])
            original_pos += 1
            template_pos += 1
        #print("")

    del model
    return template, new_intervals

def interpolate(ims: np.ndarray, exp: int, device, model)->np.ndarray:
    scale = 1
    
    def make_inference(model, I0, I1, n):
        middle = model.inference(I0, I1, scale)
        if n == 1:
            return [middle]
        first_half = make_inference(model, I0, middle, n=n//2)
        second_half = make_inference(model, middle, I1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]
    
    def pad_image(img):
        tmp = 32
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)
        return F.pad(img, padding)
    
    def reformat(im):
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return  cv2.merge((im, im, im))
    
    def make_read_buffer(ims):
        read_buffer = Queue(maxsize=500)
        for im in ims:
            read_buffer.put(im)
        return read_buffer
    # final_im = copy.deepcopy(ims[-1])
    # final_im = cv2.normalize(final_im, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    ims = ims = [reformat(im) for im in ims]
    lastframe = ims[0]
    h,w,_ = lastframe.shape
    ims = ims[1:]
    
    write_buffer = []
    read_buffer = make_read_buffer(ims)

    I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)
    temp = None # save lastframe when processing static frame
    i = 0
    while True:
        if temp is not None:
            frame = temp
            temp = None
        else:
            try:
                frame = read_buffer.get(False)
            except Empty:
                break
        I0 = I1
        I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1)
        I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        break_flag = False
        if ssim > 0.996:    
            try:    
                frame = read_buffer.get(False) # read a new frame
                temp = frame
            except Empty:
                break_flag = True
                frame = lastframe

            I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
            I1 = pad_image(I1)
            I1 = model.inference(I0, I1, scale)
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        
        if ssim < 0.2:
            output = []
            for i in range((2 ** exp) - 1):
                output.append(I0)

        else:
            output = make_inference(model, I0, I1, 2**exp-1) if exp else []

        write_buffer.append(lastframe)
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.append(mid[:h, :w])

        lastframe = frame
        if break_flag:
            break
        i+=1
    return np.array([im[:,:,0] for im in write_buffer], dtype = np.uint8)