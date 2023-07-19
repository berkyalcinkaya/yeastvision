import cv2
from skimage.io import imread
import matplotlib.pyplot as plt
from torch.nn import functional as F
import torch
import numpy as np
from yeastvision.ims.rife_model.pytorch_msssim import ssim_matlab
from queue import Queue, Empty
from skimage.util import img_as_ubyte
from .rife_model.RIFE import Model
import os
from yeastvision.ims import rife_model

RIFE_DIR = rife_model.__path__[0]

def interpolate(ims: np.ndarray, exp: int)->np.ndarray:
    scale = 1
    
    def make_inference(I0, I1, n):
        middle = model.inference(I0, I1, scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n//2)
        second_half = make_inference(middle, I1, n=n//2)
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

    ims = ims = [reformat(im) for im in ims]
    lastframe = ims[0]
    h,w,_ = lastframe.shape
    ims = ims[1:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    
    model = Model()
    model.load_model(RIFE_DIR, -1)
    model.eval()
    model.device()
    

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
            output = make_inference(I0, I1, 2**exp-1) if exp else []

        write_buffer.append(lastframe)
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.append(mid[:h, :w])

        lastframe = frame
        if break_flag:
            break
        print(i)
        i+=1
    
    return np.array([im[:,:,0] for im in write_buffer], dtype = np.uint8)