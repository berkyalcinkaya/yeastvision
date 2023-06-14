import numpy as np
from skimage.feature import blob_log as bl
from yeastvision.flou.utils import draw_blobs, normalize_im
import math
import numpy as np
from skimage.draw import disk
from skimage.feature import blob_log as bl
from skimage.exposure import equalize_adapthist
from skimage.morphology import dilation
import math

class Blob():
    def __init__(self, ims, thresh, minSize, maxSize, cells  = None):
        self.ims = ims
        self.cells = cells
        self.threshold = thresh
        self.minSigma = minSize
        self.maxSigma = maxSize
    
    def drawBlobs(self):
        masks = np.zeros_like(self.ims)
        for idx in range(self.ims.shape[0]):
            im = normalize_im(self.ims[idx])
            if type(self.cells) is np.ndarray:
                mask = self.cells[idx]
            blobs = bl(im, min_sigma = self.minSigma, max_sigma = self.maxSigma, num_sigma = 30, threshold = self.threshold)
            blobs[:, 2] = blobs[:, 2] * math.sqrt(2)

            label = 1
            for blob in blobs:
                r,c,radius = blob
                print(int(radius))
                rr, cc = disk((int(r),int(c)), int(radius), shape = im.shape)
                if type(self.cells) is np.ndarray and np.any(mask[rr,cc]>0):
                    masks[idx, rr, cc] =  label
                    label+=1
                elif type(self.cells) is not np.ndarray:
                    masks[idx, rr, cc] =  label
                    label+=1
        return masks
    
    
    @classmethod
    def run(cls, ims, thresh, minSize, maxSize, cells  = None):
        seg = cls(ims, thresh, minSize, maxSize, cells = cells)
        return seg.drawBlobs()