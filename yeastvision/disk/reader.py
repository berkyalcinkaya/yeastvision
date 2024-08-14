import glob
import os 
import numpy as np
from skimage.io import imread
from skimage.measure import label
from yeastvision.utils import get_mask_contour
import pickle
from tqdm import tqdm
from os.path import splitext, basename

def get_channels_from_dir(dir, num_channels):
    '''Loads images seperately by channel 
    
    Parameters
    ----------
    
    dir: string 
        The directory from which to load each set of channels. All files should be image files and should be named such that the first num_chhanels images below to distinct channel groups
    
    num_channels: int
        Number of channels to load
        
    Returns
    -------
    
    channels: a 4d Ndarray np.uint16 whose last dimension is the channels
        Loaded image data. Has shape (num_ims/num_channels, test_im.shape[0], test_im.shape[1], num_channels)
    
    files: list[list[string]]
        nested list structure containing all filepaths 
    
    names: list[string]
        list of length num_channels of channel id strings obtained by splitting all letters after the last _ in filenames'''

    all_files = sorted(glob.glob(os.path.join(dir, "*")))
    num_ims = len(all_files)

    if num_ims % num_channels != 0:
        raise IndexError

    test_im = imread(all_files[0])
    im_shape = (num_ims/num_channels, test_im.shape[0], test_im.shape[1], num_channels)
    channels = np.zeros(im_shape)
    files = [[]*num_channels]
    names = []
    
    for i in range(num_channels):
        names.append(all_files[i].split("_")[-1])
        for n in range(i,num_ims, num_channels):
            channels[n,:,:,i] = imread(all_files[n])
            files[i].append(all_files[n])

    return channels, files, names



def get_channel_files_from_dir(dir, num_channels):
    '''Loads images seperately by channel 
    
    Parameters
    ----------
    
    dir: string 
        The directory from which to load each set of channels. All files should be image files and should be named such that the first num_chhanels images below to distinct channel groups
    
    num_channels: int
        Number of channels to load
        
    Returns
    -------
    
    channels: a 4d Ndarray np.uint16 whose last dimension is the channels
        Loaded image data. Has shape (num_ims/num_channels, test_im.shape[0], test_im.shape[1], num_channels)
    
    files: list[list[string]]
        nested list structure containing all filepaths 
    
    names: list[string]
        list of length num_channels of channel id strings obtained by splitting all letters after the last _ in filenames'''

    all_files = sorted(glob.glob(os.path.join(dir, "*")))
    num_ims = len(all_files)

    if num_ims % num_channels != 0:
        raise IndexError

    test_im = imread(all_files[0])
    files = [[]*num_channels]
    names = []
    
    for i in range(num_channels):
        names.append(all_files[i].split("_")[-1])
        for n in range(i,num_ims, num_channels):
            files[i].append(all_files[n])

    return files, names
    



def loadPkl(path, parent):
    '''data keys
    [Images, Saturation, Masks, Contours, Channels, Labels, Cells, Lineages]
    '''
    parent.overrideNpyPath = path
    parent.sessionId = splitext(basename(path))[0]
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    # images, saturation
    ims, sat, names = data["Images"], data["Saturation"], data["Channels"]
    dirs, files = data["Dirs"], data["Files"]
    for i in range(len(ims)):
        parent.imData.dirs.append(dirs[i])
        parent.imData.files.append(files[i])
        parent.loadImages(ims[i], name = names[i])

    masks, contours,labels = data["Masks"], data["Contours"], data["Labels"]
    cellData = data['Cells']
    for i in range(len(masks)):
        maskStack = masks[i]
        hasFloats = maskStack.shape[0]>1
        if hasFloats:
            currMask = (maskStack[0,:,:,:], maskStack[1,:,:,:])
        else:
            currMask = maskStack[0,:,:,:]
        parent.loadMasks(currMask, name = labels[i], contours = contours[i])

        parent.cellData[-1] = (cellData[i])
    parent.checkDataAvailibility()
    parent.saveData()

class ImageData():
    fileEndings = ['.tif','.tiff',
                            '.jpg','.jpeg','.png','.bmp',
                            '.pbm','.pgm','.ppm','.pxm','.pnm','.jp2',
                            '.TIF','.TIFF',
                            '.JPG','.JPEG','.PNG','.BMP',
                            '.PBM','.PGM','.PPM','.PXM','.PNM','.JP2']
    def __init__(self, parent):
        self.isDummys = []
        self.parent  = parent

        self.data = None

        self.y,self.x = 0,0
        self.Z = 0 
        self.maxZ = 0
        self.maxT = 0
        self.yVals, self.xVals, self.maxTs = [0],[0],[0]
        self.dirs = []
        self.files = []

        self.channels = []
        self.saturation = []
    
    def loadMultiChannel(self, fileIds, files):
        for fileId in fileIds:
            currFiles = sorted([file for file in files if fileId in file])
            self.files.append(currFiles)
            head, tail = os.path.split(files[0])
            self.dirs.append(head)
            ims = np.array([imread(im) for im in currFiles])
            self.parent.loadImages(ims, name = fileId)
    
    def loadFileList(self, files):
        self.files.append(files)
        head, tail = os.path.split(files[0])
        self.dirs.append(head)
        ims = np.array([imread(im) for im in files])
        self.parent.loadImages(ims, name = tail.split(".")[0])

    
    def load(self, path):
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*")))
            ims = np.array([imread(im) for im in files])
            self.files.append(files)
        else:
            ims = imread(path)
            if len(ims.shape) == 2:
                ims = np.expand_dims(ims,0)
            self.files.append([path])
        self.addZ(ims)
        
        head, tail = os.path.split(path)
        self.dirs.append(head)
    
    def addZ(self, ims, saturation = None):
        if self.y==0 and self.x==0 and self.maxT == 0:
            self.maxTs, self.yVals, self.xVals = [ims.shape[-3]-1], [ims.shape[-2]], [ims.shape[-1]]
        else:
            self.maxTs.append(ims.shape[-3]-1)
            self.yVals.append(ims.shape[-2])
            self.xVals.append(ims.shape[-1])
        

        if self.parent.imLoaded:
            self.maxZ+=1
            self.channels.append(ims)
        else:
            self.maxZ = 0
            self.channels = [ims]
        
        if saturation:
            self.saturation.append(saturation)
        else:
            self.getSaturation(ims)
    
    def removeZ(self, z):
        self.maxZ-=1 if self.maxZ>0 else 0

        del self.channels[z]
        del self.yVals[z]
        del self.xVals[z]
        del self.maxTs[z]
        del self.dirs[z]
        del self.files[z]
        del self.channels[z]


    def getSaturation(self, ims):
        newSaturations = [[currIm.min(), currIm.max()] for currIm in ims]
        self.saturation.append(newSaturations)
    
    @property
    def x(self):

        return self.xVals[self.parent.imZ]
    @x.setter
    def x(self, num):
        self._x = num
    
    @property
    def y(self):

        return self.yVals[self.parent.imZ]
    @y.setter
    def y(self, num):
        self._y = num

    @property
    def maxT(self):
        return self.maxTs[self.parent.imZ]
    @maxT.setter
    def maxT(self, num):
        self._maxT = num
    
    @property
    def isDummy(self):
        return self.isDummys[self.parent.maskZ]
   
    @isDummy.setter
    def isDummy(self, boolean):
        self.isDummys[self.parent.maskZ] = boolean



class MaskData(ImageData):
    def __init__(self, parent):
        super().__init__(parent)

        self.floats = []
        self.contours = []
    
    def load(self, path):
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*")))
            ims = np.array([imread(im, plugin = "pil") for im in files])
        else:
            ims = (imread(path, plugin = "pil"))
            if len(ims.shape) == 2:
                ims = np.expand_dims(ims,0)
        
        ims = self.readMaskType(ims)
        self.addZ(ims)
    
    def readMaskType(self,ims):
        if np.all(ims == 0):
            return ims
        if self.isBinary(ims):
            ims = self.labelIms(ims)
        return ims

    def removeDummys(self):
        for _ in self.isDummys:
            del self.channels[0]
            del self.contours[0]
            del self.yVals[0]
            del self.xVals[0]
            del self.maxTs[0]
        self.maxZ = 0
    
    def loadMask(self,im):
        im = (imread(im, plugin="pil"))
    
    def loadProbIm(self,im):
        return imread(im).astype(np.float32)
    
    def labelIms(self, ims):
        return np.array([label((im>0).astype(ims.dtype)) for im in ims])

    def isBinary(self, mask):
        return np.array_equal(mask, mask.astype(bool))
    
    def isFloat(self,im):
        return im.dtype.kind == "f" and np.all(im<=1.0)
    
    def addZ(self,imData, pred = False, contours = None):
    
        hasFloats = False
        if type(imData) is tuple:
            ims,floats = imData
            hasFloats = True
            floats = (floats).astype(np.uint8)
        else:
            ims = imData
        
        ims  = self.readMaskType(ims)

        if hasFloats:
            imsCorrectDim = np.stack((ims,floats))
        else:
            imsCorrectDim = np.expand_dims(ims,0)

        if self.parent.maskLoaded:
            self.maxZ+=1
            self.parent.maskZ +=1
            self.maxTs.append(ims.shape[-3]-1)
            self.yVals.append(ims.shape[-2])
            self.xVals.append(ims.shape[-1])
            self.isDummys.append(False)
            self.channels.append(imsCorrectDim)
            newContours = contours if isinstance(contours, np.ndarray) else np.array([get_mask_contour(im) for im in tqdm(ims)])
            self.contours.append(newContours)
        else:
            self.removeDummys()
            self.parent.maskZ = 0
            self.maxZ = 0
            self.maxTs, self.yVals,self.xVals = [ims.shape[-3]-1], [ims.shape[-2]], [ims.shape[-1]]
            self.isDummy = False
            self.channels.append(imsCorrectDim)
            print("loading contours")
            newContours = contours if isinstance(contours, np.ndarray) else np.array([get_mask_contour(im) for im in tqdm(ims)])
            self.contours.append(newContours)
            

    def addDummyZ(self,ims):
        if len(self.channels)>=1:
            self.maxZ+=1
        self.isDummys.append(True)
        if self.maxZ==0:
            self.maxTs, self.yVals,self.xVals = [ims.shape[-3]-1], [ims.shape[-2]], [ims.shape[-1]]
        elif self.parent.maskZ>0:
            self.maxTs.append(ims.shape[-3]-1)
            self.yVals.append(ims.shape[-2])
            self.xVals.append(ims.shape[-1])
        ims = np.expand_dims(ims,0)
        self.channels.append(ims)
        self.contours.append(ims)
    
    def insert(self, newIms, startIdx, z, newProbIms = None):
        if type(newIms) is not list:
            newIms = [im for im in newIms]
        self.channels[z][0, startIdx:startIdx+len(newIms), :,:] = np.array(newIms)
        self.contours[z][startIdx:startIdx+len(newIms), :,:] = (np.array([get_mask_contour(im) for im in newIms]))

        if newProbIms is not None:
            self.channels[z][1, startIdx:startIdx+len(newIms), :,:] = newProbIms

    
    def addToZ(self,newIms, z, newProbIms = None):
        if type(newIms) is not list:
            newIms = [im for im in newIms]
        x,y,_ = newIms[0].shape
        if x != self.xVals[z] and y!=self.yVals[z]:
            print(f"ERROR: new image size ({x},{y}) does not match current z size of ({self.xVals[z]}, {self.yVals[z]})")
            raise ValueError
        numImages = len(newIms)
        self.maxTs[z]+=numImages
        self.channels[z][0,self.maxT+1:self.maxT+1+numImages,:,:] = newIms
        self.contours[z][self.maxT+1:self.maxT+1+numImages,:,:] = (np.array([get_mask_contour(im) for im in newIms]))

        if newProbIms:
            self.channels[z][1,self.maxT+1:self.maxT+1+numImages,:,:] = newProbIms
    
    def loadFileList(self, files):
        _, tail = os.path.split(files[0])
        ims = np.array([imread(im) for im in files])
        self.parent.loadMasks(ims, name = tail.split(".")[0])

    
    @property
    def maxT(self):
        return self.maxTs[self.parent.maskZ]
    @maxT.setter
    def maxT(self, num):
        self._maxT = num

    @property
    def x(self):
        return self.xVals[self.parent.maskZ]
    @x.setter
    def x(self, num):
        self._x = num
    
    @property
    def y(self):

        return self.yVals[self.parent.maskZ]
    @y.setter
    def y(self, num):
        self._y = num





    


    



            

    


