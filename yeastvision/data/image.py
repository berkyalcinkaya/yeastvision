from yeastvision.disk.reader import get_channels_from_dir, get_channel_files_from_dir
import os
import glob
import numpy as np
from skimage.io import imread




class Image():
    '''Convenience class for storing data on a single set of images. Only properties unique to a set of images are stored here including filenames, directory, and the image data itself
    
    Parameters
    ---------
    images: np.ndarray
    files: list[string]
    dir: list[string]'''

    filters: list[str]
    images: np.ndarray
    files: list[str]
    dir: str
    interpolated_ims: np.ndarray

    def __init__(self, files, images = None, name = None):
        self.name = name
        if files is not list:
            files = [files]
        self.files = files
        if images is not None:
            self.images = images
        else:
            images = np.array([imread(file) for file in files])

        self.dir, _ = os.path.split(files[0])

class ImageData():
    '''Encapsulates related images and metadata in a convenience class
    
    Parameters
    ----------
    path: str
        path to a stack tif or directory of images
    
    channels_in_dir: bool (optio nal)
        whether or not multiple channels are present in directory. If true, path arg should be a directory with suffixes that correspond to channels and num_channels should be specified
    
    num_channels: int (optional)
        number of channels to load. Required if channels_in_dir is True. Not required if path loads a stack tif, as number of channels is identified from last axis
    
    channel_names: list[string] (optional)
        names of channels. If not present for a directory path, channel names are taken as what follows last _ in filename. For a stack tif, channel1, channel2, .., channeln is used        
    '''
    def __init__(self, path, channels_in_dir = False, num_channels = None, channel_names = None):
        self.num_channels = num_channels
        self.channels = []

        # channel names
        if channel_names is not None:
            self.channel_names = channel_names
        else:
            self.channels_names = [None for _ in range(num_channels)]

        if channels_in_dir and os.path.isdir(path) and num_channels is not None:
            files, self.channel_names = get_channel_files_from_dir(path, num_channels)
            for channel_num, files in enumerate(self.files):
                self.channels.append(Image(files, name = self.channel_names[channel_num]))
        
        elif os.path.isdir(path) and (not channels_in_dir or num_channels==1):
            self.channel_names.append()
            self.channels.append(Image(glob.glob(os.path.join(path,"*")), name =  None))

        elif type(path) is not list:
            path = [path]

        for p in path:
            pass
    
    def load(self, path):
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*")))
            ims = np.array([imread(im) for im in files])
            self.files = files
            self.dir = path       
        else:
            ims = imread(path)
            if len(ims.shape) == 2:
                ims = np.expand_dims(ims,0)
            self.files.append([path])

        


