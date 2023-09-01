import os 
from skimage.io import imread
import glob
from yeastvision.utils import get_mask_contour
import numpy as np
from .utils import *



class Experiment():

    npz_filename = "labels.npz"

    def __init__(self, dir, num_channels = 1):
        self.dir = dir
        self.num_channels = num_channels
        self.num_labels = 0

        self.channels = []
        self.labels = []

        self.npz_path  = os.path.join(self.dir, self.npz_filename)
        if os.path.exists(self.npz_path):
            self.load_from_npz(num_channels=num_channels)
        else:
            self.load_from_dir(num_channels)
            self.load_masks()
        

    def get_dummy_labels(self, full = True):
        shape = self.shape()
        if full:
            h,w = shape
            t = self.get_t()
            print(t,h,w)
            return np.zeros((t,h,w))
        else:
            return np.zeros(shape)
    
    def get_label(self, mask_type, name = None, idx = None, t = None):
        if not self.has_labels():
            return self.get_dummy_labels(full = t == None)
        else:
            if mask_type!= self.mask_type:
                self.set_mask_type(mask_type)
            if name:
                idx = self.get_mask_index_from_name(name)
            if t:
                return self.data[idx,t,:,:]
            else:
                return self.data[idx,:,:,:]
    
    def get_channel_by_name(self, name):
        for channel in self.channels:
            if channel.name == name:
                return channel
        raise IndexError

    def get_label_by_name(self,name):
        for label in self.labels:
            if label.name == name:
                return label
        raise IndexError
    
    def get_channel(self, name = None, index = None, t = None):
        if name:
            channel = self.get_channel_by_name(name)
        else:
            channel = self.channels[index]
        if t:
            return channel.ims[t,:,:]
        else:
            return channel.ims
        
    def get_mask_index_from_name(self, name):
        for i,label in enumerate(self.labels):
            if label.name == name:
                return i
        raise IndexError

    def set_mask_type(self, mask_type):
        self.mask_type = mask_type
        self.data = self.npzdata[self.mask_type]
        
    def load_masks(self, mask_type = "labels"):
        if self.has_labels():
            self.npzdata = np.load(self.npz_path)
            self.mask_type = mask_type
            self.data = self.npzdata[mask_type]
    
    def has_labels(self):
        return self.num_labels>0

    def load_from_dir(self, num_channels):
        channels = [[] for i in range(num_channels)]
        
        files_image_only = sorted(get_image_files(self.dir, remove_masks=True))
        files = sorted(get_image_files(self.dir))

        num_images = len(files_image_only)/num_channels
        num_labels = int(len(files)/num_images - num_channels)

        labels = [[] for i  in range(int(num_labels))]

        groupsize = num_channels+num_labels
        
        for i in range(0, len(files), groupsize):
            group = files[i:i+groupsize]
            channels_in_group = group[0:num_channels]
            labels_group = group[num_channels:groupsize]
            for channel, channel_in_group in zip(channels, channels_in_group):
                channel.append(channel_in_group)
            for label, label_in_group in zip(labels, labels_group):
                label.append(label_in_group)
        
        for channel in channels:
            self.add_channel(channel, increment_count=False)
        
        if num_labels>0:
            for label in labels:
                self.add_label(files = label, increment_count=False)

    def load_from_npz(self, num_channels):
        channels = [[]*num_channels]
        
        files = sorted(get_image_files(self.dir, remove_masks=True))
        groupsize = num_channels
        
        for i in range(0, len(files), groupsize):
            group = files[i:i+groupsize]
            channels_in_group = group[0:num_channels]
            labels_group = group[num_channels:groupsize]
            for channel, channel_in_group in zip(channels, channels_in_group):
                channel.append(channel_in_group)
        
        for channel in channels:
            self.add_channel(channel, increment_count=False)
        self.load_masks()
        self.create_label_objects_from_npz()
    
    def create_label_objects_from_npz(self):
        loaded_npz = self.npzdata
        num_masks = loaded_npz["labels"].shape[0]
        for i in range(num_masks):
            self.labels.append(Label(dir = self.dir, name = self.get_label_name(None), npz_name=self.npz_filename, loaded_npz=loaded_npz, npz_index=i))
            self.num_labels+=1

    def add_channel(self, files, increment_count = True):
        self.channels.append(Channel(files, f"channel_{len(self.channels)}"))

        if increment_count:
            self.num_channels+=1

    def add_label(self, files = None, arrays = None, increment_count = True, name = None):
        new_label = Label(fnames = files, mask_arrays=arrays, dir = self.dir, npz_name=self.npz_filename, name = self.get_label_name(name))
        self.labels.append(new_label)

        self.num_labels +=1
    
    def get_t(self):
        return len(self.channels[0].ims)
    
    def get_label_name(self, name):
        if name is None:
            return f"label_{len(self.labels)}"
        n = (np.char.find(np.array(self.labels), name)+1).sum()
        if n>0:
            return f"{name}-n"
        else:
            return name
    
    def get_channel_string(self):
        channel_string = ""
        for channel in self.channels:
            channel_string+=f"\t{channel.name}: {str(channel)}\n"
        return channel_string
    
    def get_label_string(self):
        label_string = ""
        for label in self.labels:
            label_string+=f"\t{label.name}: {str(label)}\n"
        return label_string

    def shape(self):
        return self.channels[0].shape
    
    def __str__(self):
        return f"dir: {self.dir}\nt:{self.get_t()}\nnum_channels: {len(self.channels)}\n{self.get_channel_string()}\nnum_labels:{len(self.labels)}\n{self.get_label_string()}"
        


class Files():
    image_extensions = ['.tif','.tiff',
                            '.jpg','.jpeg','.png','.bmp',
                            '.pbm','.pgm','.ppm','.pxm','.pnm','.jp2',
                            '.TIF','.TIFF',
                            '.JPG','.JPEG','.PNG','.BMP',
                            '.PBM','.PGM','.PPM','.PXM','.PNM','.JP2']
    '''
    Lowest level class for storing information about a related set of filenames which should exist in the same directory and should be identifiable by a certain keyword'''
    def __init__(self, fnames, load_on_init = True):
        self.files = fnames
        self.dir = os.path.dirname(self.files[0])
        self.ims = None
        self.loaded = False
        self.load()
        self.get_properties()
    
    def load(self):
        self.ims = [imread(file) for file in self.files]
        self.loaded = True
    
    def deload(self):
        del self.ims
        self.ims = None
        self.loaded = False
    
    def get_properties(self):
        self.shape = self.ims[0].shape
        self.dtype = str(self.ims[0].dtype)


class Channel(Files):
    def __init__(self, fnames, name, labels = None, load_on_init = True):
        super().__init__(fnames, load_on_init=load_on_init)
        self.labels = labels
        self.name = name
    def __str__(self):
        return f"{self.shape} | {self.dtype}"

    

class Label(Files):
        def __init__(self, mask_arrays = None, fnames = None, dir = None, npz_name = "labels.npz", name = None, loaded_npz = None, npz_index = None):
            self.name = name

            if loaded_npz is not None and npz_name is not None and npz_index is not None:
                print("loading from npz")
                self.dir = dir
                self.npz_path = os.path.join(self.dir, npz_name)
                self.load_from_npz(loaded_npz, npz_index)

            else:              
                if fnames is not None:
                    self.files =fnames
                    self.dir = os.path.dirname(self.files[0])
                    raw_arrays = np.array([imread(file) for file in self.files])
                elif mask_arrays is not None:
                    assert(mask_arrays is not None and dir is not None)
                    self.dir = dir
                    self.files = []
                    raw_arrays = np.array(mask_arrays)            
                self.npz_path = os.path.join(self.dir, npz_name)
                self.extract_probability(raw_arrays)
                self.extract_contours()
                self.get_properties()

                self.write_to_npz()
                self.unload()
            
        def load_from_npz(self, loaded_npz, npz_index):
            labels = loaded_npz["labels"][npz_index,:,:,:]
            self.shape = labels[0].shape
            self.dtype = str(labels[0].dtype)
            self.has_probability = np.any(loaded_npz["probability"][npz_index,:,:,:]!=0)

        def get_properties(self):
            self.shape = self.labels[0].shape
            self.dtype = str(self.labels[0].dtype)
        
        def __str__(self):
            return f"{self.shape} | {self.dtype} | has_probability: {self.has_probability}"

        def write_to_npz(self):
            print(self.npz_path)
            if not os.path.exists(self.npz_path):
                np.savez(self.npz_path, **self.get_data_dict())
            else:
                append_to_npz(self.npz_path, **self.get_data_dict())

        def unload(self):
            del self.labels
            del self.contours
            if self.has_probability:
                del self.probability

        def get_data_dict(self):
            data = {}
            for property_name in ["labels", "contours", "probability"]:
                data[property_name] = getattr(self,f"get_{property_name}")()
            return data

        def extract_probability(self, raw_arrays):
            if len(raw_arrays.shape) == 4:
                self.has_probability = True
                self.labels = raw_arrays[:,:,:,0]
                self.probability = raw_arrays[:,:,:,1]
            else:
                self.has_probability = False
                self.labels = raw_arrays
            
        def extract_contours(self):
            self.contours = [get_mask_contour(mask) for mask in self.labels]
        
        def get_labels(self):
            return np.expand_dims(self.labels, 0)
        
        def get_contours(self):
            return np.expand_dims(self.contours, 0)
        
        def get_probability(self):
            if self.has_probability:
                return np.expand_dims(self.probability, 0)
            else:
                return np.expand_dims(np.zeros_like(self.labels, dtype = np.uint8), 0)

            

                