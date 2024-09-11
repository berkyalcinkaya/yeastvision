from yeastvision.track.data import LineageData, TimeSeriesData
from .utils import *
from yeastvision.utils import get_mask_contour
import os 
from skimage.io import imread
import numpy as np
from tqdm import tqdm
import logging
from typing import List

logger = logging.getLogger(__name__)


class Experiment():

    MASK_KEYWORD = "_mask"

    def __init__(self, dir, num_channels = 1, ims_only = False, v = False):
        self.id = get_id()
        self.dir = dir
        _, self.name = os.path.split(self.dir)
        self.num_channels = num_channels
        self.num_labels = 0

        self.channels = []
        self.labels = []

        if v:
            print("Loading experiment from", self.dir, "with", num_channels, "channels")

        im_npzs, mask_npzs = get_im_mask_npzs(self.dir)
        
        if v:
            print("Found", len(im_npzs), "image .npz files")
            print(im_npzs)
            print()
            print("Found", len(mask_npzs), "mask .npz files")
            print(mask_npzs)
            
        if (not ims_only) and mask_npzs:
            self.load_with_mask_npzs(mask_npzs, num_channels)
        else:
            if v:
                print()
                print("no mask npzs detected. Attempting to load from directory", self.dir)
            self.load_from_dir(num_channels, ims_only=ims_only, v = v)
        
        if im_npzs:
            self.load_channels_from_npz_files(im_npzs)
        
        self.set_mask_type("labels")
    
    def get_channnels_from_fname_list(self, filenames: List[str]):
        return [fname for fname in filenames if self.MASK_KEYWORD not in fname]

    def get_masks_from_fname_list(self, filenames: List[str]):
        return [fname for fname in filenames if self.MASK_KEYWORD in fname]

    def new_channel_name(self, index, new_name):
        if new_name not in self.get_channel_names():
            self.channels[index].set_name(new_name)
            return True
        else:
            return False

    def new_label_name(self, index, new_name):
        if new_name not in self.get_channel_names():
            self.labels[index].set_name(new_name)
            return True
        else:
            return False
    
    def load_channels_from_npz_files(self, npz_channel_files):
        self.npz_channel_files = npz_channel_files
        for path in self.npz_channel_files:
            id = np.load(path)["id"].item()
            if InterpolatedChannel.text_id in id:
                self.add_channel_object(InterpolatedChannel(npz_path=path))
            else:
                self.add_channel_object(ChannelNoDirectory(npz_path=path))
    
    def get_channel_names(self):
        return [channel.name for channel in self.channels]
    
    def get_label_names(self):
        return [label.name for label in self.labels]
        
    def get_dummy_labels(self, full = True):
        shape = self.shape()
        if full:
            h,w = shape
            t = self.get_t()
            return np.zeros((t,h,w), dtype = np.uint8)
        else:
            return np.zeros(shape, dtype = np.uint8)
    
    def get_label(self, mask_type, id=None, name = None, idx = None, t = None):
        if not self.has_labels():
            return self.get_dummy_labels(full = t is None)
        else:
            if mask_type!= self.mask_type:
                self.set_mask_type(mask_type)
            
            if id is not None:
                idx = self.get_mask_index_from_id(id)
            elif name:
                idx = self.get_mask_index_from_name(name)
            
            if t is not None:
                return self.labels[idx].data[t,:,:]
            else:
                return self.labels[idx].data[:,:,:]
    
    def get_channel_by_name(self, name):
        for channel in self.channels:
            if channel.name == name:
                return channel
        raise IndexError


    def get_channel_by_id(self, id):
        for channel in self.channels:
            if channel.id == id:
                return channel
        raise IndexError

    def get_label_by_name(self,name):
        for label in self.labels:
            if label.name == name:
                return label
        raise IndexError
    
    def get_channel(self, id=None, name = None, idx = None, t = None):
        if id is not None:
            channel = self.get_channel_by_id(id)
        elif name:
            channel = self.get_channel_by_name(name)
        else:
            channel = self.channels[idx]
        if t is not None:
            return channel.ims[t]
        else:
            return channel.ims
        
    def get_mask_index_from_name(self, name):
        for i,label in enumerate(self.labels):
            if label.name == name:
                return i
        raise IndexError


    def get_mask_index_from_id(self, id):
        for i,label in enumerate(self.labels):
            if label.id == id:
                return i
        raise IndexError

    def set_mask_type(self, mask_type):
        self.mask_type = mask_type
        for label in self.labels:
            label.set_mask_type(mask_type)
    
    def has_labels(self):
        return self.num_labels>0

    def load_from_dir(self, num_channels, ims_only=False, v = False):
        channels = [[] for i in range(num_channels)]
        
        files_image_only = sorted(get_image_files(self.dir, remove_masks=True))
        
        if len(files_image_only) == 0:
            print("DIRECTORY WITH ONLY MASKS DETECTED")
            raise FileNotFoundError
        
        files = sorted(get_image_files(self.dir))
        
        if ims_only:
            files = files_image_only
        
        if v:
            print("Found the following image files (anything with _mask remove):")
            print(files_image_only)
            print()
            print("All files (images and masks included)")
            print(files)

        num_images = len(files_image_only)/num_channels
        num_labels = int(len(files)/num_images - num_channels)

        labels = [[] for i  in range(int(num_labels))]

        groupsize = num_channels+num_labels
        
        for i in range(0, len(files), groupsize):
            group = files[i:i+groupsize]
            
            channels_in_group = self.get_channnels_from_fname_list(group)
            labels_group = self.get_masks_from_fname_list(group)
        
            for channel, channel_in_group in zip(channels, channels_in_group):
                channel.append(channel_in_group)
            for label, label_in_group in zip(labels, labels_group):
                label.append(label_in_group)
        
        for channel in channels:
            self.add_channel(channel, increment_count=False)
        
        if num_labels>0:
            for label in labels:
                self.add_label(files = label, increment_count=True)

    def load_with_mask_npzs(self, mask_npzs, num_channels):
        channels = [[] for i in range(num_channels)]
        files_image_only = sorted(get_image_files(self.dir, remove_masks=True))
        if len(files_image_only) == 0:
            print("DIRECTORY WITH ONLY MASKS DETECTED")
            raise FileNotFoundError
        
        files = sorted(get_image_files(self.dir))
        num_images = len(files_image_only)/num_channels
        num_labels = int(len(files)/num_images - num_channels)
        labels = [[] for i  in range(int(num_labels))]
        groupsize = num_channels+num_labels
        
        for i in range(0, len(files), groupsize):
            group = files[i:i+groupsize]
            
            channels_in_group = self.get_channnels_from_fname_list(group)
            labels_group = self.get_masks_from_fname_list(group)
        
            for channel, channel_in_group in zip(channels, channels_in_group):
                channel.append(channel_in_group)
            for label, label_in_group in zip(labels, labels_group):
                label.append(label_in_group)
        
        for channel in channels:
            self.add_channel(channel, increment_count=False)
        
        if num_labels>0:
            for label in labels:
                would_be_id = get_id_from_name(sorted(label)[0])
                if not any([np.load(npz_path)["id"] == would_be_id for npz_path in mask_npzs]):         
                    self.add_label(files = label, increment_count=True)
                
        for npz_path in tqdm(mask_npzs):
            self.labels.append(Label(npz_path=npz_path))
            self.num_labels+=1

    def create_label_objects_from_npz(self):
        self.npzdata = np.load(self.npz_path)
        loaded_npz = self.npzdata
        num_masks = loaded_npz["labels"].shape[0]
        for i in range(num_masks):
            self.labels.append(Label(dir = self.dir, name = self.get_label_name(name = None), npz_name=self.npz_filename, loaded_npz=loaded_npz, npz_index=i))
            self.num_labels+=1
    
    def add_channel_object(self, channel_object):
        self.channels.append(channel_object)
        self.num_channels+=1
    
    def add_label_object(self, label_object):
        self.labels.append(label_object)
        self.num_labels+=1

    def add_channel(self, files, increment_count = True):
        self.channels.append(Channel(files, f"channel_{len(self.channels)}"))

        if increment_count:
            self.num_channels+=1
    
    def delete_channel(self, index):
        self.num_channels -= 1
        self.channels[index].delete()
        del self.channels[index]
    
    def delete_label(self, index):
        self.num_labels -= 1
        self.labels[index].delete()
        del self.labels[index]

    def add_label(self, files = None, arrays = None, increment_count = True, name = None, update_data = False):
        new_label = Label(fnames = files, mask_arrays=arrays, dir = self.dir, name = self.get_label_name(name = name))
        self.labels.append(new_label)
        if increment_count:
            self.num_labels +=1
        if update_data:
            self.load_masks()
        
    def get_num_labels(self):
        return len(self.labels)

    def get_num_channels(self):
        return len(self.channels)

    def get_t(self):
        return len(self.channels[0].ims)
    
    def get_label_name(self, name = None):
        if name is None:
            name =  f"label_{len(self.labels)}"
        n = (np.char.find(np.array(self.get_channel_names()), name)+1).sum()
        if n>0:
            return f"{name}-n"
        else:
            return name
    
    def get_channel_name(self, name = None):
        if name is None:
            return f"channel_{len(self.labels)}"
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

    def get_viable_im_objs(self, mask_obj):
        viable_ims = []
        for channel in self.channels:
            if channel.t == mask_obj.t and channel.shape == mask_obj.shape:
                viable_ims.append(channel)
        return viable_ims
    
    def get_timeseries(self):
        return [channel for channel in self.channels if channel.max_t() > 0]

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
        self.files = sorted(fnames)
        self.dir = os.path.dirname(self.files[0])
        self.ims = None
        self.loaded = False
        self.load()
        self.get_properties()
        self.annotations = None
        self.id = get_id_from_name(self.files[0])

    def load(self):
        self.ims = [imread(file) for file in self.files]
        self.loaded = True
    
    def deload(self):
        del self.ims
        self.ims = None
        self.loaded = False
    
    def get_properties(self):
        self.t = len(self.ims)
        self.shape = self.ims[0].shape
        self.dtype = str(self.ims[0].dtype)
    
    
    def get_string(self, t):
        return f"{self.shape} {self.dtype}"
    
    def get_files(self, t):
        if t > self.max_t():
            return ""
        return self.files[t]

    def max_t(self):
        return self.t-1

    def set_name(self, new_name):
        self.name = new_name
    
    def x(self):
        return self.shape[-1]

    def y(self):
        return self.shape[0]

    def get_file_names(self):
        return [get_file_name(file) for file in self.files]

    def save(self):
        pass


class Channel(Files):
    def __init__(self, fnames, name, labels = None, load_on_init = True):
        super().__init__(fnames, load_on_init=load_on_init)
        self.labels = labels
        self.name = name
        self.compute_saturation()

    def __str__(self):
        return f"{self.shape} | {self.dtype}"
    
    def compute_saturation(self):
        self.saturation = [[im.min(), im.max()] for im in self.ims]
    
    def get_saturation(self, t):
        return self.saturation[t]
    
    def delete(self):
        return
    

class ChannelNoDirectory(Channel):
    def __init__(self, npz_path = None, ims = None, dir = None, name = None, annotations = None):
        if npz_path is not None:
            self.dir, fname = os.path.split(npz_path)
            self.name = os.path.splitext(fname)[0]
            data = np.load(npz_path, allow_pickle=True)
            self.annotations = data["annotations"]
            self.ims = data["ims"]
            self.id = data["id"]
            self.path = npz_path
        else:
            self.dir = dir
            self.name = name 
            self.ims = ims
            self.path = self.make_path()
            if annotations is None:
                annotations = [[] for _ in range(len(ims))]
            self.annotations = np.array(annotations, dtype = object)
            self.id = get_id()
            self.save_to_npz()
        
        self.get_properties()
        self.compute_saturation()
        self.files = [self.path]

    def make_path(self):
        return os.path.join(self.dir, self.name+".npz")

    def set_name(self, new_name):
        self.name = new_name
        new_npz_name = self.make_path()
        os.rename(self.path, new_npz_name)
        self.path = new_npz_name

    def save_to_npz(self):
        np.savez(self.path, annotations=self.annotations, ims=self.ims, id=self.id)
    
    def get_string(self,t):
        if self.annotations is not None:
            return f"{super().get_string(t)}, {(','.join(self.annotations[t])).upper()}"
        else:
            return super().get_string(t)

    def get_files(self, t):
        return self.path
    
    def get_file_names(self):
        return [get_file_name(self.path)]

    def delete(self):
        logger.info(f"Removing channel data file {self.path}")
        os.remove(self.path)
    
class InterpolatedChannel(ChannelNoDirectory):
    '''
    Stores interpolated images and associated information for front-end display and interaction. Also contains the
    class method 'insert_interpolated_values' which is used by yeastvision.ims.interpolate.get_interp_labels to produce
    masks to determine what images are interpolated
    
    Important fields:
    - new_interval (self.interp_intervals) specifies the location of interpolation with the new interpolated movie as reference, used for visualization purposes
    - intervals (self.interval_annotations) stores the interpolation intervals with reference to the original movie, served as input to
                                            yeastvision.ims.interpolate.interpolate_intervals
    - self.interp_annotations (List of booleans) is the same length as the image data and gives the location of interpolated frames with a true value, false for
                            real images.
    '''
    text_id = "interp"
    def __init__(self, npz_path = None, ims = None, dir = None, name = None, 
                 annotations = None, intervals = None, original_len = None,
                 new_intervals = None):
        
        if npz_path is not None:
            self.dir, fname = os.path.split(npz_path)
            self.name = os.path.splitext(fname)[0]
            data = np.load(npz_path, allow_pickle=True)
            self.id = data["id"]
            self.annotations = data["annotations"]
            self.ims = data["ims"]
            self.interp_annotations = data["interpolation"]
            self.interval_annotations = data["intervals"]
            self.interp_intervals = data["interp_intervals"]
            self.path = npz_path
        else:
            self.id = f"{self.text_id}_{get_id()}" # we distinguish interpolated channels with their id
            self.interval_annotations = intervals
            self.dir = dir
            self.name = name 
            self.ims = ims
            self.original_len = original_len
            self.path = os.path.join(self.dir, self.name+".npz")
            self.interp_intervals = new_intervals

            if annotations is None:
                self.annotations = [["REAL-IMAGE"] for i in range(self.original_len)]
            else:
                self.annotations = annotations
            self.extend_annotations()
            self.annotations = np.array(self.annotations)
            self.make_interpolation_annotation()
            self.save_to_npz()

        self.get_properties()
        self.compute_saturation()
    
    @staticmethod
    def insert_interpolated_values(original_list, intervals, val):
        """
        Insert a specified value at locations where interpolated frames exist.

        Parameters:
        original_list (list): The original list of length t.
        intervals (list): List of interval dictionaries with 'start', 'stop', and 'interp' keys.
        val: The value to insert at interpolated locations.

        Returns:
        list: Updated list with interpolated values inserted.
        """
        new_list = []
        current_position = 0

        for i in range(len(original_list)):
            new_list.append(original_list[i])
            current_position += 1
            
            for interval in intervals:
                start = interval['start']
                stop = interval['stop']
                interp_factor_log2 = int(interval['interp'])
                interp_factor = int(2 ** interp_factor_log2)

                if start <= i < stop:
                    num_interpolated_frames = interp_factor - 1
                    
                    for _ in range(num_interpolated_frames):
                        new_list.insert(current_position, val)
                        current_position += 1

        return new_list
    
    def extend_annotations(self):
        if not isinstance(self.annotations, list):
            self.annotations = self.annotations.tolist()
        self.annotations = InterpolatedChannel.insert_interpolated_values(self.annotations, self.interval_annotations, ["INTERP"])

    def save_to_npz(self):
        np.savez(self.path, 
                id=self.id,
                annotations = self.annotations, 
                ims = self.ims, 
                interpolation = self.interp_annotations,
                intervals = self.interval_annotations,
                interp_intervals = self.interp_intervals)
    
    def make_interpolation_annotation(self):
        self.interp_annotations = [False for i in range(self.original_len)]
        self.interp_annotations = InterpolatedChannel.insert_interpolated_values(self.interp_annotations, self.interval_annotations, True)
    
class Label(Files):
        mask_types = ["labels", "probability", "contours", "flows"]
        def __init__(self, mask_arrays = None, fnames = None, dir = None, npz_path = None, name = None):
            
            self.celldata = None
            self.lineagedata = None
            self.labels, self.contours, self.probability, self.flows = None, None, None, None

            if npz_path and os.path.exists(npz_path):
                self.npz_path = npz_path
                self.get_dir_and_name_from_npz_path()
                self.init_from_npz()

            else:   
                self.name = name           
                if fnames is not None:
                    self.id = get_id_from_name(fnames[0])
                    self.files = fnames
                    self.dir = os.path.dirname(self.files[0])
                    raw_arrays = np.array([imread(file) for file in self.files])
                elif mask_arrays is not None:
                    assert(mask_arrays is not None and dir is not None)
                    self.id = get_id()
                    self.dir = dir
                    self.files = []
                    raw_arrays = np.array(mask_arrays)            
                self.npz_path = os.path.join(self.dir, name+".npz")
                self.extract_probability_labels_flows(raw_arrays)
                self.extract_contours()
                self.get_properties()

                self.write_to_npz()
                self.unload_data_attrs()
            
            self.load()

        def has_cell_data(self):
            return isinstance(self.celldata, TimeSeriesData) and self.celldata.has_cell_data()
        
        def has_lineage_data(self):
            return isinstance(self.celldata, LineageData)
        
        def get_dir_and_name_from_npz_path(self):
            self.dir, head = os.path.split(self.npz_path)
            self.name,_ = os.path.splitext(head)

        def load(self, mask_type = "labels"):
            self.npzdata = dict(np.load(self.npz_path, allow_pickle=True))
            self.data = self.npzdata[mask_type]
        
        def set_mask_type(self, mask_type):
            assert mask_type in self.mask_types
            self.mask_type = mask_type
            self.data = self.npzdata[self.mask_type]
        
        def save_current_data(self):
            if self.mask_type == "contours":
                self.contours = self.data
            elif self.mask_type == "labels":
                self.labels = self.data
            self.save()

        def init_from_npz(self):
            data = np.load(self.npz_path, allow_pickle=True)
            self.id = data["id"]
            self.celldata = data["celldata"][0]
            self.lineagedata = data["lineagedata"][0]
            labels = data["labels"][:,:,:]
            self.shape = labels[0].shape
            self.dtype = str(labels[0].dtype)
            self.t = len(labels)
            self.has_probability = bool(np.any(data["probability"][:,:,:]!=0))
            self.has_flows = bool(np.any(data["flows"][:,:,:]!=0))
            del labels
            del data

        def get_properties(self):
            self.t = len(self.labels)
            self.shape = self.labels[0].shape
            self.dtype = str(self.labels[0].dtype)
        
        def __str__(self):
            return f"{self.shape} | {self.dtype} | has_probability_and_flows: {self.has_probability and self.has_flows}"

        def write_to_npz(self):
            np.savez(self.npz_path, **self.get_data_dict())

        def unload_data_attrs(self):
            self.labels = None
            self.contours = None
            self.probability = None
            self.flows = None

        def get_data_dict(self):
            data = {}
            data["id"] = self.id
            for property_name in ["labels", "contours", "probability", "flows"]:
                data[property_name] = getattr(self,f"get_{property_name}")()
            data["celldata"] = np.array([self.celldata])
            data["lineagedata"] = np.array([self.lineagedata])
            return data
        
        def insert(self, new_mask, t, new_prob_im = None, new_flows_im = None):
            self.npzdata["labels"][t] = new_mask
            self.npzdata["contours"][t]  = get_mask_contour(new_mask)
            if self.has_probability:
                self.npzdata["probability"][t] = new_prob_im
                self.npzdata["flows"][t] = new_flows_im
            self.save()

        def extract_probability_labels_flows(self, raw_arrays):
            if len(raw_arrays.shape) == 4:
                self.has_probability = True
                self.has_flows = True
                self.labels = raw_arrays[0,:,:,:]
                self.probability = raw_arrays[1,:,:,:]
                self.flows = raw_arrays[2,:,:,:]
            else:
                self.has_flows = False
                self.has_probability = False
                self.labels = raw_arrays
            
        def extract_contours(self):
            self.contours = [get_mask_contour(mask) for mask in tqdm(self.labels)]
        
        def get_labels(self):
            if self.labels is not None:
                return self.labels
            return self.npzdata["labels"]
        
        def get_contours(self):
            if self.contours is not None:
                return self.contours
            return self.npzdata["contours"]

        def get_probability(self):
            if self.has_probability:
                    if self.probability is not None:
                        return self.probability
                    return self.npzdata["probability"]
            else:
                return np.zeros_like(self.get_labels(), dtype = np.uint8)
        
        def get_flows(self):
            if self.has_flows:
                if self.flows is not None:
                    return self.flows
                return self.npzdata["flows"]
            else:
                return np.zeros_like(self.get_labels(), dtype = np.uint8)
        
        def get_files(self,t):
            return self.npz_path
        
        def set_name(self, new_name):
            self.name = new_name
            new_npz_name = self.make_path()
            os.rename(self.npz_path, new_npz_name)
            self.npz_path = new_npz_name
        
        def make_path(self):
            return os.path.join(self.dir, self.name+".npz")
        
        def save(self):
            self.write_to_npz()
        
        def delete(self):
            logger.info(f"Removing label data file {self.npz_path}")
            os.remove(self.npz_path)

        def set_data(self, labels, contours, probability = None, flows = None):
            self.labels = labels
            self.contours = contours
            if probability is not None and flows is not None:
                self.has_probability = True
                self.probability = self.probability
                self.flows = flows
                self.has_flows = True
            self.get_properties()
            self.save()
            self.unload_data_attrs()
            self.load()
        
        def set_contours(self, contours):
            self.contours = contours
            self.save()
            self.unload_data_attrs()
