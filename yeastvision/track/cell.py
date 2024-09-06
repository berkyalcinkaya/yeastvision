from skimage.measure import regionprops, regionprops_table
from skimage.measure import shannon_entropy as shentr
from skimage.measure import label
import numpy as np
import pandas as pd
from yeastvision.utils import normalize_im
from tqdm import tqdm
from typing import List

def average_axis_lengths(object_array):
    """
    Given a 2D NumPy array with labeled objects, compute the average axis_major_length
    and axis_minor_length for all objects.
    
    Parameters:
    object_array (np.ndarray): A 2D NumPy array where each object is labeled with a unique integer.
    
    Returns:
    tuple: A tuple containing the average major axis length and average minor axis length.
    """
    
    # Get region properties for each labeled object
    labeled_objects = label(object_array)
    regions = regionprops(labeled_objects)
    
    # Collect the major and minor axis lengths for each object
    major_lengths = [region.axis_major_length for region in regions]
    minor_lengths = [region.axis_minor_length for region in regions]
    
    # Calculate the average major and minor axis lengths
    avg_major_length = np.mean(major_lengths) if major_lengths else 0
    avg_minor_length = np.mean(minor_lengths) if minor_lengths else 0
    
    return avg_major_length, avg_minor_length

def getBirthFrame(trackedMasks, cellVal):
    return np.min((np.nonzero(trackedMasks == cellVal))[0])

def getDeathFrame(trackedMasks, cellVal):
    return np.max((np.nonzero(trackedMasks == cellVal))[0])

def entropy(label, intensity_im):
    return shentr(intensity_im, base = 2)

def var(label, intensity_im):
    return np.var(intensity_im)

def median(label, intensity_im):
    return np.median(intensity_im)

def intensity_total(label, intensity_im):
    return np.sum(intensity_im)

def concentration(label, intensity_im):
    return intensity_total(label, intensity_im)/np.sum(label)

LABEL_PROPS = ["area", "eccentricity", "label", ]
IM_PROPS = ["intensity_mean"]
EXTRA_IM_PROPS = [var, intensity_total, concentration]

def getLifeData(labels):
    lifeData = {"birth":[], "death":[]}
    for val in tqdm(np.unique(labels)[1:]):
        lifeData["birth"].append(getBirthFrame(labels, val))
        lifeData["death"].append(getDeathFrame(labels, val))
    return lifeData

def getPropDf(data):
    #newData = data.drop[data[data["labels"] == "population"].index]
    try:
        return data.drop(columns = ["labels", "birth", "death"])
    except:
        return data.drop(columns = ["labels"])

def exportCellData(data, lifedata):
    export = {"cell":[], "time":[]}
    propDf = getPropDf(data)
    for column in propDf.columns:
            export[column] = []
    
    for cell in data["labels"]:
        firstT = int(lifedata[lifedata["cell"] == cell]["birth"])
        endT = int(lifedata[lifedata["cell"] == cell]["death"])

        for i in range(firstT, endT+1):
            export["time"].append(i)
            export["cell"].append(cell)

        for props in propDf.columns:
            vals = data[data["labels"] == cell]
            vals = vals[props].tolist()[0][firstT:endT+1]
            for prop in vals:
                export[props].append(prop)
    return pd.DataFrame(data = export)

def getDaughterMatrix(self, lineageDF):
    arraysize = len(lineageDF.index.tolist())
    daughterMatrix = np.zeros((arraysize, arraysize))
    return daughterMatrix

def getPotentialHeatMapNames(data):
    return getPropDf(data).columns


def getHeatMaps(data, chosen):
    heatMaps = {}
    for prop in chosen:
        hm = data[prop].tolist()
        heatMaps[prop] = (np.array(hm))
    return heatMaps

def getCellData(labels: np.ndarray, intensity_ims: List[np.ndarray] = None, intensity_im_names:List[str] = None)->pd.DataFrame:
    '''
    Computes cell data relating to image and morphological features of the given labels. If intensity images
    are provided, then they are used for image feature extraction (ie signal intensity, mean, concentration) for
    the objects in labels. 
    
    Args
        labels (ndarray): a t x h x w array with t timepoints of h x w images containing tracked objects, identified
                        by unique indeces. Any unsigned integer type is acceptable (uint8, uint16, uint32, etc)
        intensity_ims (optional, list of ndarrays): a list of images, each of which must have the same shape as `labels`. Each will
                                                    be used to compute pixel features for each object in `labels`. Each 3D time series from
                                                    this list be normalized to the interval [0,1]
        inensity_im_names (list of strings, optional): the string identifiers of each element of `intensity_ims`. Used
                                                    to identify each pixel feature in the resulting celldata DataFrame. If none,
                                                    channel1, channel2,...,channeln is the default naming scheme
    Returns:
        pd.DataFrame - a container storing the cell properties (columns) over time for given a object index in labels (rows). Each
                        entry is a t-vector containing time series for all cell-property pairs

    '''
    total_labels = np.unique(labels) # all indeces of cells present in the movie
    total_labels = total_labels[total_labels!=0]

    total_data = {"labels": total_labels} #return value, dictionary storing cell data, to be converted to pd.df
                                          # Will have keys representing properties and values as list of lists (ie list of cell
                                         # timeseries)
    
    # add birth and death data
    total_data = total_data #| getLifeData(labels)

    # determine whether or not to include image properties in call to regionprops_table
    if intensity_ims:
        intensity_ims = [[normalize_im(im) for im in intensity_im] for intensity_im in intensity_ims]
        props = LABEL_PROPS + IM_PROPS
        extra_properties = EXTRA_IM_PROPS
    else:
        props = LABEL_PROPS
        extra_properties = None

    # COMPUTE IMAGE DATA PER TIMEPOINT and reformat into output dictioary
    # Prepare data into a format for skimage.measure.regionprops_table
    for i in tqdm(range(labels.shape[0])):
        label = labels[i,:,:]

        # The intensity_im input to regionprops must be extracted stacked on the third dimensios
        if intensity_ims:
            if len(intensity_ims)>1:
                intensity_im = np.stack([im[i] for im in intensity_ims], -1)
            else:
                intensity_im = intensity_ims[0][i]
        else:
            intensity_im = None
        
        # get imagedata
        im_data = regionprops_table(label, intensity_im, properties = props, extra_properties = extra_properties)
        
        # CELL BIRTH AND DEATH CORRECTIONN: if a label is missing at a given timepoint,
        # then we must insert a None into im_data at the location of 
        # add zeros for cells not present in im data
        for label in total_labels:
            if label not in im_data["label"]:
                # an index is not present at a certain timepoit
                for property in im_data:
                    vals = [round(val, 3) if val is not None else None for val in list(im_data[property])]
                    im_data[property] = vals

                    if property != "label":
                        # prevent index out of bound error
                        if label-1 < len(im_data[property]):
                            im_data[property][label-1] = None
                        else:
                            im_data[property].append(None)
        
        # for every property that isnt "labels" in im_data, we store a list of lists
        # into total data under the key representing the corresponding property
        for property in im_data:
            if property != "label":

                # check if key exists to determine insertion format 
                if property not in total_data:
                    total_data[property] = [[prop] for prop in im_data[property]]
                else:
                    for label in total_labels:
                        total_data[property][label-1].append(im_data[property][label-1])
    
    # Get intensity image with correct names
    if intensity_im_names:
        # determine which properteis are pixel features
        for property in im_data:

            # determine if property is a pixel (image) feature
            isImProp = False
            for prop in IM_PROPS+EXTRA_IM_PROPS:
                try:
                    prop = prop.__name__
                except AttributeError:
                    prop = str(prop)
                
                if prop in property:
                    isImProp = True
                    break
            # found an image prop, break the loop and exit
            if isImProp:
                if "-" in property:
                    idx = int(property.split("-")[-1])-1
                else:
                    idx = 0
                # swap new name, be sure to delete entry under old name
                newPropertyName = "-".join([intensity_im_names[idx], prop])
                total_data[newPropertyName] = total_data[property]
                del total_data[property]

    # get population average for each property
    for property in total_data:
        if property not in ["labels", "birth", "death"]:
            property_array = np.array(total_data[property], dtype = np.float32) # rows are cells (axis 0), columns are time (axis 1)
            property_array[property_array == 0.0] = np.nan
            total_data[property].append(list(np.nanmean(property_array, axis = 0)))
        elif property == "labels":
            total_data["labels"] = list(total_data["labels"])
            total_data["labels"].append("population")
        else:
            total_data[property].append(0)

    df = pd.DataFrame.from_dict(total_data)
    return df
