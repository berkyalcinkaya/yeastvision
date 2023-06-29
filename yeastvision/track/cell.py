from skimage.measure import regionprops, regionprops_table
from skimage.measure import shannon_entropy as shentr
import numpy as np
import pandas as pd
from yeastvision.utils import normalize_im
from tqdm import tqdm

class Cell():
    def __init__(self, mask, num):
        self.cell = (mask == num)
        self.centroid = self.getCentroid()

    def getCentroid(self):
        return regionprops(self.cell.astype(np.uint8))[0].centroid

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
EXTRA_IM_PROPS = [ var, intensity_total, concentration]

def getLifeData(labels):
    lifeData = {"birth":[], "death":[]}
    for val in tqdm(np.unique(labels)[1:]):
        lifeData["birth"].append(getBirthFrame(labels, val))
        lifeData["death"].append(getDeathFrame(labels, val))
    return lifeData

def getPropDf(data):
    #newData = data.drop[data[data["labels"] == "population"].index]
    return data.drop(columns = ["labels", "birth", "death"])

def exportCellData(data):
    export = {"cell":[], "time":[]}
    propDf = getPropDf(data)
    for column in propDf.columns:
            export[column] = []
    
    for cell in data["labels"]:
        firstT = int(data[data["labels"] == cell]["birth"])
        endT = int(data[data["labels"] == cell]["death"])

        for i in range(firstT, endT+1):
            export["time"].append(i)
            export["cell"].append(cell)

        for props in propDf.columns:
            vals = data[data["labels"] == cell][props].to_list()[firstT:endT+1]
            for prop in vals:
                export[props].append(prop)

    return pd.DataFrame(data = export)

def getDaughterMatrix(self, lineageDF):
    arraysize = len(lineageDF.index.tolist())
    daughterMatrix = np.zeros((arraysize, arraysize))
    return daughterMatrix

def getHeatMaps(data):
    heatMaps = []
    for prop in getPropDf(data).columns:
        heatMaps.append(np.array(data[prop].to_list()))
    return np.array(heatMaps)

def getCellData(labels, intensity_ims = None, intensity_im_names = None):
    total_labels = np.unique(labels) # all indeces of cells present in the movie
    total_labels = total_labels[total_labels!=0]
    total_data = {"labels": total_labels} # dictionary storing cell data, to be converted to pd.df
    
    # add birth and death data
    total_data = total_data #| getLifeData(labels)

    # determine whether or not to include image properties in call to regionprops_table
    if intensity_ims:
        intensity_ims = [normalize_im(im) for im in intensity_ims]
        props = LABEL_PROPS + IM_PROPS
        extra_properties = EXTRA_IM_PROPS
    else:
        props = LABEL_PROPS
        extra_properties = None

    # iterate through each time point in given labels (masks)
    for i in tqdm(range(labels.shape[0])):
        label = labels[i,:,:]

        if intensity_ims:
            if len(intensity_ims)>1:
                intensity_im = np.stack([im[i] for im in intensity_ims], -1)
            else:
                intensity_im = intensity_ims[0][i]
        else:
            intensity_im = None
        
        im_data = regionprops_table(label, intensity_im, properties = props, extra_properties = extra_properties)
        
        # add zeros for cells not present in im data
        for label in total_labels:
            if label not in im_data["label"]:
                for property in im_data:
                    vals = [round(val, 3) if val is not None else None for val in list(im_data[property])]
                    im_data[property] = vals

                    if property != "label":
                        # prevent index out of bound error
                        if label-1 < len(im_data[property]):
                            im_data[property][label-1] = None
                        else:
                            im_data[property].append(None)
        
        for property in im_data:
            if property != "label":
                if property not in total_data:
                    total_data[property] = [[prop] for prop in im_data[property]]
                else:
                    for label in total_labels:
                        total_data[property][label-1].append(im_data[property][label-1])
        
    if intensity_im_names:
        for property in im_data:
            isImProp = False
            for prop in IM_PROPS+EXTRA_IM_PROPS:
                try:
                    prop = prop.__name__
                except AttributeError:
                    prop = str(prop)

                if prop in property:
                    isImProp = True
                    break
            if isImProp:
                if "-" in property:
                    idx = int(property.split("-")[-1])-1
                else:
                    idx = 0
                newPropertyName = "-".join([intensity_im_names[idx], prop])
                total_data[newPropertyName] = total_data[property]
                del total_data[property]


                

    
    # # get population average for each property
    # for property in total_data:
    #     if property not in ["labels", "birth", "death"]:
    #         property_array = np.array(total_data[property], dtype = np.float32) # rows are cells (axis 0), columns are time (axis 1)
    #         property_array[property_array == 0.0] = np.nan
    #         total_data[property].append(list(np.nanmean(property_array, axis = 0)))
    #     elif property == "labels":
    #         total_data["labels"] = list(total_data["labels"])
    #         total_data["labels"].append("population")
    #     else:
    #         total_data[property].append(0)

    df = pd.DataFrame.from_dict(total_data)
    return df
