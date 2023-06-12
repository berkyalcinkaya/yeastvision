import numpy as np
import pandas as pd
from munkres import Munkres
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import euclidean_distances
from trackpy import link
import pandas as pd
from skimage.measure import regionprops_table
import matplotlib.pyplot as plt



class AreaCentroidTracker():
    def __init__(self, do_area = False):
        print("Creating Cell Tracking Object")
        self.features = ['com_x', 'com_y']
        if do_area:
            self.features.append("area")
        self.newcell = 0


    def correspondence(self, prev, curr):
        """
        Corrects correspondence between previous and current mask, returns current
        mask with corrected cell values. New cells are given the unique identifier
        starting at max(prev)+1. 
        
        This is done by embedding every cell into a feature space consisting of
        the center of mass and the area. The pairwise euclidean distance is 
        calculated between the cells of the previous and current frame. This is 
        then used as a cost for the bipartite matching problem which is in turn
        solved by the Hungarian algorithm as implemented in the munkres package.
        """
        print("Correspondence")
        prevmax = np.max(prev)
        if prevmax > self.newcell:
            self.newcell = prevmax
        
        hu_dict = self.hungarian_align(prev, curr)
        new = curr.copy()
        for key, val in hu_dict.items():
            # If new cell
            if val == -1:
                val = self.newcell
                self.newcell += 1
            
            new[curr==key] = val
            
        return new


    def hungarian_align(self, m1, m2):
        """
        Aligns the cells using the hungarian algorithm using the euclidean distance as 
        cost. 
        Returns dictionary of cells in m2 to cells in m1. If a cell is new, the dictionary 
        value is -1.
        """
        dist, ix1, ix2 = self.cell_distance(m1, m2)
        
        # If dist couldn't be calculated, return dictionary from cells to themselves 
        if dist is None:
            unique_m2 = np.unique(m2)
            return dict(zip(unique_m2, unique_m2))
        
        solver = Munkres()
        indexes = solver.compute(self.make_square(dist))
        
        # Create dictionary of cell indicies
        d = dict([(ix2.get(i2, -1), ix1.get(i1, -1)) for i1, i2 in indexes])
        d.pop(-1, None)  
        return d


    def cell_to_features(self, im, c, nsamples=None, time=None):
        """Embeds cell c in image im into feature space"""
        coord = np.argwhere(im==c)
        area = coord.shape[0]
        
        if nsamples is not None:
            samples = np.random.choice(area, min(nsamples, area), replace=False)
            sampled = coord[samples,:]
        else:
            sampled = coord
        
        com = sampled.mean(axis=0)
        
        return {'cell': c,
                'time': time,
                'sqrtarea': np.sqrt(area),
                'area': area,
                'com_x': com[0],
                'com_y': com[1]}
        
        
    def cell_distance(self, m1, m2, weight_com=3):
        """
        Gives distance matrix between cells in first and second frame, by embedding
        all cells into the feature space. Currently uses center of mass and area
        as features, with center of mass weighted with factor weight_com (to 
        make it more important).
        """
        # Modify to compute use more computed features
        #cols = ['com_x', 'com_y', 'roundness', 'sqrtarea']

        def get_features(m, t):
            cells = list(np.unique(m))
            if 0 in cells:
                cells.remove(0)
            features = [self.cell_to_features(m, c, time=t) for c in cells]
            return pd.DataFrame(features), dict(enumerate(cells))
        
        # Create df, rescale
        feat1, ix_to_cell1 = get_features(m1, 1)
        feat2, ix_to_cell2 = get_features(m2, 2)
        
        # Check if one of matrices doesn't contain cells
        if len(feat1)==0 or len(feat2)==0:
            return None, None, None
        
        df = pd.concat((feat1, feat2))
        df[self.features] = scale(df[self.features])
        
        # give more importance to center of mass
        df[['com_x', 'com_y']] = df[['com_x', 'com_y']] * weight_com

        # pairwise euclidean dist
        dist = euclidean_distances(
            df.loc[df['time']==1][self.features],
            df.loc[df['time']==2][self.features]
        )
        return dist, ix_to_cell1, ix_to_cell2
        
        
    def zero_pad(self, m, shape):
        """Pads matrix with zeros to be of desired shape"""
        out = np.zeros(shape)
        nrow, ncol = m.shape
        out[0:nrow, 0:ncol] = m
        return out


    def make_square(self, m):
        """Turns matrix into square matrix, as required by Munkres algorithm"""
        r,c = m.shape
        if r==c:
            return m
        elif r>c:
            return self.zero_pad(m, (r,r))
        else:
            return self.zero_pad(m, (c,c))
    
    @classmethod
    def track(cls, ims, do_area = False):
        tracker =  cls(do_area = do_area)

        tracked_masks = []
        tracked_masks.append(ims[0,:,:])
        for i in range(1,ims.shape[0]):
            prev, curr = ims[i-1], ims[i]
            curr_tracked  = tracker.correspondence(prev, curr)
            tracked_masks.append(curr_tracked)
        tracked_masks = np.array(tracked_masks, dtype = np.uint16)
        return tracked_masks 

class CentroidTracker():
    def __init__(self, blobs):
        self.blobs = blobs

    def track(self):
        self.extractFeatures()
        self.trackpy()
        self.assignPix()
        return self.blobs

    def extractFeatures(self):
        print("--Extracting feaures")
        props_df = pd.DataFrame(regionprops_table(self.blobs[0,:,:], properties=["label", "centroid"]))
        props_df["frame"] = 0
        for i,blob_mask in enumerate(self.blobs[1:,:,:]):
            new_props_df = pd.DataFrame(regionprops_table(blob_mask, properties=["label", "centroid"]))
            new_props_df["frame"] = i+1
            props_df = pd.concat([props_df, new_props_df], ignore_index  =True)

        self.features = props_df

    def trackpy(self):
        print("tracking with trackpy")
        self.assignments = link(self.features, search_range=10,pos_columns=["centroid-0", "centroid-1"])
        print(self.assignments)

    def assignPix(self):
        print("making assignments")
        for index, row in self.assignments.iterrows():
            i = row["frame"]
            label = row["label"]
            if label>0:
                self.blobs[int(i),:,:][self.blobs[int(i),:,:]==label] = int(row["particle"])+1