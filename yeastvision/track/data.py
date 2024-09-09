import pandas as pd
import numpy as np
from yeastvision.track.cell import getLifeData, getCellData
from yeastvision.track.lineage import LineageConstruction, has_daughters, get_daughters, get_generations, get_gametes, compute_mating_lineage
from yeastvision.utils import normalize_im

class Population():
    def __init__(self, name, replicates):
        self.name = name
        self.replicates = []
    
    def getData(self, property, replicates = None):
        if replicates:
            toGet = [replicate for replicate in replicates if replicate in self.replicates]
        else:
            toGet = self.replicates

        data = None
        for replicate in self.replicates:
            newdata = replicate.get_heatmap_data(property, self.name)
            if data is not None:
                data = TimeSeriesData.combine_population_data(data, newdata)
            else:
                data = newdata
        return data
    

class PopulationReplicate():
    populations = {}
    def __init__(self, experiment_idx, population_name, ts):
        self.population = population_name
        self.id = experiment_idx
        self.name = self.population+"-"+str(id)
        self.ts = ts

        if self.population not in PopulationReplicate.populations:
            PopulationReplicate.populations[self.population] =  Population(self.population, [self])
        else:
            PopulationReplicate.populations[self.population].replicates.append(self.ts)


class TimeSeriesData():

    '''Class used to store and create time series cell properties. A TimeSeriesData object should correspond
    to a single set of tracked masks. This object has several important properties (above) and methods. '''

    def __init__(self, mask_id, labels, channels = None, channel_names = None, cell_data = None, life_data = None, 
                 no_cell_data=False): # user can specify that they do not want to compute cell data
                                     # life data is always required
        ''''
        Instantiates a new TimeSeriesData instance, given a set of tracked masks. It will compute cell data for the masks
        if not provided. 
        
        '''
        self.mask_id = mask_id
        
        self._set_label_props(labels)

        # populations
        self.populations = [self.label_vals-1]
        self.population_names = ["all"]
        self.channel_names = channel_names
        
        if cell_data is not None:
            self.cell_data = cell_data
        else:
            if not no_cell_data:
                self.cell_data = getCellData(labels, channels, channel_names)
            else:
                self.cell_data = None
        
        if life_data is not None:
            self.life_data = life_data
        else:
            self.life_data = getLifeData(labels)
        
        self.update_props()
    
    def has_cell_data(self):
        return self.cell_data is not None

    def update_cell_data(self, labels, channels = None, channel_names = None):
        self._set_label_props(labels)
        self.life_data = getLifeData(labels)

        self.cell_data = getCellData(labels, channels, channel_names)
        self.channel_names = channel_names
        self.update_props()

    def update_props(self):
        if self.has_cell_data():
            self.properties = self.cell_data.columns.tolist()
            self.properties.remove("labels")
        else:
            self.properties = None

    def _set_label_props(self, labels):
        '''
        Collects a the number and values of unique cell indices in the given labels. Stores
        them as attributes for later data processing
        '''
        label_vals = np.unique(labels)
        self.label_vals = label_vals[label_vals>0]
        self.num_objs = len(self.label_vals)
    
    def get_population_data(self, column_name, population_name = "all", func = np.mean):
        population = self.populations[self.population_names.index(population_name)]
        df = np.array(self.cell_data.iloc[population][column_name].to_list(), dtype = float)
        return func(df, axis = 0)
    
    def get_heatmap_data(self, column_name, population_name):
        heatmap = normalize_im(np.array(self.cell_data[column_name].to_list(), dtype =float), clip =False)
        population = self.populations[self.population_names.index(population_name)]
        im = np.zeros_like(heatmap, dtype = float)
        im[im==0] = np.nan
        im[population, :] = heatmap[population, :]
        return im
    
    def add_population(self, cells, name):
        self.populations.append(list(np.array(cells)-1))
        self.names.append(name)
    
    def get_population_sample_size(self, population_name):
        return len(self.populations[self.population_names.index(population_name)])
    
    def get_cell_info(self):
        return pd.DataFrame.from_dict({"cell":self.label_vals} | self.life_data )

    def get(self, property, cell):
        data = np.array((self.cell_data[property][cell-1].copy()), dtype = float)
        return data

    def hasLineage(self):
        return False
    
    def hasMating(self):
        return False
    
    @staticmethod
    def combine_population_data(arr1, arr2):
        # Find the widths of the arrays
        width1 = arr1.shape[1]
        width2 = arr2.shape[1]

        # Calculate the difference in widths
        width_diff = abs(width1 - width2)

        # Pad the array with the shorter width with None values
        if width1 < width2:
            padding = [(0, 0), (0, width_diff)]
            arr1 = np.pad(arr1, padding, mode='constant', constant_values=None)
        elif width2 < width1:
            padding = [(0, 0), (0, width_diff)]
            arr2 = np.pad(arr2, padding, mode='constant', constant_values=None)

        # Concatenate the arrays vertically
        result = np.concatenate((arr1, arr2), axis=0)

        return result


class LineageData(TimeSeriesData):
    mothers:pd.core.frame.DataFrame
    daughters:pd.core.frame.DataFrame
    mating: pd.core.frame.DataFrame
    generations: np.ndarray
    
    _hasLineage:bool
    _hasMating:bool

    def __init__(self, mask_id, cells, buds = None, mating = None, 
                 cell_data = None, channels = None, channel_names = None, 
                 life_data = None, daughters=None, mothers=None):
        
        super().__init__(mask_id, cells, channels = channels, channel_names=channel_names, 
                         cell_data=cell_data, life_data=life_data, 
                         no_cell_data=(cell_data is None)) # if no cell data is given, then by default none is computed

        self._hasLineage = False
        if daughters is not None and mothers is not None:
            self._hasLineage = True
            self.daughters, self.mothers = daughters, mothers
            self.set_generations()
        
        if buds is not None and not self._hasLineage:
            self.add_lineages(cells,buds)
        
        self._hasMating = False
        if mating is not None:
            self.add_mating(cells, mating)
    
    def add_lineages(self, cells, buds):
        self._hasLineage = True
        lineage = LineageConstruction(cells, buds, forwardskip=3)
        self.daughters, self.mothers = lineage.computeLineages()
        self.set_generations()
    
    def add_mating(self, cells, mating):
        self._hasMating = True
        self.mating = compute_mating_lineage(mating, cells)
        self.set_mating_data(mating)
        self.set_gamete_data()

    def get_initial_generation(self):
        life_df = pd.DataFrame.from_dict(self.life_data)
        potential_gen1 = life_df[life_df["birth"]==0].index.values
        return [val+1 for val in potential_gen1 if pd.isnull(self.mothers.iloc[val, self.mothers.columns.get_loc("mother")])]
    
    def set_generations(self):
        initial_gen = self.get_initial_generation()
        generations = get_generations(self.daughters, initial_gen)
        for i, generation in enumerate(generations):
            name = f"gen{i}"
            if name not in self.population_names:
                self.population_names.append(name)
                self.populations.append(np.array(generation, dtype=np.uint16)-1)
            else:
                idx = self.population_names.index(name)
                self.populations[idx] = (np.array(generation, dtype=np.uint16)-1)


    def set_mating_data(self, mating):
        mating_vals = np.unique(mating)
        mating_vals = mating_vals[mating_vals>0]
        self.add_population(mating_vals, "mating")
    
    def set_gamete_data(self):
        gametes = self.mating["gamete1"].values().to_list() + self.mating["gamete2"].values()
        gametes = [gamete for gamete in gametes if pd.notnull(gamete) and gamete not in gametes]
        self.add_population(gametes, "gametes")

    def get_cell_info(self):
        cell_info = {"cell":self.label_vals} | self.life_data 
        if self._hasLineage:
            cell_info = cell_info | self.mothers.to_dict(orient = "list")
        if self._hasMating:
            cell_info = cell_info | self.mating.to_dict(orient = "list")

        return pd.DataFrame.from_dict(cell_info)

    def hasLineages(self):
        return self._hasLineage
    
    def hasMating(self):
        return self._hasMating
    





