import pandas as pd
import numpy as np

class LineageData():
    def __init__(self):
        self.lineageData = []
        self.daughterArrays = []

    def append(self, object):
        if type(object) is pd.core.frame.DataFrame:
            self.lineageData = []
