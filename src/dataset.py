'''This file contains the dataset class.
This class is used to load the data and transform it into tensors.
'''

import numpy as np
import torch 
from torch.utils.data import Dataset
from src.utils import create_windows

class Dataset(Dataset):

    def __init__(self, data, window_size=10, prediction_time=4):
        super().__init__()
        self.data = data
        self.window_size = window_size
        self.prediction_time = prediction_time
        self.data_windows, self.label_windows = create_windows(self.data, self.window_size, prediction_time)
    
    def __len__(self):
        return self.data_windows.shape[0]
    
    def __getitem__(self, ix):
        '''
        Return a tuple with the data and the label: 
        - data: tensor with shape (window_size, 7)
        - label: tensor with shape (prediction_time, 4)
        '''
        data, label = self.data_windows[ix], self.label_windows[ix]
        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).float()
        return data, label
        


