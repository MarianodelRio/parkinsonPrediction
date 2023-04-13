'''This file contains the dataset class.
This class is used to load the data and transform it into tensors.
'''

import numpy as np
import torch 
from torch.utils.data import Dataset
from src.utils import create_windows

class Dataset(Dataset):

    def __init__(self, data, labels=None, window_size=10):
        super().__init__()
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.windows = create_windows(self.data, self.labels, self.window_size)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, ix):
        if self.labels is not None:
            data, label = self.windows[ix]
            data = torch.from_numpy(data).float()
            label = torch.from_numpy(label).float()
            return data, label
        else:
            data = self.windows[ix]
            data = torch.from_numpy(data).float()
            return data


