'''This file contains the datamodule class. 
This class is used to load the data, split it into train and test sets and build datasets and dataloaders.

@author: Mariano del Río
@date: 2022-01-20
'''

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

from src.dataset import Dataset
from src.utils import transform_data

class DataModule(pl.LightningDataModule):
    
    def __init__(self, batch_size, path_dataset):
        super().__init__()

        self.batch_size = batch_size
        self.path_dataset = path_dataset
    
    def setup(self, stage=None):
        data, labels = transform_data(self.path_dataset)
        dataset = Dataset(data, labels, window_size=10)

        self.train, self.test = train_test_split(dataset, test_size=0.2, random_state=42)

        self.dataloader = {
            'train': DataLoader(self.train, batch_size=self.batch_size, shuffle=True),
            'test': DataLoader(self.test, batch_size=self.batch_size, shuffle=False)
        }
    
    def train_dataloader(self):
        return self.dataloader['train']
    
    def val_dataloader(self):
        return self.dataloader['test']


    
