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
from src.utils import read_clean_data, transform_dataframe_to_numpy

class DataModule(pl.LightningDataModule):
    
    def __init__(self, batch_size, path_clinical_data=None, 
                 path_supplemental_data=None, path_peptides=None, path_proteins=None, 
                 window_size=4, prediction_time=4):
        super().__init__()

        self.batch_size = batch_size
        self.path_clinical_data = path_clinical_data
        self.path_supplemental_data = path_supplemental_data
        self.path_peptides = path_peptides
        self.path_proteins = path_proteins

        self.window_size = window_size
        self.prediction_time = prediction_time
    
    def setup(self, stage=None):
        data = read_clean_data(self.path_clinical_data, self.path_supplemental_data, 
                               self.path_peptides, self.path_proteins)
        data = transform_dataframe_to_numpy(data)
        dataset = Dataset(data, window_size=self.window_size, prediction_time=self.prediction_time)

        self.train, self.test = train_test_split(dataset, test_size=0.2, random_state=42)

        self.dataloader = {
            'train': DataLoader(self.train, batch_size=self.batch_size, shuffle=True),
            'test': DataLoader(self.test, batch_size=self.batch_size, shuffle=False)
        }
    
    def train_dataloader(self):
        return self.dataloader['train']
    
    def val_dataloader(self):
        return self.dataloader['test']


    
