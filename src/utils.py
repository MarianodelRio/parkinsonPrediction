'''Utils functions for the project'''

import pandas as pd
import numpy as np

def create_windows(data, labels, window_size):
    '''Create windows of the data and labels'''
    if labels is not None:
        windows = []
        for i in range(0, len(data)):
            if i + window_size < len(data):
                windows.append((data[i:i+window_size], labels[i:i+window_size]))
        return windows
    else:
        windows = []
        for i in range(0, len(data)):
            if i + window_size < len(data):
                windows.append(data[i:i+window_size])
    
        return np.array(windows)
    

def fill_missing_values(data):
    '''Fill missing values in the data'''
    data.fillna(data.mean(numeric_only=True).round(1), inplace=True)
    for i in data.columns:
        data[i].fillna(data[i].mode()[0], inplace=True)

    missing_values = data.isnull().sum() 
    assert missing_values.sum() == 0, "There are still missing values in the data"

    return data

def transform_data(url):
    '''Read the data from the url and transform it into numpy arrays'''
    data = pd.read_csv(url)
    labels = np.array([data['updrs_1'], data['updrs_2'], data['updrs_3'], data['updrs_4']])
    data = data.drop(['updrs_1'], axis=1)
    data = fill_missing_values(data)
    data = data.to_numpy()
    labels = labels.reshape(-1, 4)
    data[:, -1] = data[:, -1] == 'OFF' 
    data = data.astype(np.float32)
    return data, labels