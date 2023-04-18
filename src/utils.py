'''Utils functions for the project'''

import pandas as pd
import numpy as np

def create_windows(data, labels, window_size):
    '''Create windows of the data and labels'''
    if labels is not None:
        windows = []
        for i in range(0, len(data)):
            if i + window_size < len(data):
                windows.append((data[i:i+window_size], labels[i+window_size]))
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
    labels = fill_missing_values(pd.DataFrame(labels.T)).to_numpy()
    data = data.to_numpy()
    labels = labels.reshape(-1, 4)
    data[:, -1] = data[:, -1] == 'OFF' 
    data = data.astype(np.float32)
    return data, labels


def smape_metric(y_true, y_pred):
    '''Calculate the symmetric mean absolute percentage error'''
    return 100 * np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

def extract_test_data(dm):
    '''Extract the test data and labels from the datamodule'''
    test_data = np.array([ sample[0].numpy() for sample in dm.val_dataloader().dataset])
    test_label = np.array([ sample[1].numpy() for sample in dm.val_dataloader().dataset])
    return test_data, test_label

    

def get_patient_visits(patient_id, train_clinical_data, suplemental_clinical_data):
    '''Get the last 10 visits of a patient'''
    patient_visits = train_clinical_data[train_clinical_data['patient_id'] == patient_id]
    
    patient_visits = pd.concat([patient_visits,(suplemental_clinical_data[suplemental_clinical_data['patient_id'] == patient_id])])
    patient_visits = fill_missing_values(patient_visits)
    patient_visits = patient_visits.sort_values(by=['visit_month'])
    patient_visits = patient_visits.tail(10)
    return patient_visits
