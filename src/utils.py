'''Utils functions for the project'''

import pandas as pd
import numpy as np

def create_windows_patient(data, window_size, prediction_time):
    '''Create windows of the data and labels for each patient. 
    Prediction time indicate number of predictions (6 months, 12 months, ...)'''
    windows = []
    for i in range(0, len(data)):
        if i + window_size < len(data):
            labels = data[i+window_size - prediction_time : i+window_size, 3:7]
            windows.append([data[i:i+window_size-prediction_time], labels])
    
    windows = np.array(windows)
    print(windows.shape)
    return windows[:, 0], windows[:,1]

def create_windows(data, window_size, prediction_time):
    '''Create windows of the data and labels'''
    patients = np.unique(data[:, 1])
    windows = []
    for patient in patients:
        patient_data = data[data[:, 1] == patient]
        patient_data, patient_labels = create_windows_patient(patient_data, window_size, prediction_time)
        print(patient_data, patient_labels)
        windows.append([patient_data, patient_labels])
    
    windows = np.array(windows)
    return windows[:,0], windows[:, 1]
    

def fill_missing_values(data):
    '''Fill missing values in the data'''
    data.fillna(data.mean(numeric_only=True).round(1), inplace=True)
    for i in data.columns:
        data[i].fillna(data[i].mode()[0], inplace=True)

    missing_values = data.isnull().sum() 
    assert missing_values.sum() == 0, "There are still missing values in the data"

    return data


def read_clean_data(train_clinical_url, suplemental_clinical_url=None, train_peptides_url=None, train_proteins_url=None):
    '''Read and clean the data. Return dataframe with all data and labels'''
    data = pd.read_csv(train_clinical_url)

    if suplemental_clinical_url is not None:
        suplemental_clinical_data = pd.read_csv(suplemental_clinical_url)
        data = pd.concat([data, suplemental_clinical_data])
    if train_peptides_url is not None:
        train_peptides_data = pd.read_csv(train_peptides_url)
        data = pd.concat([data, train_peptides_data])
    if train_proteins_url is not None:
        train_proteins_data = pd.read_csv(train_proteins_url)
        data = pd.concat([data, train_proteins_data])

    data = fill_missing_values(data)
    data = data.sort_values(by=['patient_id', 'visit_month'])
    return data

def transform_dataframe_to_numpy(data):
    data['upd23b_clinical_state_on_medication'] = data['upd23b_clinical_state_on_medication'] == 'OFF' 
    data = data.astype(np.float32).to_numpy()
    return data

def smape_metric(y_true, y_pred):
    '''Calculate the symmetric mean absolute percentage error'''
    return 100 * np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

def extract_test_data(dm):
    '''Extract the test data and labels from the datamodule'''
    test_data = np.array([ sample[0].numpy() for sample in dm.val_dataloader().dataset])
    test_label = np.array([ sample[1].numpy() for sample in dm.val_dataloader().dataset])
    return test_data, test_label

    

def get_patient_visits(patient_id, train_clinical_url, suplemental_clinical_url, peptides_url, proteins_url):
    '''Get the last 10 visits of a patient'''
    patient_visits = read_clean_data(train_clinical_url, suplemental_clinical_url, peptides_url, proteins_url)
    patient_visits = patient_visits[patient_visits['patient_id'] == patient_id] 
    patient_visits = patient_visits.tail(10)
    return patient_visits
