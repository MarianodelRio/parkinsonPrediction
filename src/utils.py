'''Utils functions for the project'''

import pandas as pd
import numpy as np

def create_windows_patient(data, window_size, prediction_time):
    '''Create windows of the data and labels for each patient. 
    Prediction time indicate number of predictions (6 months, 12 months, ...)'''
    data_windows = []
    label_windows = []
    for i in range(0, len(data)):
        if i + window_size < len(data):
            data_windows.append(data[i:i+window_size-prediction_time])
            labels = data[i+window_size - prediction_time : i+window_size, 3:7]
            label_windows.append(labels)
    
    return np.array(data_windows), np.array(label_windows)

def create_windows(data, window_size, prediction_time):
    '''Create windows of the data and labels'''
    patients = np.unique(data[:, 1])
    # Initialize a numpy array without shape
    data_windows =  []
    label_windows = []

    for patient in patients:
        patient_data = data[data[:, 1] == patient]
        patient_data, patient_labels = create_windows_patient(patient_data, window_size, prediction_time)
        data_windows.extend(patient_data)
        label_windows.extend(patient_labels)
    
    return np.array(data_windows), np.array(label_windows)
    

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

    
def get_patient_visits(patient_id, train_clinical_url, suplemental_clinical_url):
    '''Get the last 10 visits of a patient'''
    patient_visits = read_clean_data(train_clinical_url, suplemental_clinical_url=suplemental_clinical_url)
    patient_visits = patient_visits[patient_visits['patient_id'] == patient_id] 
    patient_visits = patient_visits.tail(10)
    patient_visits = transform_dataframe_to_numpy(patient_visits)
    return patient_visits

def make_predictions(test, model, train_url, sup_url, sample_prediction_df):
    for index, row in test.iterrows():
        # Get patient id
        patient_id = row['patient_id']
        # Get group key
        group_key = int(row['visit_id'][-1])
        # Get updrs test 
        updrs_test = row['updrs_test']
        # Get last visits
        last_visits = get_patient_visits(patient_id, train_url, sup_url)
        last_visits = last_visits.unsqueeze(0)
        # Get prediction
        pred = model.predict(last_visits)
        # Build prediction id (patient_id + group_key + updrs_test + months) 3342_0_updrs_1_plus_0_months
        pred_id0 = str(patient_id) + '_' + str(group_key) + '_' + str(updrs_test) + '_plus_0_months'
        pred_id1 = str(patient_id) + '_' + str(group_key) + '_' + str(updrs_test) + '_plus_6_months'
        pred_id2 = str(patient_id) + '_' + str(group_key) + '_' + str(updrs_test) + '_plus_12_months' 
        pred_id3 = str(patient_id) + '_' + str(group_key) + '_' + str(updrs_test) + '_plus_24_months' 

        # Append prediction to dataframe (prediction_id,rating,group_key)
        sample_prediction_df = sample_prediction_df.append({'prediction_id': pred_id0, 'rating': pred[0], 'group_key': group_key}, ignore_index=True)
        sample_prediction_df = sample_prediction_df.append({'prediction_id': pred_id1, 'rating': pred[1], 'group_key': group_key}, ignore_index=True)
        sample_prediction_df = sample_prediction_df.append({'prediction_id': pred_id2, 'rating': pred[2], 'group_key': group_key}, ignore_index=True)
        sample_prediction_df = sample_prediction_df.append({'prediction_id': pred_id3, 'rating': pred[3], 'group_key': group_key}, ignore_index=True)
    
    return sample_prediction_df