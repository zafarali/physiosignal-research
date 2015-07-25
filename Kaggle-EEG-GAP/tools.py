import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from glob import glob
import os

def prepare_data_train(subject_id, series_id):
    """
        read and prepare training data frame
        @params: 
            subject_id: subject number
            series_id: the series number
        @return:
            data, labels
    """

    data = pd.read_csv('./data/train/subj' + str(subject_id) + '_series' + str(series_id) + '_data.csv')
    labels = pd.read_csv('./data/train/subj' + str(subject_id) + '_series' + str(series_id) + '_events.csv')

    # drop the id column since the rows are already algined.
    data = data.drop(['id'], axis=1)
    labels = labels.drop(['id'], axis=1)

    return data, labels

def prepare_data_test(subject_id, series_id):
    data = pd.read_csv('./data/test/subj' + str(subject_id) + '_series' + str(series_id) + '_data.csv')
    return data


scaler = StandardScaler() # removes the mean and scales the variance to 1 unit.

def data_preprocess_train(X):
    X_prep = scaler.fit_transform(X)

    # preprocessing here

    return X_prep

def data_preprocess_test(X):
    X_prep = scaler.transform(X)

    # preprocessing here

    return X_prep

