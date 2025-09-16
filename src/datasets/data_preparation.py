import os
import logging
import numpy as np
import pandas as pd

from natsort import natsorted
from sklearn.utils import resample
from torch.utils.data import DataLoader
from scipy.stats import skew, kurtosis

from src.utils.utils import PROJ_PATH, RANDOM_SEED
from src.datasets.time_series import TimeSeriesDataset

logger = logging.getLogger(__name__)

training_partitions = [1, 2, 3, 4]
testing_partitions = [5]
logger.info(f'Training partitions: {training_partitions}, Testing partitions: {testing_partitions}')

data_path = PROJ_PATH / 'data'
partition_dirs = ['partition1', 'partition2', 'partition3', 'partition4', 'partition5']

FL_train_paths = [data_path / partition_dirs[part - 1] / 'FL' for part in training_partitions]
NF_train_paths = [data_path / partition_dirs[part - 1] / 'NF' for part in training_partitions]
FL_test_paths = [data_path / partition_dirs[part - 1] / 'FL' for part in testing_partitions]
NF_test_paths = [data_path / partition_dirs[part - 1] / 'NF' for part in testing_partitions]


def extract_features(csv_files):
    data = []
    for file in csv_files:
        df = pd.read_csv(file, sep='\t')
        features = df.iloc[:, 1:25]
        data.append(features)
    
    return data


def prepare_data(paths):
    
    csv_files = []
    for path in paths:
        csv_files.extend([path / file for file in os.listdir(path) if file.endswith('.csv')])
    csv_files = natsorted(csv_files)

    data = extract_features(csv_files)
    for i, df in enumerate(data):
        data[i] = df.apply(lambda x: x.fillna(x.mean()), axis=0)
    
    return data


FL_train = prepare_data(FL_train_paths)
NF_train  = prepare_data(NF_train_paths)
FL_test = prepare_data(FL_test_paths)
NF_test = prepare_data(NF_test_paths)
FL_train = resample(FL_train, replace=True, n_samples=len(FL_train) * 4, random_state=RANDOM_SEED)
NF_train = resample(NF_train, replace=True, n_samples=len(FL_train), random_state=RANDOM_SEED)

logger.info(f'FL_train len: {len(FL_train)}, NF_train len: {len(NF_train)}, FL_test len: {len(FL_test)}, NF_test len: {len(NF_test)}')

def extractInstances(flare_data_list, non_flare_data_list):
    flare_labels = np.ones(len(flare_data_list))
    non_flare_labels = np.zeros(len(non_flare_data_list))

    ts_data_list = flare_data_list + non_flare_data_list
    ts_labels = np.concatenate([flare_labels, non_flare_labels])
    permutation = np.random.permutation(len(ts_data_list))
    ts_data_list = [ts_data_list[i] for i in permutation]
    ts_labels = ts_labels[permutation]

    ts_data_list = np.array([ts.to_numpy() for ts in ts_data_list])
    
    ts_data_list_LAST = ts_data_list[:, -1, :]   
    return ts_data_list, ts_data_list_LAST, ts_labels


def extract_features(ts_data_list):
    num_time_series = ts_data_list.shape[2]
    ts_fts = []

    for i in range(num_time_series):
        ts_column = ts_data_list[:, :, i]

        median_value = np.median(ts_column, axis=1, keepdims=True)
        mean_value = np.mean(ts_column, axis=1, keepdims=True)
        std_deviation = np.std(ts_column, axis=1, keepdims=True)
        skewness_value = skew(ts_column, axis=1, keepdims=True)
        kurtosis_value = kurtosis(ts_column, axis=1, keepdims=True)
        last_value = ts_column[:, -1:]
        median_value[np.isnan(median_value)] = 0
        mean_value[np.isnan(mean_value)] = 0
        std_deviation[np.isnan(std_deviation)] = 0
        skewness_value[np.isnan(skewness_value)] = 0
        kurtosis_value[np.isnan(kurtosis_value)] = 0
        last_value[np.isnan(last_value)] = 0
        
        features = np.concatenate([median_value, mean_value, std_deviation, skewness_value, kurtosis_value, last_value], axis=1)
        ts_fts.append(features)

    ts_fts = np.concatenate(ts_fts, axis=1)

    return ts_fts

ts_data_list_train, ts_data_list_LAST_train, ts_labels_train = extractInstances(FL_train, NF_train)
ts_data_list_test, ts_data_list_LAST_test, ts_labels_test = extractInstances(FL_test, NF_test)

train_dataset = TimeSeriesDataset(ts_data_list_train, ts_labels_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

test_dataset = TimeSeriesDataset(ts_data_list_test, ts_labels_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

ts_fts_train = extract_features(ts_data_list_train)
ts_fts_test = extract_features(ts_data_list_test)
