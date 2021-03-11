import math
import argparse
import configparser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from scipy.io import loadmat
from sklearn import preprocessing
from pathlib import Path, PurePath
from collections import defaultdict
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split

seed_root_path = None
preprocessed_root_path = None
train_specInput_root_path = None
train_tempInput_root_path = None
train_label_root_path = None

test_specInput_root_path = None
test_tempInput_root_path = None
test_label_root_path = None

input_width = None
no_of_trials = None
temInput_length = None
specInput_length = None
test_split_ratio = None

def read_config(config_path):
    conf = configparser.ConfigParser()
    conf.read(config_path)

    global seed_root_path, preprocessed_path, no_of_trials, train_specInput_root_path, \
    train_tempInput_root_path, train_label_root_path, test_specInput_root_path, \
    test_tempInput_root_path, test_label_root_path
    seed_root_path = Path(conf['path']['seed_root_path'])
    preprocessed_root_path = Path(conf['path']['preprocessed_root_path'])
    train_specInput_root_path = Path(conf['path']['train_specInput_root_path'])
    train_tempInput_root_path = Path(conf['path']['train_tempInput_root_path'])
    train_label_root_path = Path(conf['path']['train_label_root_path'])
    test_specInput_root_path = Path(conf['path']['test_specInput_root_path'])
    test_tempInput_root_path = Path(conf['path']['test_tempInput_root_path'])
    test_label_root_path = Path(conf['path']['test_label_root_path'])
    
    # Create directories if don't exist
    preprocessed_root_path.mkdir(parents=True, exist_ok=True)
    train_specInput_root_path.mkdir(parents=True, exist_ok=True)
    train_tempInput_root_path.mkdir(parents=True, exist_ok=True)
    train_label_root_path.mkdir(parents=True, exist_ok=True)
    test_specInput_root_path.mkdir(parents=True, exist_ok=True)
    test_tempInput_root_path.mkdir(parents=True, exist_ok=True)
    test_label_root_path.mkdir(parents=True, exist_ok=True)

    global input_width, specInput_length, temInput_length, test_split_ratio
    input_width = int(conf['data']['input_width'])
    no_of_trials = int(conf['data']['no_of_trials'])
    specInput_length = int(conf['data']['specInput_length'])
    temInput_length = int(conf['data']['temInput_length'])
    test_split_ratio = float(conf['data']['test_split_ratio'])

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2

# Transforms 1D channel data to 2D spatial mapping
def data_1Dto2D(data_1D, X=9, Y=9):
    data_2D = np.zeros([X, Y])
    data_2D[0, 3:6] = data_1D[0:3]
    data_2D[1, 3], data_2D[1, 5] = data_1D[3], data_1D[4]
    for i in range(5):
        data_2D[i+2, :] = data_1D[5 + i * 9:5 + (i + 1) * 9]
    data_2D[7, 1:8] = data_1D[50:57]
    data_2D[8, 2:7] = data_1D[57:62]
    return data_2D

def interpolate(data, size=None):
    if not size:
        size = (input_width, input_width)
    return np.array(Image.fromarray(data).resize(size, resample=Image.BICUBIC))

# Returns a 3D array (5 x input_width x input_width) containing
# 5 spectral features arranged according to 
# their spatial configurations.
def extract_spectral(signal, frequency=200):
    DE_delta = np.zeros(shape=[0], dtype=float)
    DE_theta = np.zeros(shape=[0], dtype=float)
    DE_alpha = np.zeros(shape=[0], dtype=float)
    DE_beta = np.zeros(shape=[0], dtype=float)
    DE_gamma = np.zeros(shape=[0], dtype=float)

    for channel in range(62):
        trial_signal = signal[channel]

        delta = butter_bandpass_filter(trial_signal, 1, 4, frequency, order=3)
        theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
        alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
        beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
        gamma = butter_bandpass_filter(trial_signal, 31, 51, frequency, order=3)

        DE_delta = np.append(DE_delta, compute_DE(delta))
        DE_theta = np.append(DE_theta, compute_DE(theta))
        DE_alpha = np.append(DE_alpha, compute_DE(alpha))
        DE_beta = np.append(DE_beta, compute_DE(beta))
        DE_gamma = np.append(DE_gamma, compute_DE(gamma))

    de_2D = np.stack(
        [
            data_1Dto2D(delta), 
            data_1Dto2D(DE_theta), 
            data_1Dto2D(DE_alpha),
            data_1Dto2D(DE_beta),
            data_1Dto2D(DE_gamma)
        ])
    
    de_2D_interp = np.array([interpolate(signal) for signal in de_2D])
    return de_2D_interp

# Returns a 3D array (samples x input_width x input_width) containing
# 25 temporal features (timestamps) arranged 
# according to their spatial configurations.
def extract_temporal(signal, samples=25):
    
    timestamps = np.linspace(0, len(signal[0]) - 1, samples, dtype=int)
    temp_signals = list()
    # print(len(timestamps))

    for channel in range(62):
        trial_signal = signal[channel]
        temp_signals.append(trial_signal[timestamps])
        
    temp_signals = np.array(temp_signals).T
    temporal_2D = np.stack([data_1Dto2D(signal) for signal in temp_signals])
    temporal_2D_interp = np.array([interpolate(signal) for signal in temporal_2D])
    return temporal_2D_interp

def process(filename, subject_name):
    data = loadmat(filename)
    frequency = 200

    print("File: ", filename)

    spectoral_feats = np.empty([0, specInput_length, input_width, input_width])
    temporal_feats = np.empty([0, temInput_length, input_width, input_width])
    labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

    for trial in range(no_of_trials):
        trial_signal = data[subject_name + '_eeg' + str(trial + 1)]

        de_3D = extract_spectral(trial_signal)
        temp_3D =extract_temporal(trial_signal, samples=temInput_length)

        spectoral_feats = np.vstack([spectoral_feats, de_3D[np.newaxis, :, :, :]])
        temporal_feats = np.vstack([temporal_feats, temp_3D[np.newaxis, :, :, :]])

    spectoral_feats = np.moveaxis(spectoral_feats, 1, -1) # channel first to channel last
    temporal_feats = np.moveaxis(temporal_feats, 1, -1) # channel first to channel last

    print("Spectral Features shape:", spectoral_feats.shape)
    print("Temporal Features shape:", temporal_feats.shape)
    return spectoral_feats, temporal_feats, labels

def run():

    subject_short_names = {
        '1': 'djc',
        '2': 'jl',
        '3': 'jj',
        '4': 'lqj',
        '5': 'ly',
        '6': 'mhw',
        '7': 'phl',
        '8': 'sxy',
        '9': 'wk',
        '10': 'ww',
        '11': 'wsf',
        '12': 'wyw',
        '13': 'xyl',
        '14': 'ys',
        '15': 'zjy'
    }

    session_count = defaultdict(int)

    for filename in seed_root_path.glob('*_*.mat*'):
        subject_id = filename.name.split('_')[0] # extracts the subject id from filename
        session_count[subject_id] += 1
        spec_feats, temp_feats, labels = process(filename, subject_short_names[subject_id])

        # Split into train/test (6:4)
        spec_train, spec_test, temp_train, temp_test, y_train, y_test = train_test_split(spec_feats, temp_feats, labels, test_size=test_split_ratio)

        # Save to files
        datasets = [spec_train, spec_test, temp_train, temp_test, y_train, y_test]
        paths = [
            train_specInput_root_path, test_specInput_root_path, 
            train_tempInput_root_path, test_tempInput_root_path,
            train_label_root_path, test_label_root_path]
        
        for d, p in zip(datasets, paths):
            subject_path = p / f'subject_{subject_id}'
            subject_path.mkdir(parents=True, exist_ok=True)
            data_path = subject_path / f'section_{session_count[subject_id]}_data.npy'
            with open(data_path, 'wb') as f:
                np.save(f, d)
            print(f'Saved to : {data_path}')
        
        print(f"Saved preprocess data for subject {subject_id}.")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument of running SST-EmotionNet preprocessing.')
    parser.add_argument(
        '-c', type=str, help='Config file path.', required=True)
    args = parser.parse_args()
    read_config(args.c)
    run()