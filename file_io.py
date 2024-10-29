import scipy
from nptdms import TdmsFile
import numpy as np
from analysis import autocorrelation
from config import ACF
import os
import pickle

import matplotlib.pyplot as plt

def read_tdms_file(file_path, data_col, track_length):
    tdms_file = TdmsFile.read(file_path)

    channel = tdms_file["main"]["X_" + str(data_col-1)]
    return channel[:track_length]


def bin_data(series, bin_size):
    # Ensuring the length of series is divisible by bin_size
    print("bin")
    length = len(series) - len(series) % bin_size
    series = series[:length]
    return np.mean(series.reshape(-1, bin_size), axis=1)


def downsample_log_space(xdata, ydata, num_bins):
    # Logarithmically space the bins
    log_bins = np.logspace(np.log10(xdata[0]), np.log10(xdata[-1]), num_bins)
    # Digitize the data (assign each data point to a bin)
    bin_indices = np.digitize(xdata, log_bins)
    # Calculate the average x and y for each bin
    xdata_binned = [xdata[bin_indices == i].mean() for i in range(1, len(log_bins))]
    ydata_binned = [ydata[bin_indices == i].mean() for i in range(1, len(log_bins))]
    return np.array(xdata_binned), np.array(ydata_binned)


def process_folder(folder_name, tracks_per_file, num_files, track_length, time_between_samples, bin_num):
    results = []
    for i in range(num_files):
        print("Reading ", folder_name, str(i))
        file_path = os.path.join(folder_name, "iter_"+str(i)+".tdms")
        result = process_file(file_path, tracks_per_file, track_length, time_between_samples, bin_num)
        if result:
            results.append(result)
    return results


def process_file(file_path, data_col, track_length, time_between_samples, bin_num):
    series = read_tdms_file(file_path, data_col, track_length)

    if series is None or len(series) == 0:
        print(f"Data not found or empty in {file_path}")
        return None

    time = np.arange(0, len(series)) * time_between_samples
    bin_series = bin_data(series, bin_num)
    bin_time = bin_data(time, bin_num)

    v_series = np.diff(bin_series) / np.diff(bin_time)
    frequency, local_response = scipy.signal.periodogram(bin_series, 1 / (time_between_samples * bin_num), scaling="density")
    v_freq, v_psd_local = scipy.signal.periodogram(v_series, 1 / (time_between_samples * bin_num), scaling="density")
    responses = np.sqrt(local_response)
    v_psd = np.sqrt(v_psd_local)
    if ACF:
        acf = autocorrelation(bin_series)
        v_acf = autocorrelation(v_series)
    else:
        acf = 0
        v_acf = 0
    second_moment = np.average(bin_series ** 2)

    return {
        "series": series,
        "time": time,
        "frequency": frequency,
        "responses": responses,
        "acf": acf,
        "v_freq": v_freq,
        "v_psd": v_psd,
        "v_acf": v_acf,
        "second_moment": second_moment
    }

def save_results(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def check_and_load_or_process(filename, *args):
    if os.path.exists(filename):
        print(f"Loading results from {filename}")
        return load_results(filename)
    else:
        print(f"Processing data for {filename}")
        results = process_folder(*args)
        save_results(results, filename)
        return results