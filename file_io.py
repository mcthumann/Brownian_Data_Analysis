import scipy
from scipy.fft import fft, ifft, fftfreq
from nptdms import TdmsFile
import numpy as np
from analysis import autocorrelation, compute_VACF
from config import ACF, BIN, SINC, SAMPLING_RATE, HAMMING, BIN_NUM, RECT_WINDOW, SAVE, SIM
import os
import pandas as pd
import pickle

import matplotlib.pyplot as plt

def read_tdms_file(file_path, data_col, track_length):
    tdms_file = TdmsFile.read(file_path)

    channel = tdms_file["main"]["X_" + str(data_col-1)]
    return channel[:track_length]


def read_csv_file(file_path, data_col, track_length):
    # Read CSV with low_memory=False to avoid DtypeWarning
    df = pd.read_csv(file_path, low_memory=False)

    # Filter columns that start with "Position"
    position_columns = [col for col in df.columns if col.startswith("Position")]

    if data_col - 1 < len(position_columns):
        position_col = position_columns[data_col - 1]
        return df[position_col].iloc[:track_length]
    else:
        raise ValueError(f"Data column index {data_col} is out of range for available 'Position' columns.")

def bin_data(series, bin_size):
    # Ensuring the length of series is divisible by bin_size
    print("Bin")
    length = len(series) - len(series) % bin_size
    series = np.array(series[:length])
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

def low_pass_sinc_filter(time_trace, cutoff_frequency, sampling_rate):
    """
    Applies a low-pass filter to the input time trace using a sinc function.
    """
    # Perform FFT on the time trace
    freq_domain_trace = fft(time_trace)

    # Get frequencies corresponding to FFT components
    frequencies = fftfreq(len(time_trace), d=1 / sampling_rate)

    # Create a low-pass filter (sinc function in frequency domain)
    low_pass_filter = np.abs(frequencies) <= cutoff_frequency

    # Apply the filter
    filtered_trace_freq_domain = freq_domain_trace * low_pass_filter

    # Inverse FFT to return to time domain
    filtered_trace = ifft(filtered_trace_freq_domain).real
    return filtered_trace


def low_pass_velocity_trace(time_trace, cutoff_frequency, sampling_rate):
    """
    Creates a low-pass velocity trace from the input time trace.
    """
    # Perform FFT on the time trace
    freq_domain_trace = fft(time_trace)

    # Get frequencies corresponding to FFT components
    frequencies = fftfreq(len(time_trace), d=1 / sampling_rate)

    # Create a low-pass filter
    low_pass_filter = np.abs(frequencies) <= cutoff_frequency

    # Apply low-pass filter and differentiate by multiplying by -i * omega
    velocity_trace_freq_domain = freq_domain_trace * low_pass_filter * (-1j * 2 * np.pi * frequencies)

    # Inverse FFT to return to time domain
    velocity_trace = ifft(velocity_trace_freq_domain).real
    return velocity_trace

def apply_hamming_window(signal):
    """
    Applies a Hamming window to the signal.
    """
    print("HAMMING")
    hamming_window = np.hamming(len(signal))
    return signal * hamming_window

def rect_window_filter(signal, bin_size):
    rect_window = np.ones(bin_size)
    binned_signal = np.convolve(signal, rect_window, mode='same')/bin_size
    edge_correction = bin_size // 2
    binned_signal[:edge_correction] = signal[:edge_correction]
    binned_signal[-edge_correction:] = signal[-edge_correction:]
    return binned_signal

def process_folder(folder_name, tracks_per_file, num_traces, track_length, time_between_samples, bin_num):
    results = []
    for i in range(num_traces):
        print("Reading ", folder_name, str(i))
        result = process_file(folder_name, i, tracks_per_file, track_length, time_between_samples, bin_num)
        if result:
            results.append(result)
    return results


def process_file(folder_name, trace_num, data_col, track_length, time_between_samples, bin_num):
    if SIM:
        print(trace_num)
        series = read_csv_file(folder_name, trace_num, track_length)
    else:
        file_path = os.path.join(folder_name, "iter_" + str(trace_num) + ".tdms")
        series = read_tdms_file(file_path, data_col, track_length)

    if series is None or len(series) == 0:
        print(f"Data not found or empty in {file_path}")
        return None

    time = np.arange(0, len(series)) * time_between_samples

    # Low pass before finding the time trace for position and velocity
    if BIN:
        series = bin_data(series, bin_num)
        time = bin_data(time, bin_num)
        v_series = np.diff(series) / np.diff(time)
    elif RECT_WINDOW:
        series = rect_window_filter(series, BIN_NUM)
        v_series = np.diff(series) / np.diff(time)
    elif SINC:
        series = low_pass_sinc_filter(series, 1e7, SAMPLING_RATE)
        v_series = low_pass_velocity_trace(series, 1e7, SAMPLING_RATE)
    else:
        v_series = np.diff(series) / np.diff(time)

    if HAMMING:
        series = apply_hamming_window(series)

    frequency, local_response = scipy.signal.periodogram(series, 1 / (time_between_samples * bin_num), scaling="density")
    v_freq, v_psd_local = scipy.signal.periodogram(v_series, 1 / (time_between_samples * bin_num), scaling="density")
    responses = np.sqrt(local_response)
    v_psd = np.sqrt(v_psd_local)

    if ACF:
        acf = autocorrelation(series)
        v_acf = autocorrelation(v_series)
    else:
        acf = 0
        v_acf = 0

    second_moment = np.average(series ** 2)

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
        if SAVE:
            save_results(results, filename)
        return results