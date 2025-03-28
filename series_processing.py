import numpy as np
import scipy
from tqdm import tqdm
from scipy.fft import fft, ifft, fftfreq
from config import ACF, BIN, SINC, HAMMING, BIN_NUM, RECT_WINDOW, SAMPLE, SIM


def compute_VACF_time_domain(v_series):
    n = len(v_series)
    vacf = np.zeros(n)
    for lag in tqdm(range(n), desc="VACF Compute"):
        vacf[lag] = np.dot(v_series[:n - lag], v_series[lag:]) / (
                n - lag)  # Normalize by number of overlapping terms
    return vacf

def autocorrelation(signal):
    # Using FFT for efficient computation of autocorrelation
    f_signal = np.fft.fft(signal, n=2 * len(signal))
    acf = np.fft.ifft(f_signal * np.conjugate(f_signal)).real[:len(signal)]
    acf /= acf[1]
    return acf

def bin_data(series, bin_size):
    # Ensuring the length of series is divisible by bin_size
    length = len(series) - len(series) % bin_size
    series = np.array(series[:length])
    return np.mean(series.reshape(-1, bin_size), axis=1)

def downsample_log_space(xdata, ydata, num_bins):
    # Logarithmically space the bins
    log_bins = np.logspace(np.log10(xdata[0]), np.log10(xdata[-1]), num_bins)
    # Digitize the data
    # (assign each data point to a bin)
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
    hamming_window = np.hamming(len(signal))
    return signal * hamming_window

def rect_window_filter(signal, bin_size):
    rect_window = np.ones(bin_size)
    binned_signal = np.convolve(signal, rect_window, mode='same')/bin_size
    edge_correction = bin_size // 2
    binned_signal[:edge_correction] = signal[:edge_correction]
    binned_signal[-edge_correction:] = signal[-edge_correction:]
    return binned_signal

def process_series(series, conf, bin=1):
    ## new function ...
    time = np.arange(0, len(series))
    time = time * (1/conf.sampling_rate)

    if SIM:
        time = time * conf.t_c

    series = bin_data(series, bin)
    time = bin_data(time, bin)
    v_series = np.diff(series) / np.diff(time)

    #
    # elif RECT_WINDOW:
    #     series = rect_window_filter(series, BIN_NUM)
    #     v_series = np.diff(series) / np.diff(time)
    # elif SINC:
    #     series = low_pass_sinc_filter(series, 1e7, conf.sampling_rate)
    #     v_series = low_pass_velocity_trace(series, 1e7, conf.sampling_rate)
    # else:
    #     v_series = np.diff(series) / np.diff(time)

    if HAMMING:
        series = apply_hamming_window(series)

    if SIM:
        frequency, psd = scipy.signal.periodogram(series, 1/conf.timestep, scaling="density")
        frequency /= conf.t_c
        psd *= conf.t_c
    else:
        frequency, psd = scipy.signal.periodogram(series, fs=conf.sampling_rate/bin, scaling="density")
    v_freq, v_psd = scipy.signal.periodogram(v_series, fs=conf.sampling_rate/bin, scaling="density")

    if ACF:
        acf = autocorrelation(series)
        v_acf = compute_VACF_time_domain(v_series)

    else:
        acf = 0
        v_acf = 0

    second_moment = np.average(series ** 2)

    return {
        "sampling_rate": conf.sampling_rate,
        "track_len": conf.track_len,
        "a": conf.a,
        "eta": conf.eta,
        "rho_silica": conf.rho_silica,
        "rho_f": conf.rho_f,
        "stop": conf.stop,
        "start": conf.start,
        "series": series,
        "time": time,
        "frequency": frequency,
        "psd": psd,
        "acf": acf,
        "v_freq": v_freq,
        "v_psd": v_psd,
        "v_acf": v_acf,
        "second_moment": second_moment
    }