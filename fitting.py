import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from config import *


def downsample_log_space(xdata, ydata, num_bins):
    # Logarithmically space the bins
    log_bins = np.logspace(np.log10(xdata[0]), np.log10(xdata[-1]), num_bins)

    # Digitize the data (assign each data point to a bin)
    bin_indices = np.digitize(xdata, log_bins)

    # Calculate the average x and y for each bin
    xdata_binned = [xdata[bin_indices == i].mean() for i in range(1, len(log_bins))]
    ydata_binned = [ydata[bin_indices == i].mean() for i in range(1, len(log_bins))]

    return np.array(xdata_binned), np.array(ydata_binned)


def log_fit_results(results, fit_function):
    xdata = results[0]['frequency'][1:]
    # Average the results from all the files
    responses = [result['responses'][1:] for result in results]
    stacked_responses = np.stack(responses, axis=0)
    ydata = np.mean(stacked_responses, axis=0)

    plt.plot(xdata, ydata, label='pre_log', color='b', linewidth=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Response')
    plt.show()

    # bright_noise_responses = [bright_noise_result['responses'][1:] for bright_noise_result in bright_noise_results]
    # stacked_bn_responses = np.stack(bright_noise_responses, axis=0)
    # Might want to use this for subtraction
    # bnydata = np.mean(stacked_bn_responses, axis=0)
    # subtracted_y_data = ydata - bnydata
    # Downsample data in log space
    xdata, ydata = downsample_log_space(xdata, ydata, NUM_LOG_BINS)
    xdata_log = np.log10(xdata)
    ydata_log = np.log10(ydata)

    # Create a mask for values in xdata less than or equal to the cutoff frequency
    mask = (xdata_log <= CUTOFF_FREQUENCY) & (xdata_log >= STARTING_FREQUENCY)

    # Apply the mask to both xdata and ydata
    xdata_log = xdata_log[mask]
    ydata_log = ydata_log[mask]

    # Identify and print NaNs or infs in the log-transformed data
    problematic_indices = np.unique(np.concatenate([
        np.where(np.isnan(xdata_log))[0],
        np.where(np.isnan(ydata_log))[0],
        np.where(np.isinf(xdata_log))[0],
        np.where(np.isinf(ydata_log))[0]
    ]))

    # Print out the problematic indices
    if len(problematic_indices) > 0:
        print("Problematic indices:", problematic_indices)

        # Remove problematic indices from xdata and ydata
        xdata_log = np.delete(xdata_log, problematic_indices)
        ydata_log = np.delete(ydata_log, problematic_indices)

    plt.plot(xdata_log, ydata_log, label='post_log', color='b', linewidth=1)
    plt.xlabel('log_Frequency [Hz]')
    plt.ylabel('log_Response')
    plt.show()

    if fit_function == log_position_psd_clercx:
        guess_bounds = ([10e20, 150e-7, 3e-16], [10e30, 150e-2, 3e-13])
    else:
        guess_bounds = ([10e18, .0001, 2e-15], [10e20, 15, 3e-14])

    popt, pcov = curve_fit(fit_function, xdata_log, ydata_log, bounds=guess_bounds, maxfev=1000000)

    return popt, pcov  # Convert fitted parameters back to linear scale



