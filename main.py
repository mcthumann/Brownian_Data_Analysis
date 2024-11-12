import numpy as np

from file_io import *
from fitting import *
from config import *
from plotting import *

def run():
    folders = [""]
    for folder in folders:
        filename = folder + "_data.pkl"

        results = check_and_load_or_process(
            filename,
            BASE_PATH + folder, TRACKS_PER_FILE, NUM_FILES, TRACK_LENGTH, TIME_BETWEEN_SAMPLES, BIN_NUM
        )

        # Plot the data
        plot_results(results, folder)

        K = 1e2  # Example trap strength
        a = 3e-6  # Example particle radius in meters
        V = 1e30  # Example voltage-to-position conversion factor

        # Frequency range
        frequencies = np.logspace(5, 10, 500)  # Logarithmic frequency range from 10^1 to 10^7
        omega = 2 * np.pi * frequencies  # Convert to angular frequency

        # Calculate PSD
        psd_values = PSD_fitting_func(omega, K, a, V)

        # Plot
        plt.plot(frequencies, psd_values, label='PSD')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')
        plt.title('Power Spectral Density (PSD)')
        plt.legend()
        plt.show()

        # Fit the results
        fit_data(results)
        plt.show()

        # Run the Analysis


    plt.show()


if __name__ == '__main__':
    run()
