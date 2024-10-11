from file_io import *
from fitting import *
from analysis_cole import *
from config import *
import numpy as np
from plotting import *

FUNC = admittance_model
LOG_FUNC = log_admittance_model

def run():
    # Assuming process_folder returns data compatible with JSON serialization
    filename = 'data.pkl'

    results = check_and_load_or_process(
        process_folder,
        filename,
        "/Users/colethumann/Desktop/20241008/with_preamp_and_higher_resistor2", TRACKS_PER_FILE, NUM_FILES, TRACK_LENGTH, TIME_BETWEEN_SAMPLES, BIN_NUM
    )

    # results2 = check_and_load_or_process(
    #     process_folder,
    #     filename,
    #     "/Volumes/Seagate Portable Drive/20241011/no_cage_dark_noise", TRACKS_PER_FILE, NUM_FILES,
    #     TRACK_LENGTH, TIME_BETWEEN_SAMPLES, BIN_NUM
    # )
    # Fit the results
    #popt, pocov = log_fit_results(results, LOG_FUNC)
    #fitted_params_str = '\n'.join([f'{param}: {value:.2e}' for param, value in zip(['m', 'K', 'a'], popt)])
    #xfit = np.linspace(0, 10e6, R)
    #yfit = FUNC(xfit, *popt)

    plot_results(results, "Cage Dark Noise")
    # plot_results(results2, "No Cage Dark Noise")
    plt.show()

    #plt.text(0.05, 0.95, fitted_params_str, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
    #         bbox=dict(boxstyle="round", alpha=0.5))
    #plt.xscale("log")
    #plt.yscale("log")


if __name__ == '__main__':
    run()
