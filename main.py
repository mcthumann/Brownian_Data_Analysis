import numpy as np

from file_io import *
from fitting import *
from plotting import *
from config import *
from series_processing import process_series


def run():

    folders = [""]
    for folder in folders:
        filename = folder + "_data.pkl"

        traces = check_and_load_or_process(
            filename,
            BASE_PATH + folder, TRACKS_PER_FILE, NUM_FILES
        )
        results = []

        for trace in traces:
            conf = Config(**trace['args'])
            result = process_series(trace['series'], conf)
            results.append(result)

        if SAVE:
            save_results(results, filename)

        # # Plot the data
        plot_results(results, folder)
        for m in np.logspace(-14, -12, 20):
            K = 1e-1  # Example trap strength
            a = 3e-6  # Example particle radius in meters
            # m = 3.8170350741115986e-14

            times = np.logspace(-10, -5, 6000)
            # Calculate PSD
            vacf_values = VACF_fitting_func(times, m, K, a)

            # Plot
            plt.plot(times, vacf_values, label='mass'+str(m))
        # plt.legend()
        plt.xscale('log')

        plt.title('VACF 2 function test')
        plt.show()

        # Fit the results
        fit_data(results)
        plt.show()

        # Run the Analysis


    plt.show()


if __name__ == '__main__':
    run()
