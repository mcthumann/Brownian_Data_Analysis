
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

        # Fit the results
        fit_data(results)
        plt.show()

        # Run the Analysis


    plt.show()


if __name__ == '__main__':
    run()
