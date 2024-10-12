
from file_io import *
from fitting import *
from analysis_cole import *
from config import *
import numpy as np
from plotting import *


def run():
    folders = [""]
    for folder in folders:
        filename = folder + "_data.pkl"

        results = check_and_load_or_process(
            process_folder,
            filename,
            "C:/Users/Cole Thumann/OneDrive/Desktop/LabData", TRACKS_PER_FILE, NUM_FILES, TRACK_LENGTH, TIME_BETWEEN_SAMPLES, BIN_NUM
        )

        # Plot the data
        plot_results(results, folder)

        # Fit the results

        # Run the Analysis

    plt.show()


if __name__ == '__main__':
    run()
