import numpy as np
# Configuration

BASE_PATH = "/Users/colethumann/Desktop/20241008/with_preamp_and_higher_resistor2"
TRACKS_PER_FILE = 1
TRACK_LENGTH = 2**19 - 100
R = 200000000
TIME_BETWEEN_SAMPLES = 1.0 / R
BIN_NUM = 1
NUM_FILES = 1

ACF = False

# NUM_LOG_BINS = 100 #???
# CUTOFF_FREQUENCY = np.log10(10**5) #???
# STARTING_FREQUENCY = np.log10(10*3) #???