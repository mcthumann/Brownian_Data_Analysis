import numpy as np
# Configuration

BASE_PATH = r"C:\Users\Cole Thumann\OneDrive\Desktop\LabData\test_3PM_small_samples_200Ms\test_3PM_small_samples_200Ms"
TRACKS_PER_FILE = 1
TRACK_LENGTH = 2**14 - 100
R = 200000000
TIME_BETWEEN_SAMPLES = 1.0 / R
BIN_NUM = 1
NUM_FILES = 70

ACF = False

# NUM_LOG_BINS = 100 #???
# CUTOFF_FREQUENCY = np.log10(10**5) #???
# STARTING_FREQUENCY = np.log10(10*3) #???