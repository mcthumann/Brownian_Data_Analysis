import numpy as np
# Configuration

SAVE = False
BASE_PATH = r"C:\Users\Cole Thumann\Desktop\Lab_Repos\MarkovianEmbedding\position_velocity_data.csv"

TRACKS_PER_FILE = 1
TRACK_LENGTH = 2**15 - 100
SAMPLING_RATE = 200000000
TIME_BETWEEN_SAMPLES = 1.0 / SAMPLING_RATE

SINC = False
HAMMING = False

RECT_WINDOW = False
BIN = False # IF FALSE SET BIN NUM to 1
BIN_NUM = 4

NUM_FILES = 5

ACF = True

SIM = True

# NUM_LOG_BINS = 100 #???
# CUTOFF_FREQUENCY = np.log10(10**5) #???
# STARTING_FREQUENCY = np.log10(10*3) #???