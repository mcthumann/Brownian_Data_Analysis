import math
import scipy.constants as const

SIM = False
SAMPLE = 1

OFFSET_ITR = 54

SAVE = False
LOAD = False
# BASE_PATH = r"C:\Users\Cole Thumann\Desktop\Lab_Repos\MarkovianEmbedding\position_velocity_data.csv"
BASE_PATH = r"\\tsclient\TESTFOLDER\20241119\long-good-noise"

NUM_FILES = 20
TRACKS_PER_FILE = 1

# PSD FIT PARAMS
MIN_BOUND=1e3
MAX_BOUND=1e7
NUM_LOG_BINS=50

# initial Guesses
K_GUESS = .1
A_GUESS = 3e-6
V_GUESS = 1000

M_GUESS = 4e-14

ACF = False

SINC = False
HAMMING = False

RECT_WINDOW = False
BIN = False  # IF FALSE SET BIN NUM to 1
BIN_NUM = 100

class Config:
    def __init__(
            self,
            sampling_rate,
            track_len,
            a=5e-6,
            eta=1e-3,
            rho_silica=2200,
            rho_f=1000,
            stop=None,
            start=None
    ):
        self.a = a
        self.eta = eta
        self.rho_silica = rho_silica
        self.rho_f = rho_f
        self.sampling_rate = sampling_rate
        self.track_len = track_len
        self.stop = stop
        self.start = start

        self.mass = (4 / 3) * math.pi * (self.a / 2.0) ** 3 * self.rho_silica
        self.mass_total = self.mass + 0.5 * (4 / 3) * math.pi * (
                    self.a / 2.0) ** 3 * self.rho_f  # Mass plus added mass
        print("MASS TOTAL IS " + str(self.mass_total))
        self.gamma = 6 * math.pi * self.a * self.eta
        self.t_c = self.mass_total / self.gamma
        self.v_c = math.sqrt((const.k * 293) / self.mass_total)
        self.x_c = self.v_c * self.t_c
        self.timestep = 1.0 / self.sampling_rate
