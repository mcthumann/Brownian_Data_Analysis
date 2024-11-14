import math
import scipy.constants as const

SIM = True
SAVE = False
BASE_PATH = r"C:\Users\Cole Thumann\Desktop\Lab_Repos\MarkovianEmbedding\position_velocity_data.csv"
# BASE_PATH = r"C:\Users\Cole Thumann\Desktop\LabData\20241107\high power_last"

NUM_FILES = 1
TRACKS_PER_FILE = 1

# PSD FIT PARAMS
MIN_BOUND=1e4
MAX_BOUND=1e7
NUM_LOG_BINS=500
# initial Guesses
K_GUESS = 1
A_GUESS = 3e-6
V_GUESS = 1

ACF = True

SINC = False
HAMMING = False

RECT_WINDOW = False
BIN = False  # IF FALSE SET BIN NUM to 1
BIN_NUM = 1

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
        print("A is " +str(a))
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
        self.gamma = 6 * math.pi * self.a * self.eta
        self.t_c = self.mass_total / self.gamma
        self.v_c = math.sqrt((const.k * 293) / self.mass_total)
        self.x_c = self.v_c * self.t_c
        self.timestep = 1.0 / self.sampling_rate
