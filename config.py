import math
import scipy.constants as const

SAVE = False
BASE_PATH = r"C:\Users\Cole Thumann\Desktop\Lab_Repos\MarkovianEmbedding\position_velocity_data.csv"

SINC = False
HAMMING = False

RECT_WINDOW = False
BIN = False # IF FALSE SET BIN NUM to 1
BIN_NUM = 1

NUM_FILES = 3

ACF = True

SIM = True

a = 3e-6 # Particle size
eta = 1e-3 # Viscosity of water
rho_silica = 2200 # Density of silica
rho_f = 1000 # Density of water
mass = (4 / 3) * math.pi * (a / 2.0) ** 3 * rho_silica
mass_total = mass + .5 * (4 / 3) * math.pi * (a/2.0)** 3 * rho_f # Mass plus added mass
gamma = 6*math.pi*a*eta
timestep = 1e-4
stop= -5
start = -10
tao_c = (mass_total/gamma)
v_c = math.sqrt((const.k*293)/mass_total)
x_c = v_c*tao_c
scale = ((x_c**2)*tao_c)


TRACKS_PER_FILE = 1
TRACK_LENGTH = int((10**stop)/(timestep*tao_c)) #2**15 - 100

SAMPLING_RATE = 1.0 /(timestep*tao_c)   #200000000
TIME_BETWEEN_SAMPLES = 1.0 / SAMPLING_RATE

# NUM_LOG_BINS = 100 #???
# CUTOFF_FREQUENCY = np.log10(10**5) #???
# STARTING_FREQUENCY = np.log10(10*3) #???