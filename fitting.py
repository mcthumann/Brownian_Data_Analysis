import numpy as np
import matplotlib.pyplot as plt
from joblib.parallel import method

from config import MAX_BOUND, MIN_BOUND, NUM_LOG_BINS, K_GUESS, A_GUESS, V_GUESS, M_GUESS, SAMPLE
from scipy.optimize import minimize
import scipy
import math
import scipy.constants as const
from config import Config
from numpy.polynomial import Polynomial


# Global Parameters
class Const:
    eta = 1e-3  # Viscosity of water
    rho_f = 1000  # Density of water
    T = 293
    k_b = scipy.constants.k

def PSD_fitting_func(omega, m, K, a, V):
    # This is the PSD we look to fit.  We fit for 3 parameters
    # Namely, we fit for the trap strength K, the radius of the particle a, and the voltage to position conversion V
    gamma_s = 6 * math.pi * a * Const.eta
    tau_f = Const.rho_f * a ** 2 / Const.eta
    numerator = 2 * Const.k_b * Const.T * gamma_s * (1 + np.sqrt((1/2) * omega * tau_f))
    denominator = (m*((K/m)-omega**2) - omega * gamma_s * np.sqrt((1 / 2) * omega * tau_f)) ** 2 + omega**2 * gamma_s**2 * (
                1 + np.sqrt((1/2) * omega * tau_f))**2
    return V**2 * numerator / denominator

def VACF_fitting_func(t, m, K, a, V=1.0):
    t = t * (math.pi / 2)
    t_k = (6 * math.pi * a * Const.eta) / K
    t_f = (Const.rho_f * a ** 2) / Const.eta
    t_p = m / (6 * math.pi * a * Const.eta)
    # find roots
    # a * z^4 + b * z^3 + c * z^2 + d * z + e = 0
    a = t_p + (1 / 9.0) * t_f
    b = -np.sqrt(t_f)
    c = 1
    d = 0
    e = 1 / t_k

    # Coefficients array for the polynomial equation
    coefficients = [a, b, c, d, e]

    # Find the roots
    roots = np.roots(coefficients)

    vacf_complex = V**2 * (const.k * 293 / m) * sum(
        (z ** 3 * scipy.special.erfcx(z * np.sqrt(t))) /
        (np.prod([z - z_j for z_j in roots if z != z_j])) for z in roots
    )
    return np.real(vacf_complex)


def VACF_fitting(t, vacf_data, K, a, V, alpha):
    # This function does the fitting, fitting only for mass

    def least_squares_func(x):
        m = x[0] * 1e-14
        vacf_model = VACF_fitting_func(t, m, x[1], a, x[2])
        residuals = (vacf_data - vacf_model) * 1e12  # Rescale to avoid underflow
        # Apply exponential weighting

        weights = np.exp(-alpha * t)
        weighted_residuals = weights * residuals * 1e15  # Rescale to avoid underflow

        return np.sum(weighted_residuals ** 2)

    initial_guess = [1.0, K, V]  # Scaled initial guess
    bounds = [(1e-3, 1e6), (K*1e-4, K*1e4), (V*1e-2, V*1e2)]
    options = {
        "disp": True,  # Display iteration details
        "maxiter": 50000,
        "xatol": 1e-5,
        "fatol": 1e-5
    }
    # Optimize using least squares
    optimal_parameters = minimize(
        least_squares_func,
        initial_guess,
        method='Nelder-Mead',
        bounds=bounds,
        options=options
    )
    print(optimal_parameters.success, optimal_parameters.message)
    return optimal_parameters.x[0]*1e-14, optimal_parameters.x[1], optimal_parameters.x[2]

def PSD_fitting(freq, PSD, a, m, k, v):
    # This function does the actual fitting

    def likelihood_func(x):
        # This defines the metric that we look to minimize under the maximum likelihood
        # formalism.  The parameters that minimize this function for a given data set
        # are the most probable actual parameters for this function.  A good look
        # at its derivation for Brownian motion can be seen in Henrik's 2018 paper
        # The exact for depends on the type of noise; this one works for the gamma
        # distributed noise we expect from Brownian motion spectra
        P = PSD_fitting_func(freq * 2 * np.pi, x[0]*1e-14, x[1], a, x[2])
        return np.sum(PSD / (P) + np.log(P))

    # Note to help out the python minimization problem, we rescale our initial guesses for the parameters so
    # that they are on order unity.  I could not get this to work well without adding this feature
    optimal_parameters = minimize(likelihood_func, np.array([m*1e14, k, v]), bounds=[(m*1e12, m*1e14),(k*1e-5,k*1e5), (v*1e-5,v*1e5)], options={'disp': True})
    print(optimal_parameters.success, optimal_parameters.message)
    return optimal_parameters

def select_freq_range(freq, PSD, minimum=1, maximum=10**7):
    # If you only want to fit to part of the frequency spectrum, you can
    # fit in the PSD and this function will return the PSD just in the frequency band of interest
    freq_range = []
    PSD_range = []
    for i in range(len(PSD)):
        if freq[i] < maximum and freq[i] > minimum:
            freq_range.append(freq[i])
            PSD_range.append(PSD[i])
    return np.array(freq_range), np.array(PSD_range)


def log_bin_array(x, y, min_bound, max_bound, num_bins):
    # Define log-spaced bins
    bins = np.logspace(np.log10(min_bound), np.log10(max_bound), num_bins + 1)
    binned_x = []
    binned_y = []

    # Loop through each bin and average the PSD values within the bin range
    for i in range(len(bins) - 1):
        # Get indices within the current bin
        indices = np.where((x >= bins[i]) & (x < bins[i + 1]))[0]

        # Compute average frequency and PSD value for this bin if indices are found
        if len(indices) > 0:
            avg_freq = np.mean(x[indices])
            avg_psd = np.mean(y[indices])
            binned_x.append(avg_freq)
            binned_y.append(avg_psd)

    return np.array(binned_x), np.array(binned_y)


def fit_data(dataset, conf, avg=True):
    freqs = dataset[0]["frequency"][1:-1]
    times = dataset[0]["time"][1:-1]

    if avg:
        all_responses = np.array([item["psd"][1:-1] for item in dataset])
        vacf_ = np.array([item["v_acf"] for item in dataset])
    else:
        all_responses = dataset[0]["psd"][1:-1]
        vacf_ = dataset[0]["v_acf"]

    PSD = np.mean(all_responses, axis=0)
    vacf = np.mean(vacf_, axis=0)

    # freq_r, PSD_r = select_freq_range(freqs, PSD, 10**2, 10 **5)
    freq_r, PSD_r = log_bin_array(freqs, PSD, MIN_BOUND, MAX_BOUND, NUM_LOG_BINS)
    # freq_r, PSD_r = freqs, PSD

    # plt.plot(freq_r, PSD_r)
    # plt.title("PSD LOG BINNING")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.show()

    optimal_parameters = PSD_fitting(freq_r, PSD_r, conf.a, M_GUESS, K_GUESS, V_GUESS)
    optimal_parameters.x[0] = optimal_parameters.x[0]*1e-14
    PSD_fit = PSD_fitting_func(freqs * 2 * np.pi, optimal_parameters.x[0], optimal_parameters.x[1], conf.a, optimal_parameters.x[2])

    print("Parameters = ", optimal_parameters.x)
    plt.text(0.1, 0.1, f"Optimal Values:\nMass = {optimal_parameters.x[0]}\nK = {optimal_parameters.x[1]}\nA = {conf.a}\nV = {optimal_parameters.x[2]}",
             fontsize=8, ha='left', va='bottom', transform=plt.gca().transAxes)
    plt.text(0.1, 0.3, f"True Values:\nMass = {conf.mass_total}\nK = {.1}\nA = {conf.a}\nV = {1}",
             fontsize=8, ha='left', va='bottom', transform=plt.gca().transAxes)

    plt.plot(freqs[1:], PSD[1:])

    plt.plot(freqs[1:], PSD_fit[1:], label="fit")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

    # Log binning turned off to overweight the low lags (where mass is sensitive)
    # times_l, vacf_l = log_bin_array(times, vacf, 10**-9, 10**-4, 300)
    times_l, vacf_l = times, vacf[1:]

    alpha = 5e5  # Adjust this value based on your weighting preference
    weights = 100*np.exp(-alpha * times_l)

    optimal_mass, k_, V_ = VACF_fitting(times_l, vacf_l, 1e-1, conf.a, 1.0, alpha=alpha)
    vacf_fit = VACF_fitting_func(times, optimal_mass, k_, conf.a, V_)
    vacf_true = VACF_fitting_func(times, conf.mass_total, .1, conf.a, 1)
    # Create the plot
    fig, ax1 = plt.subplots()

    ax1.text(0.1, 0.1, f"Optimal Values:\nMass = {optimal_mass}\nK = {k_}\nA = {conf.a}\nV = {V_}",
             fontsize=8, ha='left', va='bottom', transform=plt.gca().transAxes)
    ax1.text(0.1, 0.3, f"True Values:\nMass = {conf.mass_total}\nK = {.1}\nA = {conf.a}\nV = {1}",
             fontsize=8, ha='left', va='bottom', transform=plt.gca().transAxes)

    ax1.plot(times, vacf[1:], ".")
    ax1.plot(times, vacf_fit, label="fit")
    ax1.plot(times, vacf_true, label="truth")
    ax1.set_xscale("log")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("VACF")

    # Create a secondary y-axis for weights
    ax2 = ax1.twinx()
    ax2.plot(times_l, weights, label="Weights", color="orange", linestyle="--")
    ax2.set_ylabel("Weights", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    # Add legends for both axes
    fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.85))

    # Show the plot
    plt.title("VACF Fit and Exponential Weights")
    plt.show()