import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import scipy
import math


# Global Parameters
class Const:
    eta = 1e-3  # Viscosity of water
    rho_f = 1000  # Density of water
    T = 293
    k_b = scipy.constants.k

def PSD_fitting_func(omega, K, a, V):
    # This is the PSD we look to fit.  We fit for 3 parameters
    # Namely, we fit for the trap strength K, the radius of the particle a, and the voltage to position conversion V
    gamma_s = 6 * math.pi * a * Const.eta
    tau_f = Const.rho_f * a ** 2 / Const.eta
    numerator = 2 * Const.k_b * Const.T * gamma_s * (1 + np.sqrt(1 / 2 * omega * tau_f))
    denominator = (K - omega * gamma_s * np.sqrt(1 /(2 * omega * tau_f))) ** 2 + omega ** 2 * gamma_s ** 2 * (
                1 + np.sqrt(1 / 2 * omega * tau_f)) ** 2
    return V * numerator / denominator


def PSD_fitting(freq, PSD):
    # This function does the actual fitting

    def likelihood_func(x):
        # This defines the metric that we look to minimize under the maximum likelihood
        # formalism.  The parameters that minimize this function for a given data set
        # are the most probable actual parameters for this function.  A good look
        # at its derivation for Brownian motion can be seen in Henrik's 2018 papper
        # The exact for depends on the type of noise; this one works for the gamma
        # distributed noise we expect from Brownian motion spectra
        P = PSD_fitting_func(freq * 2 * np.pi, x[0], x[1], x[2])
        return np.sum(PSD / (P + 1e-15) + np.log(P + 1e-15))

    # Note to help out the python minimization problem, we rescale our initial guesses for the parameters so
    # that they are on order unity.  I could not get this to work well without adding this feature
    optimal_parameters = minimize(likelihood_func, [1.0e-5, 1.2e-6, 1.3e-3],
                                  bounds=[(1e-12, None), (1e-6, 10e-6), (1e-10, 1e20)])
    return optimal_parameters

def select_freq_range(freq, PSD, minimum=1, maximum=10**5):
    # If you only want to fit to part of the frequency spectrum, you can
    # fit in the PSD and this function will return the PSD just in the frequency band of interest
    freq_range = []
    PSD_range = []
    for i in range(len(PSD)):
        if freq[i] < maximum and freq[i] > minimum:
            freq_range.append(freq[i])
            PSD_range.append(PSD[i])
    return np.array(freq_range), np.array(PSD_range)

def fit_data(dataset, avg=True):
    freqs = dataset[0]["frequency"][1:-1]
    if avg:
        all_responses = np.array([item["responses"][1:-1] for item in dataset])
    else:
        all_responses = dataset[0]["responses"][1:-1]
    PSD = np.mean(all_responses, axis=0)

    freq_r, PSD_r = select_freq_range(freqs, PSD, 10**5, 10 **7)

    optimal_parameters = PSD_fitting(freq_r, PSD_r)
    PSD_fit = PSD_fitting_func(freqs * 2 * np.pi, optimal_parameters.x[0], optimal_parameters.x[1],
                               optimal_parameters.x[2]*1e21)

    print("Parameters = ", optimal_parameters.x)
    print(1, 1.5, 1)

    plt.plot(freqs[1:], np.abs(PSD[1:]), label="rho = 2500")
    # plt.plot(freqs2[1:], np.abs(PSD2[1:]), label = "rho = 500")
    plt.plot(freqs[1:], np.abs(PSD_fit[1:]), label="fit")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.show()