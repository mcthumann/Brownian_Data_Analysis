# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:39:06 2024

@author: Jason
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt


# Current best fitting functions for MSD

class Const:
    eta = 1e-3  # Viscosity of water
    rho_f = 1000  # Density of water
    K = 10e-6
    T = 292
    k_b = scipy.constants.k
    runs = 10000
    V = 10 ** 6
    output_file_name = "full_sim_data.txt"


V_const = 10 ** 14
mass_const = 10 ** 14


def MSD_fitting_func(t, m, K, a, V):
    trap_const = K
    use_mass = m
    m_f = 2 / 3 * np.pi * a ** 3 * 1000
    t_k = (6 * np.pi * a * Const.eta) / K
    t_f = (Const.rho_f * a ** 2) / Const.eta
    t_p = m / (6 * np.pi * a * Const.eta)
    # find roots
    # a * z^4 + b * z^3 + c * z^2 + d * z + e = 0
    a_ = t_p
    b = -1 * np.sqrt(t_f)
    c = 1
    d = 0
    e = 1 / t_k

    # Coefficients array for the polynomial equation
    coefficients = [a_, b, c, d, e]

    # Find the roots
    roots = np.roots(coefficients)

    # I need to learn how to vectorize my code better
    term_1 = scipy.special.erfcx(roots[0] * np.sqrt(t)) / (
                roots[0] * (roots[0] - roots[1]) * (roots[0] - roots[2]) * (roots[0] - roots[3]))
    term_2 = scipy.special.erfcx(roots[1] * np.sqrt(t)) / (
                roots[1] * (roots[1] - roots[0]) * (roots[1] - roots[2]) * (roots[1] - roots[3]))
    term_3 = scipy.special.erfcx(roots[2] * np.sqrt(t)) / (
                roots[2] * (roots[2] - roots[1]) * (roots[2] - roots[0]) * (roots[2] - roots[3]))
    term_4 = scipy.special.erfcx(roots[3] * np.sqrt(t)) / (
                roots[3] * (roots[3] - roots[1]) * (roots[3] - roots[2]) * (roots[3] - roots[0]))

    D = Const.k_b * Const.T / (6 * np.pi * Const.eta * a)
    # Returns theoretical MSD
    return np.real(V * (2 * Const.k_b * Const.T / trap_const + 2 * Const.k_b * Const.T / (use_mass) * (
                term_1 + term_2 + term_3 + term_4)))


def select_times(t, MSD, low, high):
    # Selects time range of interest from array
    time, M = [], []
    for i in range(len(t)):
        if t[i] < high and t[i] > low:
            time.append(t[i])
            M.append(MSD[i])

    return np.array(time), np.array(M)


def MSD_fitting_not_mass(t, MSD_data, K_guess, a_guess, V_guess, m_guess):
    # Function to fit the radius, trap constant, and Volts to meter conversion
    initial_guess = [V_guess, a_guess, K_guess]

    def least_squares_func(x):
        # Fit for mass only, using K and a as constants
        K = x[2] / 10 ** 6
        m = m_guess
        V = x[0] * V_const
        a = x[1] * 10 ** -6
        vacf_model = MSD_fitting_func(t, m, K, a, V)
        # Least squares: minimize the sum of squared differences
        statistics_per_lag = np.arange(len(t), 0, -1)
        return np.sum((statistics_per_lag / np.sum(statistics_per_lag)) *((np.log(np.real(MSD_data)) - np.log(np.real(vacf_model))) ** 2))

    optimal_parameters = scipy.optimize.minimize(least_squares_func, initial_guess, method="Nelder-Mead")
    return optimal_parameters

def MSD_fitting_not_mass_radius(t, MSD_data, K_guess, a_guess, V_guess, m_guess):
    # Function to fit the radius, trap constant, and Volts to meter conversion
    initial_guess = [V_guess, K_guess]

    def least_squares_func(x):
        # Fit for mass only, using K and a as constants
        K = x[1] / 10 ** 6
        m = m_guess
        V = x[0] * V_const
        a = a_guess * 10 ** -6
        vacf_model = MSD_fitting_func(t, m, K, a, V)
        # Least squares: minimize the sum of squared differences
        statistics_per_lag = np.arange(len(t), 0, -1)
        return np.sum((statistics_per_lag / np.sum(statistics_per_lag)) *((np.log(np.real(MSD_data)) - np.log(np.real(vacf_model))) ** 2))

    optimal_parameters = scipy.optimize.minimize(least_squares_func, initial_guess, method="Nelder-Mead")
    return optimal_parameters


def MSD_fitting_just_mass(t, MSD_data, K_guess, a_guess, V_guess, m_guess):
    # Function to fit the mass given the other parapmeters of interest
    # We fit the log of the functions instead of just the functions in order to weight all the variables
    # effectively
    def fitting_func(t, ma):
        return np.log(np.abs(MSD_fitting_func(t, ma / mass_const, K_guess, a_guess, V_guess)))

    optimal_parameters = scipy.optimize.curve_fit(fitting_func, t, np.log(MSD_data), p0=m_guess, bounds=(10 ** -2, 10**3))
    return optimal_parameters


def multiple_loops(loop_num, t, MS, K, a, V, M, long_time_bounds = (2e-5, 2e-2), short_time_bounds=(0, 2e-5), fit_a = False):
    # This is the main control function for the fitting procedure
    m_array = []
    for j in range(loop_num):
        print(f"K = {K}, a= {a}, v = {V}, m={M}")
        # Gets the long/intermediate time scales of interest to fit the radius, V to m, and trap constant
        t_l, long_time = select_times(t, MS,long_time_bounds[0], long_time_bounds[1])

        # Does the fitting for the 3 long time variables
        if(fit_a == True):
            long_time_fit = MSD_fitting_not_mass(t_l, long_time, K * 10 ** 6, a * 10 ** 6, V, M)
            ls = long_time_fit.x
            V_use, a_use, K_use = ls[0], ls[1], ls[2]
        else:
            long_time_fit = MSD_fitting_not_mass_radius(t_l, long_time, K * 10 ** 6, a * 10 ** 6, V, M)
            ls = long_time_fit.x
            V_use, a_use, K_use = ls[0], a*10**6, ls[1]

        # Does the fitting for the mass just using the short-time MSD measurements
        t_s, short_time = select_times(t, MS, short_time_bounds[0], short_time_bounds[1])
        m_fit = \
        MSD_fitting_just_mass(t_s, short_time, K_use / 10 ** 6, a_use / 10 ** 6, V_use * V_const, M * mass_const)[0]

        # Updates our guesses for the next loop run
        K = K_use / 10 ** 6
        a = a_use / 10 ** 6
        V = V_use
        M = m_fit[0] / mass_const
        m_array.append(M)

        # Plots the fit to keep track of convergence - probably should be commented out in the final version
        plt.plot(t, MS, ".", label="Data")
        plt.plot(t, MSD_fitting_func(t, M, K_use / 10 ** 6, a_use / 10 ** 6, V_use * V_const), label="Fit")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.show()

    return M, a_use, K_use, V_use, np.array(m_array)










