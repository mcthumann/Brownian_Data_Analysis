import numpy as np
from config import *


def autocorrelation(signal):
    print("autocorr")
    # Using FFT for efficient computation of autocorrelation
    f_signal = np.fft.fft(signal, n=2 * len(signal))
    acf = np.fft.ifft(f_signal * np.conjugate(f_signal))[:len(signal)].real
    acf /= acf[0]
    return acf


def model(w, a_scale, k_trap, m):
    a = MICROSPHERE_RADIUS
    k_b = 1.380649e-23
    temp = 295
    nu = .010016
    gamma_s = 6 * np.pi * nu * a
    w_o = np.sqrt(k_trap / m)
    p = 997.97
    tao_f = (p * (a ** 2)) / nu

    # Position PSD
    psd = (2 * k_b * temp * gamma_s * (1 + np.sqrt(.5 * w * tao_f))) / (
            (m * (w_o ** 2 - w ** 2) - w * gamma_s * np.sqrt(.5 * w * tao_f)) ** 2 + (
                (w ** 2) * (gamma_s ** 2) * ((1 + np.sqrt(.5 * w * tao_f)) ** 2)))
    return a_scale * psd


def log_model(log_w, scale, log_k_trap, log_m):
    # Convert from log scale to linear scale
    a = MICROSPHERE_RADIUS
    w = np.power(10, log_w)
    m = np.power(10, log_m)
    k_trap = np.power(10, log_k_trap)

    # Constants
    k_b = 1.380649e-23
    temp = 295
    nu = 0.010016
    gamma_s = 6 * np.pi * nu * a
    w_o = np.sqrt(k_trap / m)
    p = 997.97
    tao_f = (p * (a ** 2)) / nu

    # Position PSD
    psd = (2 * k_b * temp * gamma_s * (1 + np.sqrt(0.5 * w * tao_f))) / (
            (m * (w_o ** 2 - w ** 2) - w * gamma_s * np.sqrt(0.5 * w * tao_f)) ** 2 +
            ((w ** 2) * (gamma_s ** 2) * ((1 + np.sqrt(0.5 * w * tao_f)) ** 2))
    )

    # Return the logarithm of the PSD
    return np.log(scale * psd)


def log_admittance_model(log_frequency, scale, trap_stiffness, mass):
    # Constants
    k_b = 1.380649e-23  # Boltzmann constant
    temp = 295  # Temperature
    nu = .010016  # Kinematic viscosity
    rho = 997.97  # Density of water
    radius = MICROSPHERE_RADIUS
    # Convert logarithmic inputs back to linear scale
    frequency = np.power(10, log_frequency)

    # Calculations
    gamma_s = 6 * np.pi * nu * radius
    tao_f = (rho * (radius ** 2)) / nu
    m_f = (4 / 3) * np.pi * (radius ** 3) * rho

    w = 2 * np.pi * frequency  # Angular frequency
    admittance = 1 / (-1j * w * mass + gamma_s * (1 + np.sqrt(-1j * w * tao_f)) -
                      ((1j * w * m_f) / 2) + (trap_stiffness / (-1j * w)))

    v_psd = 2 * k_b * temp * admittance.real
    p_psd = v_psd / w ** 2

    psd_position_scaled = scale * p_psd  # Scale the PSD by the factor A
    log_psd_position = np.log10(psd_position_scaled)  # Compute the logarithm of the scaled PSD
    return log_psd_position


def position_psd_clercx(freq, a_scale, k_trap, m):
    # Constants
    k_b = 1.380649e-23  # Boltzmann constant
    temp = 295  # Temperature
    nu = .010016  # Kinematic viscosity
    rho = 997.97  # Density of water
    radius = MICROSPHERE_RADIUS
    gamma_s = 6 * np.pi * nu * radius

    omega = 2 * np.pi * freq
    xi = gamma_s
    k_t = k_b * temp

    discriminant = (xi / m) ** 2 - 4 * k_trap / m
    if discriminant < 0:
        return np.zeros_like(freq)  # Return zero for non-physical parameters

    alpha = xi / m - np.sqrt(discriminant)
    beta = xi / m + np.sqrt(discriminant)

    psd_position = (k_t / m) * ((beta / (beta ** 2 + omega ** 2) - alpha / (alpha ** 2 + omega ** 2)) ** 2) / (
            (beta - alpha) ** 2 * omega ** 2)

    return a_scale * psd_position


def log_position_psd_clercx(log_freq, scale, k_trap, m):
    # Constants
    k_b = 1.380649e-23  # Boltzmann constant
    temp = 295  # Temperature
    nu = 0.010016  # Kinematic viscosity
    rho = 997.97  # Density of water
    radius = MICROSPHERE_RADIUS
    xi = 6 * np.pi * nu * radius

    # Convert logarithmic inputs back to linear scale using base 10
    freq = np.power(10, log_freq)  # Convert back to linear frequency

    omega = 2 * np.pi * freq
    k_t = k_b * temp

    discriminant = (xi / m) ** 2 - 4 * k_trap / m
    if discriminant < 0:
        return np.full_like(freq, -np.inf)  # Log of zero for non-physical parameters

    alpha = xi / m - np.sqrt(discriminant)
    beta = xi / m + np.sqrt(discriminant)

    with np.errstate(divide='ignore', invalid='ignore'):
        psd_position = (k_t / m) * ((beta / (beta ** 2 + omega ** 2) - alpha / (alpha ** 2 + omega ** 2)) ** 2) / (
                (beta - alpha) ** 2 * omega ** 2)
        psd_position[omega == 0] = np.nan  # Avoid divide by zero

    psd_position_scaled = scale * psd_position  # Scale the PSD by the factor A
    log_psd_position = np.log10(psd_position_scaled)  # Compute the logarithm of the scaled PSD
    return log_psd_position


def admittance_model(w, a_scale, k_trap, m):
    a = MICROSPHERE_RADIUS
    k_b = 1.380649e-23
    temp = 295
    nu = .010016
    gamma_s = 6 * np.pi * nu * a
    w_o = np.sqrt(m / k_trap)
    p = 997.97
    m_f = (4 / 3) * np.pi * (a ** 3) * p
    tao_f = (p * (a ** 2)) / nu

    # Position PSD
    # return (2*k_b*temp*gamma_s*(1+np.sqrt(.5*w*tao_f)))/((m*(w_o**2 - w**2) -
    # w*gamma_s*np.sqrt(.5*w*tao_f))**2 + ((w**2)*(gamma_s**2)*((1+np.sqrt(.5*w*tao_f))**2)))
    w = np.array(w, dtype=complex)
    m = np.array(m, dtype=complex)
    gamma_s = np.array(gamma_s, dtype=complex)
    tao_f = np.array(tao_f, dtype=complex)
    m_f = np.array(m_f, dtype=complex)
    k_b = np.array(k_b, dtype=complex)
    temp = np.array(temp, dtype=complex)

    # Admittance
    admittance = 1 / (
            -1j * w * m + gamma_s * (1 + np.sqrt(-1j * w * tao_f)) - ((1j * w * m_f) / 2) + (k_trap / (-1j * w)))
    admittance = np.array(admittance, dtype=complex)

    k_b = np.array(k_b, dtype=float)
    temp = np.array(temp, dtype=float)
    w = np.array(w, dtype=float)

    # Now perform the operation
    v_psd = 2 * k_b * temp * admittance.real
    v_psd = np.array(v_psd, dtype=float)
    p_psd = v_psd / w ** 2
    return a_scale * p_psd
