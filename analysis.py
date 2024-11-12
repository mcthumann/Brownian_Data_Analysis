from scipy.fft import fft, ifft, fftfreq
import numpy as np

def compute_VACF(self, velocity_trace, transient=0.0):
    v = velocity_trace[int(transient * len(velocity_trace)):]
    N = len(v)
    max_lag = int(self.lag_fraction * N)
    # Ensure v is a NumPy array for FFT operation
    v = np.array(v)

    f_v = fft(v, n=2 * N)
    acf = ifft(f_v * np.conj(f_v)).real[:N]
    acf = acf[:max_lag]
    acf /= np.arange(N, N - max_lag, -1)
    return acf

def autocorrelation(signal):
    print("autocorr")
    # Using FFT for efficient computation of autocorrelation
    f_signal = np.fft.fft(signal, n=2 * len(signal))
    acf = np.fft.ifft(f_signal * np.conjugate(f_signal)).real[:len(signal)]
    acf /= acf[1]
    return acf

# time = np.arange(0, len(series)) * TIME_BETWEEN_SAMPLES
#
# def gamma(omega):
#     alpha = np.sqrt(-1j * omega * tau_f)
#     lambdas = a * omega / np.sqrt(-1 * c ** 2 + 1j * omega * (bulk + 4 / 3 * shear) / density)
#     numerator = (1 + lambdas) * (9 + 9 * alpha + alpha ** 2) + 2 * lambdas ** 2 * (1 + alpha)
#     denominator = 2 * (1 + lambdas) + (1 + alpha + alpha ** 2) * lambdas ** 2 / alpha ** 2
#     return 4 * np.pi * shear * a / 3 * numerator / denominator
#
#
# def admittance(omega):
#     return 1 / (-1j * omega * m + gamma(omega) + K / (-1j * omega))
#
#
# def incompressible_admittance(omega):
#     return 1 / (-1j * omega * (m + m_f / 2) + gamma_s * (1 + np.sqrt(-1j * omega * tau_f)) + K / (-1j * omega))
#
#
# def incompressible_gamma(omega):
#     return gamma_s * (1 + np.sqrt(-1j * omega * tau_f))
#
#
# def velocity_spectral_density(omega, admit_func):
#     return 2 * k_b * T * np.real(admit_func(omega))
#
#
# def position_spectral_density(omega, admit_func):
#     return velocity_spectral_density(omega, admit_func) / omega ** 2
#
#
# def thermal_force_PSD(omega, SPD, gamma, mass):
#     G = (-1 * omega ** 2 * mass - 1j * omega * gamma + K) ** -1
#     return np.abs(G) ** -2 * SPD
#
#
# def ACF_from_SPD(admit_function, SPD_func, times):
#     low_freq = np.linspace(1, 10 ** 4, integ_points)
#     mid_freq = np.linspace(10 ** 4, 10 ** 6, integ_points)
#     high_freq = np.linspace(10 ** 6, 10 ** 9, integ_points)
#     top_freq = np.linspace(10 ** 9, 10 ** 12, integ_points)
#
#     frequencies = np.concatenate((low_freq, mid_freq, high_freq, top_freq))
#     ACF = np.zeros(len(times))
#
#     for i in range(len(times)):
#         ACF[i] = 2 * np.real(
#             scipy.integrate.simps(SPD_func(frequencies, admit_function) * np.exp(-1j * frequencies * times[i]),
#                                   frequencies)) / (2 * np.pi)
#
#     return ACF
#
#
# def ACF_from_admit(admit_func, times):
#     lowest = (10 ** -10, 1, integ_points * 2)
#     low_freq = np.linspace(1, 10 ** 4, integ_points)
#     mid_freq = np.linspace(10 ** 4, 10 ** 6, integ_points)
#     high_freq = np.linspace(10 ** 6, 10 ** 9, integ_points)
#     top_freq = np.linspace(10 ** 9, 10 ** 12, integ_points)
#
#     frequencies = np.concatenate((low_freq, mid_freq, high_freq, top_freq))
#     ACF = np.zeros(len(times))
#     admit_guy = np.real(admit_func(frequencies)) / frequencies ** 2
#     for i in range(len(times)):
#         ACF[i] = scipy.integrate.simps(np.cos(frequencies * times[i]) * admit_guy, frequencies)
#
#     return ACF
#
#
# def thermal_ACF_from_SPD(admit_func, tSPD_func, times, SPD_func, gamma, mass):
#     low_freq = np.linspace(10 ** -4, 10 ** 4, integ_points * 10)
#     mid_freq = np.linspace(10 ** 4, 10 ** 6, integ_points * 5)
#     high_freq = np.linspace(10 ** 6, 10 ** 9, integ_points)
#     top_freq = np.linspace(10 ** 9, 10 ** 12, integ_points)
#
#     # frequencies = np.concatenate((mid_freq, high_freq, top_freq))
#     frequencies = low_freq
#     ACF = np.zeros(len(times))
#     SPD = tSPD_func(frequencies, SPD_func(frequencies, admit_func), gamma(frequencies), mass)
#     for i in range(len(times)):
#         ACF[i] = 2 * np.real(
#             scipy.integrate.simps(SPD * np.exp(-1j * frequencies * times[i] * frequencies) / (2 * np.pi)))
#
#     return ACF
#
#
# def mean_square_displacement(PACF):
#     return 2 * k_b * T / K - 2 * PACF
#
#
# def calculate():
#     power = np.linspace(0, 10.5, VSP_length)
#     freq = (np.ones(VSP_length) * 10) ** power
#     VSPD_compressible = velocity_spectral_density(freq, admittance)
#     VSPD_incompressible = velocity_spectral_density(freq, incompressible_admittance)
#     PSD_incompressible = VSPD_incompressible / freq ** 2
#     PSD_compressible = VSPD_compressible / freq ** 2
#
#     TPSD_compressible = thermal_force_PSD(freq, PSD_compressible, gamma(freq), m)
#     TPSD_incompressible = thermal_force_PSD(freq, PSD_incompressible, incompressible_gamma(freq), m + 1 / 2 * m_f)
#
#     VACF_compressible = ACF_from_SPD(admittance, velocity_spectral_density, times)
#     VACF_incompressible = ACF_from_SPD(incompressible_admittance, velocity_spectral_density, times)
#
#     PACF_compressible = ACF_from_SPD(admittance, position_spectral_density, times)
#     PACF_incompressible = ACF_from_SPD(incompressible_admittance, position_spectral_density, times)
#
#     MSD_compressible = mean_square_displacement(PACF_compressible)
#     MSD_incompressible = mean_square_displacement(PACF_incompressible)
#
#     compress_correction = (k_b * T / K / PACF_compressible[0])
#     incompress_correction = (k_b * T / K / PACF_incompressible[0])
#
#     PACF_incompressible *= compress_correction
#     PACF_compressible *= incompress_correction
#
#     TPSD_compressible = thermal_force_PSD(freq, PSD_compressible, gamma(freq), m)
#     TPSD_incompressible = thermal_force_PSD(freq, PSD_incompressible, incompressible_gamma(freq), m + 1 / 2 * m_f)
#
#     return times, freq, VSPD_compressible, VSPD_incompressible, PSD_incompressible, PSD_compressible, VACF_compressible, VACF_incompressible, PACF_compressible, PACF_incompressible, TPSD_compressible, TPSD_incompressible