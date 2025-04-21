import numpy as np
import scipy.constants as constants
import scipy
import math

def hydro_msd(t, temp, roots, m):
    return 2*constants.k*temp*c_inverse_form(t, roots, m)

# NEW AND CORRECTED CODE

def a_inverse_form(t, roots, m):
    return np.real((1/m) * sum(
        (z**3 * scipy.special.erfcx(z * np.sqrt(t))) /
        (np.prod([z - z_j for z_j in roots if z != z_j])) for z in roots))

def b_inverse_form(t, roots, m):
    return np.real((1/m) * sum(
        (z * scipy.special.erfcx(z * np.sqrt(t))) /
        (np.prod([z - z_j for z_j in roots if z != z_j])) for z in roots))

def c_inverse_form(t, roots, m):
    a = roots[0]
    b = roots[1]
    c = roots[2]
    d = roots[3]

    m_over_k = 1/(a*b*c*d)

    ret = np.real((1/m) * sum(
        (scipy.special.erfcx(z * np.sqrt(t))) /
        (z*(np.prod([z - z_j for z_j in roots if z != z_j]))) for z in roots))

    return ret + m_over_k/m

def s_half_b_inverse_form(t, roots, m):
    return np.real((-1/m) * sum(
        (z**2 * scipy.special.erfcx(z * np.sqrt(t))) /
        (np.prod([z - z_j for z_j in roots if z != z_j])) for z in roots))

def s_minus_half_b_inverse_form(t, roots, m):
     return np.real((-1/m) * sum(
        (scipy.special.erfcx(z * np.sqrt(t))) /
        (np.prod([z - z_j for z_j in roots if z != z_j])) for z in roots))

def ensemble_r_term(t1, t2, m, K, roots, temp):
    return np.real((constants.k*temp)*(c_inverse_form(t1, roots, m) + c_inverse_form(t2, roots, m) - c_inverse_form(np.abs(t2-t1), roots, m) - m*b_inverse_form(t1, roots, m)*b_inverse_form(t2, roots, m) - K*c_inverse_form(t1, roots, m)*c_inverse_form(t2, roots, m)))

def e_and_f(t, mass, radius, rho, eta, x0, v0, roots, m):
    gamma = 6*np.pi*radius*eta
    z = 6*radius**2*np.pi*np.sqrt(rho*eta)
    return mass*x0*a_inverse_form(t, roots, m) + mass*v0*b_inverse_form(t, roots, m) + gamma*x0*b_inverse_form(t, roots, m) + z*x0*s_half_b_inverse_form(t, roots, m) + z*v0*s_minus_half_b_inverse_form(t, roots, m)

def x_t1_x_t2(t1, t2, m, K, radius, eta, rho_f, x0, v0, roots, temp):
    return e_and_f(t1, m, radius, rho_f, eta, x0, v0, roots, m)*e_and_f(t2, m, radius, rho_f, eta, x0, v0, roots, m) + ensemble_r_term(t1,t2, m, K, roots, temp)

def full_hydro_msd(t1, t2, m, K, radius, eta, rho_f, x0, v0, roots, temp):
    return x_t1_x_t2(t1, t1, m, K, radius, eta, rho_f, x0, v0, roots, temp) + x_t1_x_t2(t2, t2, m, K, radius, eta, rho_f, x0, v0, roots, temp) - 2 * x_t1_x_t2(t1, t2, m, K, radius, eta, rho_f, x0, v0, roots, temp)

def compute_roots(m, K, r, eta, rho_f):
    t_f = (rho_f * r ** 2) / eta
    t_p = m / (6 * np.pi * r * eta)
    a = 1
    b = -6*math.pi*r**2*np.sqrt(rho_f*eta)/m
    c = 6*math.pi*r*eta/m
    d = 0
    e = K/m

    coeffs = [a, b, c, d, e]
    return np.roots(coeffs)
