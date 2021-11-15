import numpy as np
import math


e = math.e
gridx = 200
gridy = 200

resting_potential = -0.080

state = np.ones((gridx, gridy)) * resting_potential

K_o = 0.0054
K_i = 0.145
Na_i = 0.018
Na_o =  0.14
Ca_o = 0.0018
Ca_i = 2 * 10 ** (-4)

g_Na = 23
E_Na = 0.0544

g_si = 0.09
E_si = None # I guess from Nernst potential

g_K = 0.282
E_K = -0.077

g_K1 = 0.6047
E_K1 = None # Around -25 mV 

g_b = 0.03921
E_b = -59.87


#--------------Currents---------------------

def I_na(V, m, h, j):
    return g_Na * (m ** 3) * h * j * (V - E_Na)

def I_si(V, d, f):
    return g_si * d * f * (V - E_si)

def I_K(V, X, X_i, G_bar):
    return G_bar * X * X_i * (V - E_K)

def I_K1(V, K1_inf, G_bar):
    return G_bar * K1_inf * (V - E_K1)

def I_Kp(V, Kp):
    return 0.0183 * Kp * (V - E_K1)

def I_b(V):
    return g_b * (V + E_b)

#---------------Taus+Steady-----------------

def tau(alpha, beta):
    return 1 / (alpha + beta)

def steady_state(alpha, beta):
    return alpha / (alpha + beta)

def dGatedt(gate, alpha, beta, V):
    a = alpha(V)
    b = beta(V)
    return (steady_state(a, b) - gate) / tau(a, b)

#---------------Others----------------------

def dCa_idt(I_si, Ca_i):
    return -10 ** (-4) * I_si + 0.07 * (10 ** (-4) - Ca_i)

def Gbar_K():
    return g_K * math.sqrt(K_o / 5.4)

def Gbar_K1():
    return g_K1 * math.sqrt(K_o / 5.4)


#--------------Rate constants---------------

def alpha_j(V):
    if V >= -0.04:
        return 0.0
    else:
        temp1 = -127140 * e ** (0.2444 * V) - 3.474 * (10 ** (-5)) * e ** (-0.04391 * V)
        temp2 = (V + 37.78) / (1 + e ** (0.311 * (V + 79.23)))
        return temp1 * temp2

def alpha_h(V):
    if V >= -0.04:
        return 0.0
    else:
        return 0.135 * e ** ((80 + V) / -6.8)


def beta_h(V):
    if V >= -0.04:
        return 1 / (0.13 * (1 + e ** ((V + 10.66) / (-11.1))))
    else:
        3.56 * e ** (0.079 * V) + 3.1 * 10 ** (-5) * e ** (0.35 * V)
        

def beta_j(V):
    if V >= -0.04:
        0.3 * e ** (-2.535 * 10 ** (-7) * V) / (1 + e ** (-0.1 * (V + 32)))
    else:
        0.1212 * e ** (-0.01052 * V) / (1 + e ** (-0.1378 * (V + 40.14)))

def alpha_m(V):
    return 0.32 * (V + 47.13) / (1 - e ** (-0.1 * (V + 47.13)))

def beta_m(V):
    return 0.08 * e ** (-V / 11)   

def alpha_d(V):
    return 0.092 * e ** (-0.01 * (V - 5)) / (1 + e ** (-0.072 * (V - 5)))

def beta_d(V):
    return 0.07 * e ** (-0.017 * (V + 44)) / (1 + e **(0.05 * (V + 44)))

def alpha_f(V):
    return 0.012 * e ** (-0.008 * (V + 28)) / (1 + e **(0.15 * (V + 28)))

def beta_f(V):
    return 0.0065 * e ** (-0.002 * (V + 30)) / (1 + e **(-0.2 * (V + 30)))

def X_i(V):
    if V > -0.1:
        return 2.837 * (e ** (0.04 * (V + 77)) - 1) / ((V + 77) * e ** (V + 35))
    else:
        return 1

def alpha_X(V):
    return 0.0005 * e ** (0.083 * (V + 50)) / (1 + e **(0.057 * (V + 50)))

def beta_X(V):
    return 0.0013 * e ** (-0.06 * (V + 20)) / (1 + e **(-0.04 * (V + 20)))

def alpha_K1(V):
    return 1.02 / (1 + e ** (0.2385 * (V - E_K1 - 59.215)))

def beta_K1(V):
    return (0.49124 * e ** (0.08032 * (V - E_K1 + 5.476)) + e ** (0.06175 * (V - E_K1 - 594.31))) / (1 + e ** (-.5143 * (V - E_K1 + 4.753)))

def Kp(V):
    return 1 / (1 + e ** ((7.488 - V) / 5.98))