import numpy as np
import math

e = math.e

K_o = 5.4
K_i = 145
Na_i = 18
Na_o =  140
Ca_o = 1.8
Ca_i_initial = 2 * 10 ** (-4)

g_Na = 23
E_Na = 54.4

g_si = 0.09
E_si_o = -10

g_K = 0.282
E_K = -77

g_K1 = 0.6047
E_K1 = -84 # TODO verify this

g_b = 0.03921
E_b = -59.87

transform_constant = 1

#--------------Currents---------------------

def I_Na(V, m, h, j):
    return g_Na * np.power(m, 3) * h * j * (V - E_Na) * transform_constant

def I_si(V, d, f, E_si):
    return g_si * d * f * (V - E_si) * transform_constant

def I_K(V, X, X_i):
    return gbar_K * X * X_i * (V - E_K) * transform_constant

def I_K1(V, K1_inf):
    return gbar_K1 * K1_inf * (V - E_K1) * transform_constant

def I_Kp(V, Kp):
    return 0.0183 * Kp * (V - E_K1) * transform_constant

def I_b(V):
    return g_b * (V - E_b) * transform_constant


#---------------Taus+Steady-----------------

def tau(alpha, beta):
    return 1 / (alpha + beta)

def steady_state(alpha, beta):
    return alpha / (alpha + beta)

def dGatedt(gate, alpha, beta):
    return (steady_state(alpha, beta) - gate) / tau(alpha, beta)

#---------------Others----------------------

def dCa_idt(I_si, Ca_i):
    return -10 ** (-4) * I_si + 0.07 * (10 ** (-4) - Ca_i)

def Gbar_K():
    return g_K * math.sqrt(K_o / 5.4)

def Gbar_K1():
    return g_K1 * math.sqrt(K_o / 5.4)

gbar_K = Gbar_K()
gbar_K1 = Gbar_K1()

def E_si(Ca_i):
    return 7.7 - 13.0287 * np.log(Ca_i)


#--------------Rate constants---------------

def alpha_j(V):
    cond = V >= -40
    vals = np.zeros(V.shape, dtype=np.double)
    temp1 = -127140 * np.exp(0.2444 * V) - 3.474 * (10 ** (-5)) * np.exp(-0.04391 * V)
    temp2 = (V + 37.78) / (1 + np.exp(0.311 * (V + 79.23)))
    np.putmask(vals, ~cond, temp1 * temp2)
    return vals

def alpha_h(V):
    cond = V >= -40
    vals = np.zeros(V.shape, dtype=np.double)
    np.putmask(vals, ~cond, 0.135 * np.exp((80 + V) / -6.8))
    return vals


def beta_h(V):
    cond = V >= -40
    vals = np.zeros(V.shape, dtype=np.double)
    np.putmask(vals, cond, 1 / (0.13 * (1 + np.exp((V + 10.66) / (-11.1)))))
    np.putmask(vals, ~cond, 3.56 * np.exp(0.079 * V) + 3.1 * 10 ** (5) * np.exp(0.35 * V))
    return vals
        

def beta_j(V):
    cond = V >= -40
    vals = np.zeros(V.shape, dtype=np.double)
    np.putmask(vals, cond, 0.3 * np.exp(-2.535 * 10 ** (-7) * V) / (1 + np.exp(-0.1 * (V + 32))))
    np.putmask(vals, ~cond, 0.1212 * np.exp(-0.01052 * V) / (1 + np.exp(-0.1378 * (V + 40.14))))
    return vals

def alpha_m(V):
    return 0.32 * (V + 47.13) / (1 - np.exp(-0.1 * (V + 47.13)))

def beta_m(V):
    return 0.08 * np.exp(-V / 11)   

def alpha_d(V):
    return 0.095 * np.exp(-0.01 * (V - 5)) / (1 + np.exp(-0.072 * (V - 5)))

def beta_d(V):
    return 0.07 * np.exp(-0.017 * (V + 44)) / (1 + np.exp(0.05 * (V + 44)))

def alpha_f(V):
    return 0.012 * np.exp(-0.008 * (V + 28)) / (1 + np.exp(0.15 * (V + 28)))

def beta_f(V):
    return 0.0065 * np.exp(-0.02 * (V + 30)) / (1 + np.exp(-0.2 * (V + 30)))

def X_i(V):
    cond = V > -100
    vals = np.ones(V.shape, dtype=np.double)
    np.putmask(vals, cond, 2.837 * (np.exp(0.04 * (V + 77)) - 1) / ((V + 77) * np.exp(0.04 * (V + 35))))
    return vals

def Kp(V):
    return 1 / (1 + np.exp((7.488 - V) / 5.98))

def alpha_X(V):
    return 0.0005 * np.exp(0.083 * (V + 50)) / (1 + np.exp(0.057 * (V + 50)))

def beta_X(V):
    return 0.0013 * np.exp(-0.06 * (V + 20)) / (1 + np.exp(-0.04 * (V + 20)))

def alpha_K1(V):
    return 1.02 / (1 + np.exp(0.2385 * (V - E_K1 - 59.215)))

def beta_K1(V):
    return (0.49124 * np.exp(0.08032 * (V - E_K1 + 5.476)) + np.exp(0.06175 * (V - E_K1 - 594.31))) / (1 + np.exp(-.5143 * (V - E_K1 + 4.753)))
