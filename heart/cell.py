import numpy as np
import math

e = math.e

alpha_beta_conversion = 1

K_o_param = 5.4
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

g_K1 =  0.6047
E_K1 = -84 # TODO verify this

g_b = 0.03921
E_b = -59.87

#--------------Currents---------------------


def I_Na(V, m, h, j):
    return g_Na * np.power(m, 3) * h * j * (V - E_Na)


def I_si(V, d, f, E_si):
    return g_si * d * f * (V - E_si)


def I_K(V, X, X_i):
    return gbar_K * X * X_i * (V - E_K)


def I_K1(V, K1_inf):
    return gbar_K1 * K1_inf * (V - E_K1)


def I_Kp(V, Kp):
    return 0.0183 * Kp * (V - E_K1)


def I_b(V):
    return g_b * (V - E_b)


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

def Gbar_K(K_o):
    global gbar_K

    gbar_K = g_K * np.sqrt(K_o / 5.4)


def Gbar_K1(K_o):
    global gbar_K1

    gbar_K1 = g_K1 * np.sqrt(K_o / 5.4)

gbar_K = None
gbar_K1 = None

def E_si(Ca_i):
    return 7.7 - 13.0287 * np.log(Ca_i)


#--------------Rate constants---------------

def alpha_j(V):
    cond = V >= -40
    vals = np.zeros(V.shape, dtype=np.float)
    temp1 = -127140 * np.exp(0.2444 * V) - 3.474 * (10 ** (-5)) * np.exp(-0.04391 * V)
    temp2 = (V + 37.78) / (1 + np.exp(0.311 * (V + 79.23)))
    np.putmask(vals, ~cond, temp1 * temp2)
    return vals * alpha_beta_conversion

def alpha_h(V):
    cond = V >= -40
    vals = np.zeros(V.shape, dtype=np.float)
    np.putmask(vals, ~cond, 0.135 * np.exp((80 + V) / -6.8))
    return vals * alpha_beta_conversion

def beta_h(V):
    cond = V >= -40
    vals = np.zeros(V.shape, dtype=np.float)
    np.putmask(vals, cond, 1 / (0.13 * (1 + np.exp((V + 10.66) / (-11.1)))))
    np.putmask(vals, ~cond, 3.56 * np.exp(0.079 * V) + 3.1 * 10 ** (5) * np.exp(0.35 * V))
    return vals * alpha_beta_conversion
        
def beta_j(V):
    cond = V >= -40
    vals = np.zeros(V.shape, dtype=np.float)
    np.putmask(vals, cond, 0.3 * np.exp(-2.535 * 10 ** (-7) * V) / (1 + np.exp(-0.1 * (V + 32))))
    np.putmask(vals, ~cond, 0.1212 * np.exp(-0.01052 * V) / (1 + np.exp(-0.1378 * (V + 40.14))))
    return vals * alpha_beta_conversion


def alpha_m(V):
    return 0.32 * (V + 47.13) / (1 - np.exp(-0.1 * (V + 47.13)))  * alpha_beta_conversion


def beta_m(V):
    return 0.08 * np.exp(-V / 11) * alpha_beta_conversion


def alpha_d(V):
    return 0.095 * np.exp(-0.01 * (V - 5)) / (1 + np.exp(-0.072 * (V - 5))) * alpha_beta_conversion


def beta_d(V):
    return 0.07 * np.exp(-0.017 * (V + 44)) / (1 + np.exp(0.05 * (V + 44))) * alpha_beta_conversion


def alpha_f(V):
    return 0.012 * np.exp(-0.008 * (V + 28)) / (1 + np.exp(0.15 * (V + 28))) * alpha_beta_conversion


def beta_f(V):
    return 0.0065 * np.exp(-0.02 * (V + 30)) / (1 + np.exp(-0.2 * (V + 30))) * alpha_beta_conversion

def X_i(V):
    cond = V > -100
    vals = np.ones(V.shape, dtype=np.float)
    np.putmask(vals, cond, 2.837 * (np.exp(0.04 * (V + 77)) - 1) / ((V + 77) * np.exp(0.04 * (V + 35))))
    return vals


def Kp(V):
    return 1 / (1 + np.exp((7.488 - V) / 5.98))


def alpha_X(V):
    return 0.0005 * np.exp(0.083 * (V + 50)) / (1 + np.exp(0.057 * (V + 50))) * alpha_beta_conversion


def beta_X(V):
    return 0.0013 * np.exp(-0.06 * (V + 20)) / (1 + np.exp(-0.04 * (V + 20))) * alpha_beta_conversion


def alpha_K1(V):
    return 1.02 / (1 + np.exp(0.2385 * (V - E_K1 - 59.215))) * alpha_beta_conversion


def beta_K1(V):
    return (0.49124 * np.exp(0.08032 * (V - E_K1 + 5.476)) + np.exp(0.06175 * (V - E_K1 - 594.31))) / (1 + np.exp(-.5143 * (V - E_K1 + 4.753))) * alpha_beta_conversion


if __name__ == '__main__':
    from heart import get_s0, dStatedt, make_state
    from matplotlib import pyplot as plt

    gridx = 1
    gridy = 1

    t_start = 0
    t_end = 200
    dt = 0.015
    
    s0 = get_s0(gridx, gridy)
    ts = np.arange(t_start, t_end, dt)
    state = s0

    Vs = []

    for t in ts:

        Vs.append(state["V"])

        dState, other = dStatedt(
            state,
            t,
            0,
            0,
            gridx,
            gridy,
            False,
            False,
            (0, 0),
            1,
            50,
            20,
            isolated=True
        )

        newState = make_state()

        for key, stateVar in dState.items():
            newState[key] = state[key] + stateVar * dt
        
        state = newState
    
    Vs = np.array(Vs)
    Vs = np.squeeze(Vs)
    plt.plot(ts, Vs)
    plt.show()
        


    

    