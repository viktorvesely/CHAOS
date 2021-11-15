import cell
import numpy as np
import math 

resting_potential = -0.08
gridx = 100
gridy = 100
tissue_resistivity = 80
rho = np.ones((gridx, gridy), dtype=np.double) * tissue_resistivity
surface = 0.01 # m^2
thickness = 0.0015 # m 
SV = 1 / thickness # surface to volume ratio
Cm = 1 * 10 ** (-6)

stim_freq = 0.5
def I_stim(t):
    I = np.zeros((gridx, gridy)) 
    I[gridx // 2, gridy // 2] = math.cos(2 * math.pi * stim_freq * t)
    return I

#---------state_variables-------------

def make_state():
    return {
        "V": None,
        "m": None,
        "j": None,
        "h": None,
        "d": None,
        "f": None,
        "X": None,
        "Ca_i": None
    }


s0 = make_state()

s0["V"] = np.ones((gridx, gridy), dtype=np.double) * resting_potential

s0["m"] = np.ones((gridx, gridy), dtype=np.double) * cell.steady_state(
    cell.alpha_m(resting_potential),
    cell.beta_m(resting_potential)
)

s0["j"] = np.ones((gridx, gridy), dtype=np.double) * cell.steady_state(
    cell.alpha_j(resting_potential),
    cell.beta_j(resting_potential)
)

s0["h"] = np.ones((gridx, gridy), dtype=np.double) * cell.steady_state(
    cell.alpha_h(resting_potential),
    cell.beta_h(resting_potential)
)

s0["d"] = np.ones((gridx, gridy), dtype=np.double) * cell.steady_state(
    cell.alpha_d(resting_potential),
    cell.beta_d(resting_potential)
)

s0["f"] = np.ones((gridx, gridy), dtype=np.double) * cell.steady_state(
    cell.alpha_f(resting_potential),
    cell.beta_f(resting_potential)
)

s0["X"] = np.ones((gridx, gridy), dtype=np.double) * cell.steady_state(
    cell.alpha_X(resting_potential),
    cell.beta_X(resting_potential)
)

s0["Ca_i"] = np.ones((gridx, gridy), dtype=np.double) * cell.Ca_i_initial


gate_list = ["m", "h", "j", "d", "f", "X"]


#----------------Other variables------------------------

def make_other():
    return {
        "I_Na" : None,
        "I_si" : None,
        "I_K" : None,
        "I_K1" : None,
        "I_Kp" : None,
        "I_b" : None
    }

def geometric(V):
    dWidth = surface / gridx
    dHeight = surface / gridy
    
    gradient = np.gradient(V, varargs=(dWidth, dHeight))
    
    geometry = np.array(gradient) / (SV * tissue_resistivity)

    return geometry
 
def gates(s, dState):
    for gate in gate_list:

        alphaF = getattr(cell, "alpha_" + gate)
        betaF = getattr(cell, "beta_" + gate)

        alpha = alphaF(s["V"])
        beta = betaF(s["V"])

        dState[gate] = cell.dGatedt(s[gate], alpha, beta)


def ions(s, dState, o):
    o["I_Na"] = cell.I_Na(
        s["V"],
        s["m"],
        s["h"],
        s["j"]
    )
    
    o["I_K"] = cell.I_K(
        s["V"],
        s["X"],
        cell.X_i(s["V"])
    )

    o["I_K1"] = cell.I_K1(
        s["V"],
        cell.steady_state(
            cell.alpha_K1(s["V"]),
            cell.beta_K1(s["V"])
        )
    )

    o["I_Kp"] = cell.I_Kp(
        s["V"],
        cell.Kp(
            s["V"]
        )
    )

    o["I_b"] = cell.I_b(
        s["V"]
    )

    o["I_si"] = cell.I_si(
        s["V"],
        s["d"],
        s["f"],
        cell.E_si(
            s["Ca_i"]
        )
    )

    # State variable update - Ca_i

    dState["Ca_i"] = cell.dCa_idt(
        o["I_si"],
        s["Ca_i"]
    )

    return o["I_Na"] + o["I_si"] + o["I_K"] + o["I_K1"] + o["I_Kp"] + o["I_b"]


def dStatedt(s, t):
    dState = make_state()
    o = make_other()

    # State variable update - gates
    gates(s, dState)

    # Gather data for V_m
    I_ions = ions(s, dState, o)
    I_inj = I_stim(t)
    geometry = geometric(s["V"])

    # State variable update - V_m
    dState["V"] = (-1 / Cm) * (I_ions - I_inj - geometry)
    



    

    
