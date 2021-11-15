import cell
import numpy as np
import math 
import sys

import cv2

t_start = 0
t_end = 10
dt = 0.02
stim_freq = 0.5

resting_potential = -0.08
gridx = 100
gridy = 100
tissue_resistivity = 80
surface = 0.01 # m^2
thickness = 0.0015 # m 
SV = 1 / thickness # surface to volume ratio
Cm = 1 * 10 ** (-6)

c_min = -0.09
c_max = 0.025

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

RP = np.ones((gridx, gridy)) * resting_potential

s0["V"] = np.ones((gridx, gridy), dtype=np.double) * resting_potential

s0["m"] = np.ones((gridx, gridy), dtype=np.double) * cell.steady_state(
    cell.alpha_m(RP),
    cell.beta_m(RP)
)

s0["j"] = np.ones((gridx, gridy), dtype=np.double) * cell.steady_state(
    cell.alpha_j(RP),
    cell.beta_j(RP)
)

s0["h"] = np.ones((gridx, gridy), dtype=np.double) * cell.steady_state(
    cell.alpha_h(RP),
    cell.beta_h(RP)
)

s0["d"] = np.ones((gridx, gridy), dtype=np.double) * cell.steady_state(
    cell.alpha_d(RP),
    cell.beta_d(RP)
)

s0["f"] = np.ones((gridx, gridy), dtype=np.double) * cell.steady_state(
    cell.alpha_f(RP),
    cell.beta_f(RP)
)

s0["X"] = np.ones((gridx, gridy), dtype=np.double) * cell.steady_state(
    cell.alpha_X(RP),
    cell.beta_X(RP)
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
    
    gradient = np.gradient(V, dWidth, dHeight)
    
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

    return dState
    



def solve(trajectory=False):
    duration = t_end - t_start
    steps = int(duration / dt)
    states = [s0]
    state = s0
    t = t_start

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter("./out/prop.avi", fourcc, steps / duration, (gridx,gridy))

    for _ in range(steps):
        sys.stdout.write('t=%s\r' % str(t))
        dState = dStatedt(state, t)
        newState = make_state()

        for key, stateVar in dState.items():
            newState[key] = state[key] + stateVar * dt

        if trajectory:
            states.append(newState)
        
        V = state["V"] 
        col = np.clip(
            (V - c_min) / (c_max - c_min),
            0,
            1
        )
        img = np.zeros((gridx, gridy, 3))
        img[:,:,0] = col
        img[:,:,2] = col
        video.write(img)

        state = newState
        t += dt
    
    cv2.destroyAllWindows()
    video.release()
    print()
    
    return states if trajectory else None

solve()

    

    
