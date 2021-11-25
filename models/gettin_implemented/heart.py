import cell
import numpy as np
import sys
import matplotlib.pyplot as plt
import json
import time

t_start = 0
t_end = 1
dt = 0.00001

stim_start = 0.05
stim_end = 0.20
stim_amplitude = 10 # uA

videoOut = True
spatial_influence = False
debug_graphs = True
track_vars = ["I_si", "I_Na", "I_K", "V", "m", "h", "j", "d", "f", "X", "X_i", "I_stim"]

resting_potential = -81.1 # mV
gridx = 2
gridy = 2
midx = gridx // 2
midy = gridy // 2
tissue_resistivity = 80
cell_size = 500 * 10 ** (-4) # cm
surface = gridx * gridy * cell_size ** 2  # cm^2
thickness = 0.15 # m 
SV = thickness # surface to volume ratio TODO maybe it's 1 / thickness
Cm = 1

c_min = -90 # mv
c_max = 10 # mV

def I_stim(t):
    stim = stim_amplitude if (t >= stim_start) and (t <= stim_end) else 0
    I = np.ones((gridx, gridy)) * stim
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
        "I_b" : None,
        "I_stim": None,
        "X_i": None
    }


def spatial_term(V):
    if not spatial_influence:
        return 0

    RN = np.roll(V, (0,-1), (0,1)) # right neighbor
    LN = np.roll(V, (0,+1), (0,1)) # left neighbor
    TN = np.roll(V, (-1,0), (0,1)) # top neighbor
    BN = np.roll(V, (+1,0), (0,1)) # bottom neighbor

    spatial = (RN - V + LN - V) / (2 * SV * cell_size ** (2) * tissue_resistivity) 
    spatial += (TN - V + BN - V) / (2 * SV * cell_size ** (2) * tissue_resistivity)
    # TODO missing the t + dt step <- how to get this? : (((( 

    return spatial
 
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
    
    o["X_i"] = cell.X_i(s["V"])

    o["I_K"] = cell.I_K(
        s["V"],
        s["X"],
        o["X_i"]
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

    return o["I_Na"] + o["I_K"] + o["I_K1"] + o["I_Kp"] + o["I_b"]  + o["I_si"]


def dStatedt(s, t):
    dState = make_state()
    o = make_other()

    # State variable update - gates
    gates(s, dState)

    # Gather data for V_m
    I_ions = ions(s, dState, o)
    I_inj = I_stim(t)
    o["I_stim"] = I_inj
    geometry = spatial_term(s["V"])

    # State variable update - V_m
    dState["V"] = (-1 / Cm) * (I_ions - I_inj - geometry)

    return dState, o
    

def fill_track(track, s, o):
    for var in track_vars:
        value = s[var][midx, midy] if var in s else o[var][midx, midy]
        track[var].append(value)

def solve(trajectory=False, videoOut=False):
    states = [s0]
    state = s0
    ts = np.arange(t_start, t_end, dt)

    track = None
    if debug_graphs:
        track = dict()
        for var in track_vars:
            track[var] = []

    if videoOut:
        grids = []

    for t in ts:
        sys.stdout.write('t=%s\r' % str(t))
        dState, other = dStatedt(state, t)
        newState = make_state()

        for key, stateVar in dState.items():
            newState[key] = state[key] + stateVar * dt

        if trajectory:
            states.append(newState)
        
        if debug_graphs:
            fill_track(track, state, other)  

        if videoOut:
            V = state["V"] 
            col = np.clip(
                (V - c_min) / (c_max - c_min),
                0,
                1
            )
            grids.append(col.tolist())

        state = newState
    
    print()

    if videoOut:
        with open("./out/video.js", 'w') as fi:
            jString = "var data = {}; var dt = {};".format(json.dumps(grids), dt)
            fi.write(jString)


    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Debug')
    axs[0 , 0].title.set_text('Vm')
    axs[0 , 0].plot(ts, track["V"])
    
    axs[1 , 0].title.set_text('I')
    axs[1 , 0].plot(ts, track["I_si"], label="I_si")
    axs[1 , 0].plot(ts, track["I_K"], label="I_K")
    axs[1 , 0].plot(ts, track["I_Na"], label="I_Na")
    axs[1 , 0].plot(ts, track["I_stim"], label="I_stim")
    axs[1 , 0].legend()

    axs[0 , 1].title.set_text('Na Gates')
    axs[0 , 1].plot(ts, track["m"], label="m")
    axs[0 , 1].plot(ts, track["h"], label="h")
    axs[0 , 1].plot(ts, track["j"], label="j")
    axs[0 , 1].legend()

    axs[1 , 1].title.set_text('K gates')
    axs[1 , 1].plot(ts, track["X"], label="X")
    axs[1 , 1].plot(ts, track["X_i"], label="X_i")
    axs[1 , 1].legend()

    plt.show(block=False)
    
    return states if trajectory else None
perf_start = time.perf_counter()
solve(videoOut=videoOut)
perf_end = time.perf_counter()
print(f"Took {perf_end - perf_start} seconds. Press [Enter] to close")
input("")



    
