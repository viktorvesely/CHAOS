import numpy as np
import sys
import matplotlib.pyplot as plt
import json
import time
import cProfile
import pstats

import cell
import solver
import resistivity
 

#----------SHARED CONSTANTS ACROSS RUNS------------

# What solver should the program use
euler = True
periodicX = False
periodicY = False

# Video params
every_nth_frame = 200
c_min = -82 # mv
c_max = 40 # mV

# Debugging
debug_graphs = False 
track_vars = ["I_si", "I_Na", "I_K", "V", "m", "h", "j", "d", "f", "X", "X_i", "I_stim"]

# Physic constants
resting_potential = -81.1014 # mV
dt = 0.015 # ms
dx = 200 * 10 ** (-4) # cm
dy = dx
thickness = 0.08 # cm, https://www.ncbi .nlm.nih.gov/pmc/articles/PMC5841556/
SV = 0.24 * 10 ** (4) # surface to volume ratio 
Cm = 1

gate_list = ["m", "h", "j", "d", "f", "X"]
 
#--------------Solving functions----------------

def heartbeat(t, I, BPS, stim_start, stim_end, stim_amplitude):
    T = t % (1000 / BPS)
    stim = stim_amplitude if (T >= stim_start) and (T <= stim_end) else 0
    I[0, 0] = stim


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


def dStatedt(
    s,
    t,
    rhoDx,
    rhoDy,
    gridx,
    gridy,
    BPS,
    stim_start,
    stim_end,
    stim_amplitude,
    action=0
    ):
    dState = make_state()
    o = make_other()

    # State variable update - gates
    gates(s, dState)

    # Gather data for V_m
    I_ions = ions(s, dState, o)
    I_stim = np.zeros((gridx, gridy))
    heartbeat(t, I_stim, BPS, stim_start, stim_end, stim_amplitude)
    o["I_stim"] = I_stim + action

    if euler:
        dState["V"] = solver.euler_solve(
            V=s["V"],
            I_ions=I_ions,
            I_stim=o["I_stim"],
            Sv=SV,
            C=Cm,
            rhoDx=rhoDx,
            rhoDy=rhoDy,
            dt=dt,
            dx=dx,
            dy=dy,
            periodicX=periodicX,
            periodicY=periodicY
        )
    else:
        # Solve with ADI method
        dState["V"] = solver.solve(
            V=s["V"],
            I_ions=I_ions,
            I_stim=o["I_stim"],
            Sv=SV,
            C=Cm,
            rhoDx=rhoDx,
            rhoDy=rhoDy,
            dt=dt,
            dx=dx,
            dy=dy,
            periodicX=periodicX,
            periodicY=periodicY
        )



    return dState, o
    

def fill_track(track, s, o):
    for var in track_vars:
        value = s[var][0, 0] if var in s else o[var][0, 0]
        track[var].append(value)

def solve(params, videoOut=False, verbal=False, onTick=None):

    
    t_start = params.get("t_start")
    t_end = params.get("t_end")
    t_duration = t_end - t_start

    BPS = params.get("BPS")
    rythm = (1000 / BPS)
    stim_start = params.get("stim_start")
    stim_end = params.get("stim_end")
    stim_amplitude = params.get("stim_amplitude")    

    if (stim_start > stim_end or
        stim_end > rythm or
        stim_start < 0):
        raise ValueError(f"Incorrect parameters. Change BPS: {BPS}, stim_start: {stim_start}, or stim_end: {stim_end} so it makes sense")

    gridx = params.get("gridx")
    gridy = params.get("gridy")

    rhoDx, rhoDy = resistivity.get_resistivity_masks(
        (gridx, gridy),
        (params.get("min_resistivity"), params.get("max_resistivity")), 
        params.get("resistivity_path")
    )

    # ----------Initialize state 0------------------
    s0 = make_state()

    RP = np.ones((gridx, gridy)) * resting_potential

    s0["V"] = np.ones((gridx, gridy), dtype=np.float) * resting_potential

    s0["m"] = np.ones((gridx, gridy), dtype=np.float) * cell.steady_state(
        cell.alpha_m(RP),
        cell.beta_m(RP)
    )

    s0["j"] = np.ones((gridx, gridy), dtype=np.float) * cell.steady_state(
        cell.alpha_j(RP),
        cell.beta_j(RP)
    )

    s0["h"] = np.ones((gridx, gridy), dtype=np.float) * cell.steady_state(
        cell.alpha_h(RP),
        cell.beta_h(RP)
    )

    s0["d"] = np.ones((gridx, gridy), dtype=np.float) * cell.steady_state(
        cell.alpha_d(RP),
        cell.beta_d(RP)
    )

    s0["f"] = np.ones((gridx, gridy), dtype=np.float) * cell.steady_state(
        cell.alpha_f(RP),
        cell.beta_f(RP)
    )

    s0["X"] = np.ones((gridx, gridy), dtype=np.float) * cell.steady_state(
        cell.alpha_X(RP),
        cell.beta_X(RP)
    )

    s0["Ca_i"] = np.ones((gridx, gridy), dtype=np.float) * cell.Ca_i_initial

    # ----------SIM_START---------------
    state = s0
    ts = np.arange(t_start, t_end, dt)

    track = None
    if debug_graphs:
        track = dict()
        for var in track_vars:
            track[var] = []

    if videoOut:
        grids = []

    next_time = 0
    next_step = (t_duration * 0.05) / 1000

    next_frame = 0
    
    on_tick_listener = onTick is not None

    for i, t in enumerate(ts):

        if verbal and time.time() >= next_time:
            sys.stdout.write('t=%s\r' % str(t))
            next_time = time.time() + next_step 

        action = 0
        if on_tick_listener:
            action = onTick(state["V"], t)

        dState, other = dStatedt(
            state,
            t,
            rhoDx,
            rhoDy,
            gridx,
            gridy,
            BPS,
            stim_start,
            stim_end,
            stim_amplitude,
            action=action
        )
        
        newState = make_state()

        for key, stateVar in dState.items():
            if key == "V":
                # From implicit solver
                newState[key] = stateVar
                continue

            newState[key] = state[key] + stateVar * dt
        
        if debug_graphs:
            fill_track(track, state, other)  

        if videoOut and i >= next_frame:
            V = state["V"] 
            col = np.clip(
                (V - c_min) / (c_max - c_min),
                0,
                1
            )
            grids.append(col.tolist())
            next_frame += every_nth_frame

        state = newState
    
    if verbal:
        print("\n")

    if videoOut:
        with open("./roentgen/video.js", 'w') as fi:
            jString = "var data = {}; var dt = {};".format(json.dumps(grids), (dt * every_nth_frame))
            fi.write(jString)

    if debug_graphs:
        V_max = np.max(track["V"])
        print(f"Vmax_mid = {V_max}")

        fig, axs = plt.subplots(2, 2)
        fig.suptitle('Debug')
        axs[0 , 0].title.set_text('Vm')
        axs[0 , 0].plot(ts, track["V"])
        
        axs[1 , 0].title.set_text('I')
        axs[1 , 0].plot(ts, track["I_si"], label="I_si")
        axs[1 , 0].plot(ts, track["I_K"], label="I_K")
        #axs[1 , 0].plot(ts, track["I_Na"], label="I_Na")
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



#perf_start = time.perf_counter()
#with cProfile.Profile() as pr:
#solve(videoOut=videoOut)
#perf_end = time.perf_counter()

#stats = pstats.Stats(pr)
#stats.sort_stats(pstats.SortKey.TIME)
#stats.print_stats()
#print(f"Solve time: {perf_end - perf_start} seconds.")

# if debug_graphs:
#     input("Press [Enter] to close")



    
