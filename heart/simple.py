import numpy as np
from matplotlib import pyplot as plt
import os
from os.path import join
import scipy.sparse as sp

from scipy.signal import butter, lfilter, freqz

from reservoir import calc_sr

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

class Actions:

    def __init__(self, fs, block_duration, t_start, cutoff=0.9, order=6, washout=50):
        self.block_duration = block_duration
        period = 1 / fs
        self.fs = fs
        self.cutoff = cutoff
        self.order = order
        self.block_size = int(np.ceil(block_duration / period))
        self.t_start = None
        self.block = None
        self.washout = washout
        self.amp1 = self.amp2 = self.omega1 = self.omega2 = self.phi1 = self.phi2 = 0
        self.ampNoise = 0
        self.head = self.washout
        self.generate_block(t_start)

    
    def get_data(self, t_start):
        t = np.linspace(t_start, t_start + self.block_duration, self.block_size + self.washout)
        return (
            self.amp1 * np.sin(self.omega1 * t + self.phi1) + 
            self.amp2 * np.sin(self.omega2 * t + self.phi2) +
            self.ampNoise * np.random.random(self.block_size + self.washout)
        )
    
    def r(self, _max, _min):
        return np.random.random() * (_max - _min) + _min 

    def generate_block(self, t_start):
        self.amp1 = self.r(2.5, 0.5)
        self.amp2 = self.r(0.7, 0.1)
        self.omega1 = self.r(0.5, 0.08)
        self.omega2 = self.r(0.1, 0.01)
        self.phi1 = self.r(2 * np.pi, 0.0)
        self.phi2 = self.r(2 * np.pi, 0.0)
        self.ampNoise = self.r(3.5, 1.0)

        self.head = self.washout
        self.block = self.get_data(t_start)

        self.block = butter_lowpass_filter(self.block, self.cutoff, self.fs, self.order)
    
    def __call__(self, t):
        if self.head >= self.block.size:
            self.generate_block(t)

        value = self.block[self.head]
        self.head += 1

        return value

def dsdt(state, t, a, b, omega, action):
    x, y = state
    delta = np.array([
        y - a * ((x * x * x) / 3 - x),
        - x + b * np.cos(t * omega) + action
    ])
    # delta = np.array([
    #     y,
    #     - x - a * (x * x - 1) * y + b * np.cos(t * omega)
    # ])
    return delta

def driver(t, omega=3.37015, b=5.0):
    return b * np.cos(t * omega)

def no_action(t):
    return 0.0

def solve(
    t_start,
    t_end,
    pertubance=0,
    dt=0.02,
    actor=None,
    a = 5,
    b = 5,
    omega = 3.37015
):
    
    state = np.array([0.5, 0.0]) + pertubance
    trajectory = []
    actions = []

    if actor is None:
        actor = no_action
    
    t = t_start
    while t <= t_end:
        
        action = actor(t)
        trajectory.append(state)
        actions.append(action)
        delta = dsdt(state, t, a, b, omega, action)
        state = state + delta * dt
        t += dt
    

    trajectory = np.array(trajectory)
    actions = np.array(actions)

    return trajectory, actions


def healthy(t_end=500):
    t_start = 0
    t_end = t_end

    omega = 1
    target, _ = solve(t_start, t_end, omega=omega)

    return target[:, 0]


def showdiff(t_end=200):
    t_start = 0

    xh = healthy(t_end)
    states, _ = solve(t_start, t_end)

    plt.plot(states[:, 0], label="sad")
    plt.plot(xh, label="happy")
    plt.legend()
    plt.show()

def chaos():
    t_start = 0
    t_end = 150_00

    pertubance = np.array([-0.000005, 0.0000001])
    
    n, _ = solve(t_start, t_end)
    p, _ = solve(t_start, t_end, pertubance=pertubance)

    fig, ax = plt.subplots(2, 1, figsize=(20, 10), dpi=90)

    b = 0
    e = 150_000
    diff = np.log(np.abs(n[:,0] - p[:, 0]))
    ax[0].plot(n[b:e, 0], n[b:e, 1], label="n")
    ax[0].plot(p[b:e, 0], p[b:e, 1], label="p")
    ax[0].legend()
    ax[1].plot(diff[b:e])
    plt.show()

    #print(np.mean(diff[-501:-1]))


def architecture(pars):

    n = pars.get("n_reservior")
    w_sigma = pars.get("simple_sigma")
    spectral_radius = pars.get("spectral_radius")

    # ---------------------- W ------------------------------
    w = sp.rand(
            n,
            n, 
            density= 1 - pars.get("simple_density"),
            format="csr"
        )*  w_sigma * 2 

    w.data = w.data - w_sigma
    w = w.toarray()

    sr = calc_sr(w)

    w = (w / sr) * spectral_radius

    # ---------------------- W_in -----------------------------

    n_input = 5

    w_in_scale = pars.get("simple_w_in")
    w_in = np.random.random((n, n_input)) * 2 - 1
    w_in[:,:-1] =  w_in[:,:-1] * w_in_scale

    # Setup bias
    w_in_bias = pars.get("simple_w_in_bias") 
    w_in[:,-1] = w_in[:,-1] * w_in_bias

    # ---------------------- W_out ----------------------------
    n_output = 1
    n_readouts = n_input + n
    w_out = np.random.normal(
        0,
        0.5,
        (n_output, n_readouts)
    )

    # ---------------------- Leaky mask------------------------

    leaky_alpha_min = pars.get("leaky_alpha_min")
    leaky_alpha_max = pars.get("leaky_alpha_max")
    leaky_mask = np.random.random((n, 1)) * (leaky_alpha_max - leaky_alpha_min) + leaky_alpha_min 
    
    return w_in, w, w_out, leaky_mask


def generate_train_data(N, experiment_name, every_nth):
    
    dt = 0.02
    period = dt * every_nth
    fs = 1 / period

    t_start = 0
    t_end = N * period + dt
    actor = Actions(fs, 20, t_start)
    ts = np.linspace(t_start, t_end, N)

    states, actions = solve(t_start, t_end, dt=dt, actor=actor)
    states = states[::every_nth]
    actions = actions[::every_nth]

    path = join(os.getcwd(), "hearts", experiment_name)
    if not os.path.isdir(path):
        print(f"[{experiment_name}] Created")
        os.mkdir(path)
        os.mkdir(join(path, "data"))
    
    actions = np.reshape(actions, (-1, 1))

    np.save(join(path, 'data', 'states_0_0.npy'), states)
    np.save(join(path, 'data', 'actions_0_0.npy'), actions)

    print(f"Data generated ({states.shape[0]})")

    print(f"Min action: {np.min(actions)}, Max action: {np.max(actions)}")

    fig, ax = plt.subplots(2, 1, figsize=(12, 20), dpi=90)
    ax[0].plot(ts, states[:-1,0], label="x")
    #ax[0].plot(ts, states[:-1,1], label="y")
    ax[0].legend()
    #ax[1].plot(states[0], states[1]) 
    ax[1].plot(actions)
    plt.show()
        
    

def show_actions():
    dt = 0.02
    t_start = 0
    t_end = 100
    actor = Actions(1 / dt, 30, t_start)

    ts = np.arange(t_start, t_end, dt)
    actions = np.array([ actor(t) for t in ts])

    plt.plot(actions)
    plt.show()




if __name__ == "__main__":
    generate_train_data(20_000, "simple", every_nth=5)
    # exit()
    #healthy()
    # print(dsdt(np.array([0.5, 0.0]), 0.0, 5.0, 5.0, 3.37015, 0.0))
    #chaos()
    #showdiff()

    