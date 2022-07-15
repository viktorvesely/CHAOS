import numpy as np
from math import pi
from scipy import signal

class SinusNoise:

    min_theta = -10_000
    max_theta = 10_000

    def __init__(self, minAction, maxAction, numActions, settings):
        self.sqrt2 = np.sqrt(2)
        self.f = settings['frequency']
        self.max = maxAction
        self.min = minAction
        self.numActions = numActions
        self.thetas = np.random.random(size=numActions) * (SinusNoise.max_theta - SinusNoise.min_theta) + SinusNoise.min_theta

    def __call__(self, t):
        t = t / 1_000
        range11 = (
            np.sin(self.f * 2 * pi * (t + self.thetas)) + 
            np.sin(self.f * 2 * pi * (t + self.thetas) / self.sqrt2)
        ) / 2
        range01 = (range11 + 1) / 2
        return range01 * (self.max - self.min) + self.min

# Code from https://stackoverflow.com/a/39032946/7020366
class WhiteNoise:
    
    block_duration = 12_000 # ms
    washout = 200
    amp_max = 6.0
    amp_min = 3.0
    cut_max = 5
    cut_min = 2

    def __init__(self, minAction, maxAction, numActions, settings):
        self.max = maxAction    
        self.min = minAction
        self.numActions = numActions
        self.cutoff = self.get_cut_off()
        self.order = settings['order']
        self.fs = settings['fs']
        self.mean_percentage = settings['mean_percentage']
        self.zeros = np.zeros(numActions)
        self.block = None
        self.next_block = -float('inf')
        self.t = None
        self.amp_f_start = self.amplify_factor()
        self.amp_f_end = self.amplify_factor()

    def get_cut_off(self):
        return np.random.random() * (self.cut_max - self.cut_min) + self.cut_min

    def butter_highpass(self):
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = signal.butter(self.order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_highpass_filter(self):
        b, a = self.butter_highpass()
        y = signal.filtfilt(b, a, self.block, axis=0, padtype=None)
        return y
    
    def amplify_factor(self):
        return np.random.random() * (self.amp_max - self.amp_min) + self.amp_min

    def next_amplify_range(self):
        self.amp_f_start = self.amp_f_end
        self.amp_f_end = self.amplify_factor()

    def generate_block(self):
        self.block = np.random.random((WhiteNoise.block_duration + WhiteNoise.washout, self.numActions))
        self.block = self.butter_highpass_filter()

        self.block = self.block[WhiteNoise.washout:]

        mu = np.mean(self.block, axis=0)
        self.block = self.block - mu * self.mean_percentage
        high = np.amax(self.block, axis=0)
        self.block = np.clip(self.block, self.zeros, high)
        
        transformed_range = (self.max - self.min) / high
        self.block = (self.block) * transformed_range + self.min

        self.block = amplify_actions(self.block, self.amp_f_start, self.amp_f_end)
        self.next_amplify_range()
        self.cutoff = self.get_cut_off()
    
        self.next_block = self.t + WhiteNoise.block_duration
    
    def __call__(self, t):
        
        self.t = round(t)

        if self.t > self.next_block:
            self.generate_block()
        
        index = int(self.t % WhiteNoise.block_duration)

        return self.block[index]

    


class RectNoise:

    def __init__(self, minAction, maxAction, numAction, settings):
        self.activeMu = settings['activePeriodMsMu'] 
        self.activeStd = settings['activePeriodMsStd']
        self.passiveMu = settings['passivePeriodMsMu'] 
        self.passiveStd = settings['passivePeriodMsStd']
        self.max = maxAction
        self.min = minAction
        self.next_change = np.ones(numAction) * settings['t_start']
        self.level = np.zeros(numAction)
        self.numAction = numAction
        self.ons = np.zeros(numAction)

    def sample_period(self):
        return np.random.normal(
            loc=self.activeMu,
            scale=self.activeStd,
            size=self.numAction
        ) * (1 - self.ons) + np.random.normal(
            loc=self.passiveMu,
            scale=self.passiveStd,
            size=self.numAction
        ) * self.ons

    
    def sample_level(self):
        return np.random.random(size=self.numAction) * (self.max - self.min) + self.min

    def __call__(self, t):
        
        mask = t >= self.next_change
        self.level[mask] = self.sample_level()[mask]
        self.next_change[mask] = self.sample_period()[mask] + t
        self.ons[mask] = 1 - self.ons[mask]
    
        out = self.level * self.ons
        return out

def amplify(v, t):
    v = np.copy(v)
    i = np.argmax(v)
    mag = np.linalg.norm(v)
    v[i] = v[i] + t * 10_000
    v = v / np.linalg.norm(v)
    v = v * mag
    return v

def amplify_actions(m, start_f, end_f):
    """
    Axis 0: time
    Axis 1: n_actions
    """
    fs = np.linspace(start_f, end_f, m.shape[0])
    m = np.copy(m)
    i = np.argmax(m, axis=1)
    k = np.arange(m.shape[0], dtype=int)
    mag = np.linalg.norm(m, axis=1)
    nonZ = mag != 0
    m[k,i] = m[k,i] * fs
    m[nonZ] = (m[nonZ].T / np.linalg.norm(m[nonZ], axis=1)).T
    m = (m.T * mag).T
    return m
    
    

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import sys

    noise = "white"

    if len(sys.argv) > 1:
        noise = sys.argv[1]

    if noise == "rect":
        noise = RectNoise(0, 40, 2, {
            "passivePeriodMsMu": 150,
            "activePeriodMsMu": 40,
            "passivePeriodMsStd": 15,
            "activePeriodMsStd": 9,
            "t_start": 10
        })
    elif noise == "sinus":    
        noise = SinusNoise(0, 40, 2, {"frequency": 4})
        print(noise.thetas)
    elif noise == "white":
        noise = WhiteNoise(0, 40, 4, {
            "cutoff": 4,
            "fs": 400, 
            "order": 1,
            "mean_percentage": 0.9
        })
    else:
        raise ValueError(f"Noise with name {noise} is not supported")

    t = np.arange(0, 5_000, 50)
    actions = np.array([noise(t[i]) for i in range(t.size)])
    fig, ax = plt.subplots(2, 1, figsize=(14, 6))

    indicies = t > -1
    t = t[indicies]
    n_actions = actions.shape[1]

    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Injected current (mA)")
    lines = ax[0].plot(t / 1000, actions)
    labels = [f"$a_{i + 1}$" for i in range(n_actions)]
    ax[0].legend(lines, labels)

    ax[1].plot(np.random.random(actions.shape[0] * 100))

    # amplified = amplify_actions(actions, 10)
    # ax[1].set_xlabel("Time (s)")
    # ax[1].set_ylabel("Injected current (mA)")
    # lines = ax[1].plot(t / 1000, amplified)
    # labels = [f"$a_{i + 1}$" for i in range(n_actions)]
    # ax[1].legend(lines, labels)

    plt.show()
    
    
