import numpy as np
from math import pi

class SinusNoise:

    min_theta = -10_000
    max_theta = 10_000

    def __init__(self, maxAction, minAction, numActions, settings):
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

class RectNoise:

    def __init__(self, maxAction, minAction, numAction, settings):
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


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import sys

    noise = "rect"

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
    else:
        raise ValueError(f"Noise with name {noise} is not supported")

    t = np.arange(0, 2000, 0.2)
    
    actions = [noise(t[i]) for i in range(t.size)]
    actions = np.array(actions).T
    plt.xlabel("Time (s)")
    plt.ylabel("Injected current (mA)")
    plt.plot(t / 1000, actions[0], label="action1")
    plt.plot(t / 1000, actions[1], label="action2")
    plt.legend()
    plt.show(block=True)
    
    
