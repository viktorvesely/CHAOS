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
        self.mu = settings['periodMuMs'] 
        self.std = settings['periodStdMs']
        self.max = maxAction
        self.min = minAction
        self.next_change = np.ones(numAction) * settings['t_start']
        self.level = np.zeros(numAction)
        self.numAction = numAction

    def sample_period(self):
        return np.random.normal(loc=self.mu, scale=self.std, size=self.numAction)
    
    def sample_level(self):
        return np.random.random(size=self.numAction) * (self.max - self.min) + self.min

    def __call__(self, t):
        
        mask = t >= self.next_change
        self.level[mask] = self.sample_level()[mask]
        self.next_change[mask] = self.sample_period()[mask] + t
        return self.level.copy()


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    #noise = RectNoise(-1, 1, 2, {"periodMuMs": 50, "periodStdMs": 10, "t_start": 3})
    noise = SinusNoise(-1, 1, 2, {"frequency": 4})
    print(noise.thetas)
    t = np.arange(0, 2000, 0.2)

    actions = [noise(t[i]) for i in range(t.size)]
    actions = np.array(actions).T

    plt.plot(t, actions[0])
    plt.plot(t, actions[1])
    plt.show(block=True)
    
    
