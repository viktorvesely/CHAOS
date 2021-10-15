import random
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime

random.seed(datetime.now())

n_training = 4999
n_testing = 1000
training = None
testing = None

min_next_change = 30
max_next_change = 120

signal_frequency = 1/19

def rand_int(l, u):
    return math.floor(random.random() * (u - l)) + l

def get_next_change(i):
    return i + rand_int(min_next_change, max_next_change)

def get_sin(i):
    return math.sin(2 * math.pi * signal_frequency * i)

def get_rect(i):
    return 1 if get_sin(i) > 0 else 0

def gen_data(N):
    
    next_change = 0
    is_rect = random.random() >= 0.5

    data = []

    for i in range(n_training):
        if i >= next_change:
            is_rect = not is_rect
            next_change = get_next_change(i)
        
        y = int(is_rect)
        x = get_rect(i) if is_rect else get_sin(i)

        data.append((x, y))

    return data

training = np.array(gen_data(n_training))
testing = np.array(gen_data(n_testing))

training_t = np.transpose(training[0:300])
ts = np.arange(0, len(training_t[0]))

fig, axs = plt.subplots(2)
fig.suptitle('Check')
axs[0].plot(ts, training_t[0])
axs[1].plot(ts, training_t[1])

plt.show()

np.save("./data/train.npy", training)
np.save("./data/test.npy", testing)


        
