import numpy as np
from matplotlib import pyplot as plt
import os
import scipy.io
from os.path import join

radius = 1
dt = 0.01
g = 9.8
D = 0.1
m = 1

def dsdt(state, torque):

    phiDot = state[0, 0]
    phi = state[1, 0]

    delta = np.array([[
        torque / (radius * m) + g * np.sin(phi) / radius - D * phiDot,
        phiDot
    ]]).T

    new_state = state + dt * delta

    new_state[1, 0] = new_state[1, 0] % (np.pi * 2)

    return new_state


def test_pendulum():
    steps = 2000
    state = np.array([[
        0, 0.1
    ]]).T

    trajectory = [ np.squeeze(state) ]
    torque = -0.5

    for _ in range(steps):
        state = dsdt(state, torque)
        trajectory.append(np.squeeze(state))

    
    trajectory = np.array(trajectory).T
    plt.plot(trajectory[0])
    plt.plot(trajectory[1])
    plt.show()


def convert_data(experiment_name):
    states = scipy.io.loadmat(join(os.getcwd(), 'data', 'states.mat'))["plantStateTrainDataCartPL"]
    actions = scipy.io.loadmat(join(os.getcwd(), 'data', 'actions.mat'))["torqueTrainDataPL"]
    
    path = join(os.getcwd(), 'hearts', experiment_name)
    if os.path.isdir(path):
        raise ValueError(f"Experiment with name {experiment_name} already exists")

    os.mkdir(path)
    os.mkdir(join(path, 'data'))

    X = states[1]
    Y = states[2]
    states = np.array([X, Y])

    np.save(join(path, 'data', 'states_0_0.npy'), states.T)
    np.save(join(path, 'data', 'actions_0_0.npy'), actions.T)
    
    
    

if __name__ == '__main__':
    convert_data("pendelum")
    


    

