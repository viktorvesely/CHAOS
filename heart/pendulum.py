import numpy as np
from matplotlib import pyplot as plt
import os
import scipy.io
from os.path import join

radius = 1.0
dt = 0.01
g = 9.8
D = 0.1
m = 1.0

states = None
statesCart = None
actions = None
matlabStates = None
oractions = None

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


def p(b, e):
    phi = np.arctan2(statesCart[b:e, 1], statesCart[b:e, 2])
    #phiMatlab = np.arctan2(matlabStates[b:e, 1, 0], matlabStates[b:e, 2, 0])
    plt.plot(actions[b:e, 0], label="a")
    plt.plot(phi, label="phi")
    plt.plot(statesCart[b:e, 0], label="phiDot")
    #plt.plot(phiMatlab, label="matlab")
    #plt.plot(matlabStates[b:e, 0, 0], label="matlabDot")
    plt.legend()
    plt.show()

def test_pendulum():

    global states, actions, statesCart
    states = scipy.io.loadmat(join(os.getcwd(), 'data', 'states.mat'))["plantStateTrainDataCartPL"]
    actions = scipy.io.loadmat(join(os.getcwd(), 'data', 'actions.mat'))["torqueTrainDataPL"][0]

    statesCart = states.T
    statesRad = np.copy(states)
    statesRad[1] = np.arctan2(states[1], states[2])
    statesRad = statesRad[:-1,:].T

    statesRad = np.reshape(statesRad, (statesRad.shape[0], statesRad.shape[1], 1))
    states = statesRad

    state = np.array([
        [0.0],
        [np.pi]
    ])

    for i, teacher in enumerate(states):

        torque = actions[i]
        state = dsdt(state, torque)
        diff = np.abs(state - teacher)

        if (diff > 0.0001).any():
            print(f"Error at {i} tick. Difference: {np.linalg.norm(diff)}")
            return

    print("Everything is tip top")

def get_noise_seq(L, dur):
    noiseseq = np.zeros(L)
    thisnoise = 0
    for i in range(L):
        if np.random.random() < 1 / dur:
            thisnoise = np.random.random()
    
        noiseseq[i] = thisnoise

    return noiseseq

def get_PID_targets(N):

    if N / 2 != int(N / 2):
        raise ValueError("N has to be divisible by 2")

    # N = int(N / 2)

    Nwarmup = int(N / 10)
    Nright = int(4.5 * N / 10)
    # Nleft = N - Nwarmup - Nright;
    target = np.pi * np.ones(Nwarmup)
    target2 = np.mod(np.pi + 2 * np.pi * np.arange(1, (Nright + 1)) / Nright, 2 * np.pi)
    target3 = np.mod(np.pi + 2 * np.pi * np.arange(Nright, -1, -1) / Nright, 2 * np.pi)
    
    target = np.append(target, target2)
    target = np.append(target, target3)
   

    # t = np.arange(N)
    # fr = 2 * np.pi
    # target_next = (
    #     np.sin(fr * 0.00006 * t + 15.81332) * 0.8
    #     + np.cos(fr * 0.0002 * t + 9.121203123) * 0.2
    # )

    # t_min = np.min(target_next)
    # t_max = np.max(target_next)

    # target_next = (target_next - t_min) / (t_max - t_min)
    # target_next = target_next * 2 * np.pi

    # target = np.append(target, target_next)

    return target

def get_targets(N):
    
    t = np.arange(N)
    fr = 2 * np.pi
    target = (
        np.sin(fr * 0.00006 * t + 15.81332) * 0.8
        + np.random.normal(loc=0, scale=0.003, size=N)
        + np.cos(fr * 0.0002 * t + 9.121203123) * 0.2
    )

    t_min = np.min(target)
    t_max = np.max(target)

    target = (target - t_min) / (t_max - t_min)
    target = target * 2 - 1
    target = target * np.pi

    targetCart = np.zeros((N, 3))
    targetCart[:,0] = np.zeros(N)
    targetCart[:,1] = np.cos(target)
    targetCart[:,2] = np.sin(target)


    #target = np.ones(N) * np.pi / 4 + np.sin(fr * 0.0001 * t + 2.1321249) * 0.15
    target = np.ones(N) * np.pi / 4 + np.sin(fr * 0.0003 * t + 2.1321249) * 0.15
    targetCart[:,0] = np.zeros(N)
    targetCart[:,1] = np.cos(target)
    targetCart[:,2] = np.sin(target)

    return targetCart
    


def arctan2ToPendulum(phi):
    phi = np.copy(phi)
    negPhi = phi < 0
    phi[negPhi] = 2 * np.pi - phi[negPhi]
    return phi

def get_training_data(experiment_name, data_length):
    global states, actions, statesCart

    Ileak = 0
    Pgain = 10
    Dgain = 100
    Igain = 0
    noiseLevel = 30#1
    noiseDuration = 1
    
    state = np.array([
        [0.0],
        [np.pi]
    ])

    states = [ ]
    actions = []
    noise = noiseLevel * get_noise_seq(data_length, noiseDuration)
    target_states = get_PID_targets(data_length)

    cur_err = prev_err = integrated_err = 0
    
    for i in range(data_length):
        prev_err = cur_err

        phi = state[1, 0]
        cur_err = target_states[i] - phi

        if cur_err > np.pi:
            cur_err = cur_err - 2 * np.pi;
        elif cur_err < - np.pi:
            cur_err = cur_err + 2 * np.pi;
    
        integrated_err = (1 - Ileak) * integrated_err + cur_err
        PIDout = Pgain * cur_err + Dgain * (cur_err - prev_err) + Igain * integrated_err
        torque = PIDout + noise[i]

        #states.append(state)
        state = dsdt(state, torque)
        states.append(state)
        actions.append(torque)

    states = np.squeeze(np.array(states))
    
    statesCart = np.zeros((states.shape[0], 3))
    statesCart[:,0] = states[:,0]
    statesCart[:,1] = np.cos(states[:,1])
    statesCart[:,2] = np.sin(states[:,1])
    
    actions = np.array(actions)
    actions = np.reshape(actions, (-1, 1))

    a_max = np.max(actions)
    a_min = np.min(actions)

    scaling_const = ((a_max - a_min / 2))
    actions = actions / scaling_const
    print(f"New min: {np.min(actions)}; max: {np.max(actions)}")
    print(f"Rescaling constant {scaling_const}")

    path = join(os.getcwd(), 'hearts', experiment_name)
    if not os.path.isdir(path):
        os.mkdir(path)
        os.mkdir(join(path, 'data'))

    np.save(join(path, 'data', 'states_0_0.npy'), statesCart)
    np.save(join(path, 'data', 'actions_0_0.npy'), actions)

    print(f"[{experiment_name}] Data generated")

    show()

    

def convert_data(experiment_name):
    states = scipy.io.loadmat(join(os.getcwd(), 'data', 'states.mat'))["plantStateTrainDataCartPL"]
    actions = scipy.io.loadmat(join(os.getcwd(), 'data', 'actions.mat'))["torqueTrainDataPL"]
    a_min = np.min(actions)
    a_max = np.max(actions)
    actions = (actions) / (a_max - a_min) * 2 
    
    path = join(os.getcwd(), 'hearts', experiment_name)
    if not os.path.isdir(path):
        os.mkdir(path)
        os.mkdir(join(path, 'data'))

    # X = states[1]
    # Y = states[2]
    # states = np.array([X, Y])

    np.save(join(path, 'data', 'states_0_0.npy'), states.T)
    np.save(join(path, 'data', 'actions_0_0.npy'), actions.T)
    

def show():
    global states, actions
    #states = scipy.io.loadmat(join(os.getcwd(), 'data', 'states.mat'))["plantStateTrainDataCartPL"]
    #actions = scipy.io.loadmat(join(os.getcwd(), 'data', 'actions.mat'))["torqueTrainDataPL"]
    # a_min = np.min(actions)
    # a_max = np.max(actions)
    # print(a_min, a_max)
    # actions = actions / ((a_max - a_min) / 2)

    # print((a_max - a_min) / 2)

    path = join(os.getcwd(), 'hearts', 'pendulum')

    states = np.load(join(path, 'data', 'states_0_0.npy'))
    actions = np.load(join(path, 'data', 'actions_0_0.npy'))

    begin = 0
    window = 10_000
    end = begin + window

    #plt.plot(states[0][begin:end], label="phidot")
    # plt.plot(np.arccos(X[begin:end]), label="phi")
    # plt.plot(target[begin:end], label="target")
    plt.plot(states[begin:end, 1], label="x", linewidth=1)
    plt.plot(states[begin:end, 2], label="y")
    # plt.plot(actions[0][begin:end], label="torque")
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    #convert_data("pendulum")
    #show()
    #test_pendulum()
    get_training_data("pendulum", 10_000)
    
    # targetsCart = get_targets(10_000)
    # plt.plot(targetsCart[:, 1], label="x", linewidth=4)
    # plt.plot(targetsCart[:, 2], label="y")
    # plt.legend()
    # plt.show()

    

