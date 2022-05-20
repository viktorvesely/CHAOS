from re import S
import numpy as np

def dsdt(state, torque):
    dt = 0.01
    r = 1
    m = 1
    g = 9.8
    D = 0.1

    phi = state[1, 0]
    phiDot = state[0, 0]

    new_state = state +  dt * np.array([
        [ torque / (r * m) + g * np.sin(phi) / r - D * phiDot ],
        [ phiDot ]
    ])

    new_state[1, 0] = new_state[1, 0] % (2 * np.pi)

    return new_state


def get_PID_targets(N):

    Nwarmup = int(np.floor(N / 10))
    Nright = int(np.floor(4.5 * N / 10))
    target = np.pi * np.ones(Nwarmup)
    target2 = np.mod(np.pi + 2 * np.pi * np.arange(1, Nright + 1) / Nright, 2 * np.pi)
    target3 = np.mod(np.pi + 2 * np.pi * np.arange(Nright, -1, -1) / Nright, 2 * np.pi)
    
    target = np.append(target, target2)
    target = np.append(target, target3)

    return target

def get_noise_seq(N, duration):
    noiseseq = np.zeros(N)
    thisnoise = 0
    for i in range(N):
        if np.random.random() < (1 / duration):
            thisnoise = np.random.random()
        noiseseq[i] = thisnoise

    return noiseseq

def generate_training_data(N):
    
    torqueNoiseLevel = 30; torqueNoiseDuration = 1;
    Pgain = 10; Dgain = 100; Igain = 0; Ileak = 0.0;

    state = np.array([
        [0],
        [np.pi]
    ])

    states = []
    actions = []

    c_err = p_err = i_err = 0

    noise = torqueNoiseLevel * (get_noise_seq(N, torqueNoiseDuration) - 1)
    targets = get_PID_targets(N)

    for t in range(N):
    
        p_err = c_err
        c_err = targets[t] - state[1, 0]

        if c_err > np.pi:
            c_err = c_err - 2 * np.pi
        elif c_err < - np.pi:
            c_err = c_err + 2 * np.pi

        i_err = (1 - Ileak) * i_err + c_err
        PIDout = (
            Pgain * c_err + 
            Dgain * (c_err - p_err) + Igain * i_err
        )
        
        torque = PIDout + noise[t]
        state = dsdt(state, torque)
        states.append(state)
        actions.append(torque)
    
    states = np.array(states)
    
    statesCart = np.zeros((states.shape[0], 3, 1))
    statesCart[:,0,:] = states[:, 0, :]
    statesCart[:,1,:] = np.cos(states[:, 1, :])
    statesCart[:,2,:] = np.sin(states[:, 1, :])
    
    actions = np.array(actions)
    a_min = np.min(actions)
    a_max = np.max(actions)
    normalizing_const = ((a_max - a_min) / 2)
    actions = actions / normalizing_const
    print(f"Rescaling factor {normalizing_const}")
    print(f"New min: {np.min(actions)}, max: {np.max(actions)}")
    actions = np.reshape(actions, (actions.shape[0], 1, 1))

    return statesCart, actions




    
    


    
    

