from re import S
from matplotlib import pyplot as plt
import time
import numpy as np
import os

from os.path import join

import esn
import pendulum
import esn_original


def p(b, w):

    e = b + w + 1

    fig, ax = plt.subplots(4, 1, figsize=(12, 10), dpi=90)

    ax[0].plot(controller.us[b:e])
    ax[0].set_title("Input")

    n_neurons = 8
    neurons = np.random.choice(controller.neurons.shape[1], size=n_neurons, replace=False)
    
    ax[1].plot(controller.neurons[b:e, neurons])
    ax[1].set_title("Neurons")

    ax[2].set_title("Output")
    ax[2].plot(ys[b:e], label="Teacher")
    ax[2].plot(yhats[b:e], label="Prediction")
    ax[2].legend()

    traj = np.squeeze(controller.trajectory)
    trajCart = np.zeros((traj.shape[0], 2))
    trajCart[:, 0] = np.cos(traj[:,1])
    trajCart[:, 1] = np.sin(traj[:,1])
    target = np.squeeze(pendulum.get_targets(10_000))
    ax[3].set_title("Test_traj")
    ax[3].plot(trajCart[b:e, 0], label="real x", linewidth=3)
    ax[3].plot(trajCart[b:e, 1], label="real y", linewidth=3)
    ax[3].plot(target[b:e, 1], label="target x")
    ax[3].plot(target[b:e, 2], label="target y")
    ax[3].legend()
    plt.show(block=True)


if __name__ == '__main__':
    
    N = 10_000

    start = time.perf_counter()
    states, actions, normalizing_const = pendulum.generate_training_data(N)
    
    # s_states = np.reshape(states, (states.shape[0], states.shape[1]))
    # s_actions = np.reshape(actions, (actions.shape[0], actions.shape[1]))

    # path = join(os.path.abspath("../"), "heart", "hearts", "pendulum", "data")
    # np.save(join(path, "states_0_0.npy"), s_states)
    # np.save(join(path, "actions_0_0.npy"), s_actions)

    controller = esn.ESN(normalizing_const)
    controller.train(states, actions)
    ys, yhats = controller.test_train(states, actions)
    controller.test(pendulum.get_targets(10_000), pendulum.dsdt)
    end = time.perf_counter()
    print(f"Time: {end - start}")
    
    
    # targets = pendulum.get_PID_targets(10_000)
    # targetCart = np.zeros((targets.shape[0], 2))
    # targetCart[:,0] = np.cos(targets)
    # targetCart[:,1] = np.sin(targets)
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(targetCart)
    # plt.show(block=False)

    p(1000, 4000)



    
    
    