from matplotlib import pyplot as plt
import time
import numpy as np

import esn
import pendulum


def p(b, w):

    e = b + w + 1

    fig, ax = plt.subplots(3, 1, figsize=(10, 6), dpi=90)

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
    plt.show(block=True)


if __name__ == '__main__':
    
    N = 10_000

    start = time.perf_counter()
    states, actions = pendulum.generate_training_data(N)
    controller = esn.ESN()
    controller.train(states, actions)
    ys, yhats = controller.test_train(states, actions)
    end = time.perf_counter()
    print(f"Time: {end - start}")
    p(100, 300)



    
    
    