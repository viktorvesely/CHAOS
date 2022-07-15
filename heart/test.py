import noise
from settings import Params
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Pool

def test_actions(no):

    pars = Params("./params.json")
    period = 1 / pars.get("sampling_frequency")

    settings = pars.get("noise_white_settings")

    injectors = tuple(
        np.array(pars.get("injectors")).T
    )

    num_actions = len(injectors[0])

    actor = noise.WhiteNoise(
        pars.get("min_action"),
        pars.get("max_action"),
        num_actions,
        settings
    )

    ts = np.arange(0, 2_000, period * 1000)

    actions = [actor(t) for t in ts]

    return actions



if __name__ == "__main__":
    fig, ax = plt.subplots(2, 1, figsize=(12,8), dpi=100)
    
    with Pool(2) as p:
        res = p.map(test_actions, [None, None])

    ax[0].plot(res[0])
    ax[1].plot(res[1])

    plt.show()