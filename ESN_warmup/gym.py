import numpy as np

from multiprocessing import Pool
import tqdm
from esn import ESN

bias = 0.02

error = 1000.0
inputPar = 10.0
statePar = 10.0

def train():
    wave = ESN("./data", 100)
    wave.train(graphical=True)
    wave.saveModel()

    print("Test loss")
    print(wave.test(graphical=True))

    input("Press [Enter] to finish")

def trainParams(args):
    iPar = args[0]
    sPar = args[1]
    wave = ESN("./data", 100)
    ESN.scale_input = iPar
    ESN.scale_state = sPar
    errors = wave.train()
    
    return (errors[0], iPar, sPar) 



def optimize():
    #scales = np.arange(0.01, 0.8, 0.04)
    scales = np.arange(0.1, 0.8, 0.4)
    
    done = False
    nRuns = scales.size ** 2    
    nDone = 0
    runs = []

    for iPar in scales:
        for sPar in scales:
            runs.append([iPar, sPar])

    p = Pool(4)
    minErr = 100000
    params = None
    for result in tqdm.tqdm(p.imap_unordered(trainParams, runs), total=nRuns):
        if result[0] < minErr:
            print("New best (err, ipar, spar): " + str(result))
            minErr = result[0]
            params = result
    
    
    if params is not None:
        print("Error, ipar, spar")
        print(str(params))
    else:
        print("This is not good")


if __name__ == '__main__':
    train()

