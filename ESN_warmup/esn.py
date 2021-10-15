import numpy as np
import math
from matplotlib import pyplot as plt

def mul(m1, m2):
    return np.matmul(m1, m2)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class ESN:

    scale_input = 1.8
    scale_bias = 0.2
    spectral_radius = 1.1
    params_mu = 0
    ridge_k = 0.0001
    wash_out = 0
    max_param = 1
    min_param = -1

    def __init__(self, dataPath, nReservoir):

        self.dataPath = dataPath
        self.trainset = self.loadData("train.npy") 
        self.testset = self.loadData("test.npy")

        self.nNeurons = nReservoir
        n = self.nNeurons

        self.state = None
        self.stateWeights = self.get_state_weights()
        self.trainable = self.gaussianParams(n)
        self.bias = self.gaussianParams(n)  * ESN.scale_bias
        self.input = self.gaussianParams(n) * ESN.scale_input
        self.history = None
        self.predictions = None

        self.t = 0
    
    def get_spectral_radius(self, matrix):
        eigens = np.resize(np.linalg.eigvals(matrix), (matrix.size, 1))
        sR = np.linalg.norm(np.max(eigens))
        return sR

    def get_state_weights(self):
        weights = self.gaussianParams(self.nNeurons, self.nNeurons)
        weights *= ESN.spectral_radius / self.get_spectral_radius(weights)
        return weights


    def saveModel(self):
        self.save(self.history, "history")
        self.save(self.trainable, "trainable")
        self.save(self.stateWeights, "stateWeights")
        self.save(self.input, "inputWeights")
        self.save(self.bias, "bias")
        self.save(self.predictions, "predictions")
        self.save(self.trainset.T[1], "labels")
        self.save(self.trainset.T[0], "inputs")

    def getRandomState(self):
        return self.initParams(self.nNeurons)

    def gaussianParams(self, x1size, x2size=1):
        return np.random.normal(size=(x1size, x2size), loc=ESN.params_mu) 

    def initParams(self, x1size, x2size=1):
        #return np.random.randn(x1size, x2size) * ESN.max_param
        return (np.random.randn(x1size, x2size) * (ESN.max_param - ESN.min_param)) + ESN.min_param

    def loadData(self, name):
        return np.load(self.dataPath + "/" + name)

    def sample(self, set):
        return (set[self.t][0], set[self.t][1])

    def update(self, x):
        self.state = np.tanh(mul(self.stateWeights, self.state)  +  self.input * x + self.bias)
        self.t += 1

    def output(self):
        return np.resize(self.state, self.nNeurons).dot(self.trainable)

    def loss(self, ys, yhats):
        SE = np.power(ys - yhats, 2)
        MSE = np.sum(SE) / ys.size
        RMSE = MSE ** (1 / 2)
        yhats_mu = np.mean(yhats) # TODO maybe it's ys
        variance = np.sum(np.power(yhats - yhats_mu, 2)) / yhats.size
        NRMSE = (MSE / variance) ** (1 / 2)
        return (NRMSE, RMSE) 

    def harvest(self):
        history = None
        for i in range(self.trainset.shape[0]):
            x, _ = self.sample(self.trainset)
            self.update(x)

            if i < ESN.wash_out:
                 continue

            flatSpace = np.copy(self.state)
            flatSpace = np.resize(flatSpace, (1, self.nNeurons))
            if history is None:
                history = flatSpace
            else: 
                history = np.append(history, flatSpace, axis=0)
        
        # TODO is this supposed to be here?
        #return np.transpose(history)
        return history

    def fastTest(self):
        yhats = np.dot(self.history, self.trainable)
        ys = self.trainset.T[1][ESN.wash_out:]
        return self.loss(ys, yhats)


    def train(self, graphical=False):
        self.t = 0
        self.state = self.getRandomState()

        # Harvesting
        if graphical:
            print("Harvesting")
        history = self.harvest()
        self.history = history

        # Ridge regression
        ys = self.trainset.T[1][ESN.wash_out:]
        identity = np.identity(self.nNeurons)
        # identity[0, 0] = 0
        # similiarity = mul(history.T, history)
        # biasIdentity = ESN.ridge_k * identity
        # self.trainable = mul(mul(np.linalg.inv(similiarity + biasIdentity), history.T), ys)
        similiarity = np.dot(history.T, history)
        biasIdentity = ESN.ridge_k * identity
        self.trainable = np.dot(np.dot(np.linalg.inv(similiarity + biasIdentity), history.T), ys)

        #print("Fast test loss:")
        #print(self.fastTest())

        print("Train loss")
        loss = self.test(self.trainset, graphical=False)
        print(loss)

        return loss

    def save(self, arr, name):
        np.save("./model/" + name + ".npy", arr)

    def showPredictions(self, ys, yhats):
        py = ys[200:600]
        phats = yhats[200:600]
        xs = np.arange(200,600)

        figs, ax = plt.subplots(1, 2, figsize=(12, 8))
        ax[0].plot(xs, py)
        ax[1].plot(xs, phats)
        plt.show(block=False)

        
    def test(self, ds=None, graphical=False):
        self.t = 0

        ds = self.testset if ds is None else ds

        self.state = self.getRandomState()

        yhats = np.array([])
        
        for i in range(ds.shape[0]):
            x, _ = self.sample(ds)
            self.update(x)
            yhat = self.output() 
            yhats = np.append(yhats, yhat)

        ys = ds.T[1] 

        if graphical:
            self.showPredictions(ys, yhats)
        self.predictions = yhats

        return self.loss(ys, yhats)