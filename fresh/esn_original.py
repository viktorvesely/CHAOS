import numpy as np
import os
import sys
import scipy.sparse as sp


class Doctor:

    def __init__(
        self
    ):       
 
        self.beta = 0.0001
        self.washout_period = 40
        self.d = 1


        self.n_reservior = 200
        n = self.n_reservior
        self.spectral_radius = 0.4
        self.connectivity = 0.2
    
        w = sp.rand(
                n,
                n, 
                density= 1 - self.connectivity,
                format="csr"
            ) * 1 * 2 

        w.data = w.data - 1
        w = w.toarray()

        sr = np.max(
            np.abs(
                np.linalg.eigvals(
                    w
                )
            )
        )

        self.w = (w / sr) * self.spectral_radius
        # ---------------------- W_in -----------------------------

        self.in_size = 6
        self.out_size = 1
 
        #n_half = int(np.ceil(n / 2))
        self.w_in = np.random.random((self.n_reservior, self.in_size)) * 2 - 1
        self.w_in = self.w_in * 0.85

        # Setup bias
        self.w_in[:,-1] = np.random.random(self.n_reservior) * 2 - 1
        self.w_in[:,-1]= self.w_in[:,-1] * 0.008

        # ---------------------- W_out ----------------------------
        self.n_readouts = self.in_size + self.n_reservior
        self.w_out = np.random.normal(
            0,
            0.5,
            (self.out_size, self.n_readouts)
        )
        self.leaky_mask = np.random.random((n, 1)) * (0.99 - 0.99) + 0.99 
        
        self.x = self.initial_state()
        self.train_state = None

        self.extenders = []

        self.XX = np.zeros((self.n_readouts, self.n_readouts))
        self.XXC = np.zeros((self.n_readouts, self.n_readouts))
        
        self.YX = np.zeros((self.out_size, self.n_readouts))
        self.YXC = np.zeros((self.out_size, self.n_readouts))
    
        
    

    def normalize_batch(self, states, actions):

        s_shape = states.shape
        states = np.reshape(states, (s_shape[0], s_shape[1], 1))
        a_shape = actions.shape
        actions = np.reshape(actions, (a_shape[0], a_shape[1], 1))

        n_samples = s_shape[0]

        return states, actions, n_samples
    
        

    def test_train(self, states, actions):
        
        print("Testing on train data")

        ys = []
        yhats = []

        n_samples = states.shape[0]
        self.x = self.initial_state()

        for i in range(self.d, n_samples - self.d):

            u_now = states[i]
            y = actions[i]
            u_future = states[i + self.d]

            yhat = self(u_now, u_future)

            if i < self.washout_period + self.d:
                continue

            yhats.append(yhat)
            ys.append(y)
        
        ys = np.squeeze(np.array(ys))
        yhats = np.squeeze(np.array(yhats))
        ys = np.reshape(ys, (1, ys.shape[0]))
        yhats = np.reshape(yhats, (1, yhats.shape[0]))
     
        print(yhats.shape)
        MSE = np.mean((yhats - ys) * (yhats - ys), axis=1)
        variances = np.var(ys, axis=1)
        NMSE = MSE / variances
        NRMSE = np.sqrt(NMSE)
        NRMSE = np.mean(NRMSE)
        print(f"Training error: {NRMSE}")

        return ys, yhats
        


    def train(self, states, actions, verbal=True):
        
        if verbal:
            print("Training network")

        n_samples = states.shape[0]
        self.x = self.initial_state()

        for i in range(0, n_samples - self.d):

            
            u_now =  states[i]
            y = actions[i]            
            u_future = states[i + self.d]

            self(u_now, u_future)

            if i < (self.washout_period + self.d):
                continue

            train_state_t = self.train_state.T
    
            self.XX = self.XX + np.matmul(self.train_state, train_state_t)
            self.YX = self.YX + np.matmul(y, train_state_t)

  
        self.calc_w_out()
        

    def calc_w_out(self):
        inversed = np.linalg.inv(self.XX + self.beta * np.identity(self.n_readouts))
        self.w_out = np.matmul(self.YX, inversed)

    def initial_state(self):
        return np.zeros((self.n_reservior, 1))

    def fast_append(self, vectorA, vectorB):
        result = np.zeros((vectorA.size + vectorB.size, 1))
        result[:vectorA.size] = vectorA
        result[vectorA.size:] = vectorB
        return result

    def fast_append_and_insert_one(self, vectorA, vectorB):
        result = np.ones((vectorA.size + vectorB.size + 1, 1))
        result[:vectorA.size] = vectorA
        result[vectorA.size:-1] = vectorB
        return result

    def triple_fast_append(self, vectorA, vectorB, vectorC):
        sa, sb, sc = vectorA.size, vectorB.size, vectorC.size
        result = np.zeros((sa + sb + sc, 1))
        result[:sa] = vectorA
        result[sa:sa+sb] = vectorB
        result[sa+sb:] = vectorC

        return result

    def compile_readouts(self, u):
        readouts = np.zeros((self.n_readouts, 1))

        base = 0
        for blueprint in self.extenders:
            addition = blueprint()
            size = addition.size
            readouts[base:base + size] = addition
            base += size

        readouts[base:base + self.x.size] = self.x
        base += self.x.size
        readouts[base:] = u

        return readouts

    def __call__(self, u_now, u_future):

        self.u_now = u_now
        self.u_future = u_future
        
        u = np.array([
            [u_now[0, 0]], [u_now[1, 0]], [u_now[2, 0]],
            [u_future[1, 0]], [u_future[2, 0]],
            [1.0]
        ])
        # u = self.fast_append_and_insert_one(u_now, u_future)

        self.x = self.x * (1 - self.leaky_mask) + self.leaky_mask * np.tanh(
            self.w_in.dot(u) +
            self.w.dot(self.x)
        )

        self.train_state = self.compile_readouts(u)

        return np.matmul(self.w_out, self.train_state)
    


    