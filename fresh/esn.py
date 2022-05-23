import numpy as np
import scipy.sparse as sp

class ESN:

    def __init__(self, rescaling):

        self.beta = 0.000001
        self.washout = 50
        self.spectral_radius = 0.5
        self.w_in_scaling = 0.7
        self.d = 2

        self.in_size = 5
        self.out_size = 1
        self.n_reservior = 200
        self.connectivity = 0.4
        self.n_readouts = self.in_size + self.n_reservior + 1

        self.leakyness_max = 0
        self.leakyness_min = 0
        self.rescaling = rescaling

        self.leaky_mask = np.random.random((self.n_reservior, 1)) * (self.leakyness_max - self.leakyness_min) + self.leakyness_min

        self.w = np.zeros((self.n_reservior, self.n_reservior))
        for i in range(self.n_reservior):
            for j in range(self.n_reservior):
                if np.random.random() <= self.connectivity:
                    self.w[i, j] = np.random.random()
        current_sr = np.max(np.abs(np.linalg.eigvals(self.w)))
        self.w = self.w / current_sr * self.spectral_radius

        # sigma = 1
        # w = sp.rand(
        #         self.n_reservior, self.n_reservior,
        #         density= 1 - self.connectivity,
        #         format="csr"
        #     ) * sigma * 2 

        # w.data = w.data - sigma
        # self.w = w.toarray()
        # current_sr = np.max(np.abs(np.linalg.eigvals(self.w)))
        # self.w = self.w / current_sr * self.spectral_radius
                
        self.w_in = np.random.random((self.n_reservior, self.in_size)) * 2 - 1
        self.w_in = self.w_in * self.w_in_scaling

        self.w_out = np.ones((self.out_size, self.n_readouts))

        self.x = self.initial_state()
        self.u = None
        self.train_state = None

        self.us = []
        self.neurons = []
    
    def test(self, targets, update):

        N = 10_000
        self.x = self.initial_state()

        state = np.array([
            [0.0],
            [np.pi]
        ])

        self.trajectory = []

        for i in range(N):
            phiDot = state[0, 0]
            phi = state[1, 0]

            u_now = np.array([
                [phiDot],
                [np.cos(phi)],
                [np.sin(phi)]
            ])

            if self.d + i >= targets.shape[0]:
                break

            u_ref = targets[i + self.d]

            action = self(u_now, u_ref)
            
            torque = action[0, 0] * self.rescaling

            if i < self.washout:
                torque = 0

            torque = np.clip(torque, -500, 500)
            
            state = update(state, torque)
            self.trajectory.append(state)
        
        

    def initial_state(self):
        return np.zeros((self.n_reservior, 1))

    def test_train(self, states, actions):
        n_samples = states.shape[0] - self.d

        ys = []
        yhats = []

        self.x = self.initial_state()

        for i in range(n_samples):
            u_now = states[i]
            y = actions[i]
            u_future = states[i + self.d]

            yhat = self(u_now, u_future)

            if i < self.washout:
                continue

            ys.append(y)
            yhats.append(yhat)

        ys = np.squeeze(np.array(ys)).T
        yhats = np.squeeze(np.array(yhats)).T
        
        print(yhats.shape, ys.shape)
        MSE = np.mean((ys - yhats) *  (ys - yhats))
        var = np.var(ys)
        NMSE = MSE / var
        NRMSE = np.sqrt(NMSE)
        
        print(f"Train error: {NRMSE}")

        return ys.T, yhats.T


    def train(self, states, actions):

        n_samples = states.shape[0] - self.d
        X = np.zeros((self.n_readouts, n_samples))
        Y = np.zeros((self.out_size, n_samples))
        head = 0

        self.x = self.initial_state()

        for i in range(n_samples):
            
            u_now = states[i]
            y = actions[i]
            u_future = states[i + self.d]

            self.neurons.append(self.x)

            self(u_now, u_future)
            self.us.append(self.u)

            if i < self.washout:
                continue
            
            X[:,head] = np.squeeze(self.train_state)
            Y[:,head] = np.squeeze(y)
            head += 1

            # XX = XX + np.matmul(self.train_state, self.train_state.T)
            # YX = YX + np.matmul(y, self.train_state.T)

        
        self.neurons = np.squeeze(np.array(self.neurons))
        self.us = np.squeeze(np.array(self.us))

        # inversed = np.linalg.inv(XX + self.beta * np.identity(self.n_readouts))
        # self.w_out = np.matmul(YX, inversed)

        X_T = X.T
        inverse = np.linalg.inv(np.matmul(X, X_T) + self.beta * np.identity(self.n_readouts))
        self.w_out = np.matmul(np.matmul(Y, X_T), inverse)

        # sampleRunlength = n_samples - self.d
        # cov_mat = np.matmul(X, X_T) / sampleRunlength
        # pVec = X * Y / sampleRunlength
        # self.w_out = np.matmul(np.linalg.inv(cov_mat), pVec).T
        # print(self.w_out.shape)
    
        
    def __call__(self, u_now, u_future):
        
        input_scaling = np.array([
            [0.0020],
            [2.0000],
            [2.0000],
            [2.0000],
            [2.0000]
        ])

        input_shift = np.array([
            [0.0000],
            [2.0000],
            [2.0000],
            [2.0000],
            [2.0000]
        ])


        self.u = np.zeros((self.in_size, 1))
        self.u[:u_now.size] = u_now
        self.u[u_now.size:] = np.array([
            [u_future[1, 0]],
            [u_future[2, 0]]
        ])

        self.u = input_scaling * self.u + input_shift
    
        self.x = np.tanh(
            np.matmul(self.w_in, self.u) +
            np.matmul(self.w, self.x) 
        )

        self.train_state = np.ones((self.n_readouts, 1))
        self.train_state[:self.u.size,:] = self.u
        self.train_state[self.u.size:-1,:] = self.x
        
        return np.matmul(self.w_out, self.train_state)
