import numpy as np

class Doctor:

    def __init__(
        self, 
        architecture,
        beta, 
        washout, 
        w_params,
        spectral_radius,
        seed=None
    ):
        
        self.n_input, self.n_reservior, self.n_output = architecture
        self.n_readouts = self.n_input + self.n_reservior + 1

        self.beta = beta
        self.washout_period = washout
        self.washout_time = 0
        self.seed = seed
        self.spectral_radius = spectral_radius
        self.w_in_scale, self.w_in_mu, self.w_scale, self.w_mu = w_params

        np.random.seed(self.seed)
        self.w_in, self.w, self.w_out = self.construct_architecture()
        self.x = self.initial_state()
        self.train_state = None


        self.XX = np.zeros((self.n_readouts, self.n_readouts))
        self.YX = np.zeros((self.n_output, self.n_readouts))
        

    def train(self, generator):
        
        i = 0

        for sample, washout in generator:

            i += 1

            if washout:
                self.washout_time = i + self.washout_period

            if i < self.washout_period:
                continue

            u, y = sample

            self(u)
            train_state_t = doc.train_state.T
            self.XX = self.XX + np.matmul(self.train_state, train_state_t)
            self.YX = self.YX + np.matmul(y, train_state_t)
        
        inversed = np.linalg.inv(self.XX + self.beta * np.identity(self.n_readouts))
        self.w_out = np.matmul() 


        

    def initial_state(self):
        return np.zeros((self.n_reservior, 1))

    def fast_insert_one(self, vector):
        app = np.ones((vector.size + 1, 1))
        app[1:] = vector
        return app

    def fast_append_and_insert_one(self, vectorA, vectorB):
        result = np.ones((vectorA.size + vectorB.size + 1, 1))
        result[1 : vectorA.size + 1] = vectorA
        result[vectorA.size + 1:] = vectorB
        return result
        

    def construct_architecture(self):
        
        w_in = np.random.normal(
            self.w_in_mu,
            self.w_in_scale,
            (self.n_reservior, self.n_input + 1)
        )

        w = np.random.normal(
            self.w_mu,
            self.w_scale,
            (self.n_reservior, self.n_reservior)
        )

        sr = np.max(np.abs(np.linalg.eigvals(w)))

        w = (w / sr) * self.spectral_radius

        w_out = np.random.normal(
            0,
            1,
            (self.n_output, self.n_readouts)
        )


        return w_in, w, w_out  

    def __call__(self, u):
        u_one = self.fast_insert_one(u)

        self.x = np.tanh(
            np.matmul(self.w_in, u_one) +
            np.matmul(self.w, self.x)
        )

        self.train_state = self.fast_append_and_insert_one(u, self.x) 

        return np.matmul(self.w_out, self.train_state)
        
        


if __name__ == "__main__":
    import time 

    n_passes = 10
    n_in = 10
    n_out = 10
    n_reservior = 100

    us = np.random.random((n_passes, n_in, 1)) - 0.5
    
    start = time.perf_counter()
    doc = Doctor((n_in, n_reservior, n_out), 0.0001, 10, (1, 0, 1, 0), 0.95)
    for i in range(n_passes):
        u = us[i]
        yhat = doc(u)
        train_state_t = doc.train_state.T
        doc.XX = doc.XX + np.matmul(doc.train_state, train_state_t)
        doc.YX = doc.YX + np.matmul(yhat, train_state_t)
    end = time.perf_counter()

    print(f"Execution time {end - start}")

    print(doc.XX[0])
    print(doc.YX[0])
        