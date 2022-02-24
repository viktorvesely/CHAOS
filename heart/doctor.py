import numpy as np
import os
import json

from settings import Params
from loader import dedicate_folder, load_experiment_generator

class Doctor:

    def __init__(
        self, 
        architecture,
        beta, 
        washout,
        w_params,
        spectral_radius,
        d,
        path,
        log_neurons,
        seed=None
    ):
        
        self.n_input, self.n_reservior, self.n_output = architecture
        self.n_readouts = self.n_input + self.n_reservior + 1

        self.beta = beta
        self.washout_period = washout
        self.seed = seed
        self.spectral_radius = spectral_radius
        self.d = d
        self.path = path
        self.indicies = np.random.choice(self.n_reservior, size=log_neurons, replace=False)
        self.log_neurons = log_neurons > 0
        self.debug_neurons = []
        self.w_in_scale, self.w_in_mu, self.w_scale, self.w_mu = w_params

        np.random.seed(self.seed)
        self.w_in, self.w, self.w_out = self.construct_architecture()
        self.x = self.initial_state()
        self.train_state = None

        self.XX = np.zeros((self.n_readouts, self.n_readouts))
        self.YX = np.zeros((self.n_output, self.n_readouts))
        self.XXC = np.zeros((self.n_output, self.n_readouts))
        self.YXC = np.zeros((self.n_output, self.n_readouts))

    def train(self, generator):
        
        for batch in generator:

            self.washout_period

            for i, sample in enumerate(batch):
                
                if self.log_neurons:
                    self.debug_neurons.append(
                        np.squeeze(self.x[self.indicies])
                    )

                if i + self.d >= len(batch):
                    break

                u_now, y = sample
                u_future, _ = batch[i + self.d]

                self(u_now, u_future)

                if i < self.washout_period:
                    continue

                train_state_t = self.train_state.T

                # Kahan summation for XX
                small = np.matmul(self.train_state, train_state_t) - self.XXC
                temp = self.XX + small
                self.XXC = (temp - self.XX) - small
                self.XX = temp

                # Kahan summation for YX
                small = np.matmul(y, train_state_t) - self.YXC
                temp = self.YX + small
                self.YXC = (temp - self.YX) - small
                self.YX = temp
        
        inversed = np.linalg.inv(self.XX + self.beta * np.identity(self.n_readouts))
        self.w_out = np.matmul(self.YX, inversed)
        self.save_model()
        

    def save_model(self):
        p = self.path

        np.save(os.path.join(p, "w_in.npy"), self.w_in)
        np.save(os.path.join(p, "w.npy"), self.w)
        np.save(os.path.join(p, "w_out.npy"), self.w_out)
        
        if self.log_neurons:
            np.save(
                os.path.join(p, "neurons.npy"),
                np.array(self.debug_neurons)
            )
        
    def load_model(self):
        p = self.path

        self.w_in = np.load(os.path.join(p, "w_in.npy"))
        self.w = np.load(os.path.join(p, "w.npy"))
        self.w_out = np.load(os.path.join(p, "w_out.npy"))


    def initial_state(self):
        return np.zeros((self.n_reservior, 1))

    def fast_append(self, vectorA, vectorB):
        result = np.zeros((vectorA.size + vectorB.size, 1))
        result[:vectorA.size] = vectorA
        result[vectorA.size:] = vectorB
        return result

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

    def __call__(self, u_now, u_future):

        u = self.fast_append_and_insert_one(u_now, u_future)

        self.x = np.tanh(
            np.matmul(self.w_in, u) +
            np.matmul(self.w, self.x)
        )

        self.train_state = self.fast_append(u, self.x) 

        return np.matmul(self.w_out, self.train_state)
    

def get_architecture(pars):
    heart_path = os.path.join(os.getcwd(), "hearts", pars.get("dataset"))
    heart_pars = Params(os.path.join(heart_path, "params.json"))

    state_size = len(heart_pars.get("detectors"))
    n_input = 2 * state_size

    n_reservior = pars.get("n_reservior")
    
    n_output = len(heart_pars.get("injectors"))

    return [n_input, n_reservior, n_output]
    
    
    
def boot_doctor_train(name, doc_pars):
    
    name, path = dedicate_folder(name, os.path.join(os.getcwd(), "doctors"))
    
    with open(os.path.join(path, "doctor_params.json"), "w") as f:
        json.dump(doc_pars.params(), f)

    architecture = get_architecture(doc_pars)

    doctor = Doctor(
        architecture,
        doc_pars.get('beta'),
        doc_pars.get('washout'),
        [
            doc_pars.get('w_in_scale'),
            doc_pars.get('w_in_mu'),
            doc_pars.get('w_scale'),
            doc_pars.get('w_mu')
        ],
        doc_pars.get('spectral_radius'),
        doc_pars.get('d'),
        path,
        doc_pars.get('log_neurons')
    )

    doctor.train(
        load_experiment_generator(
            doc_pars.get('dataset'), 
            os.path.join(os.getcwd(), 'hearts')
        )
    )



if __name__ == "__main__":
    
    boot_doctor_train("test", Params("./doctor_params.json"))


    

# CODE FOR showcasing

# import time 

#     n_passes = 10
#     n_in = 10
#     n_out = 10
#     n_reservior = 100

#     us = np.random.random((n_passes, n_in, 1)) - 0.5
    
#     start = time.perf_counter()
#     doc = Doctor((n_in, n_reservior, n_out), 0.0001, 10, (1, 0, 1, 0), 0.95)
#     for i in range(n_passes):
#         u = us[i]
#         yhat = doc(u)
#         train_state_t = doc.train_state.T
#         doc.XX = doc.XX + np.matmul(doc.train_state, train_state_t)
#         doc.YX = doc.YX + np.matmul(yhat, train_state_t)
#     end = time.perf_counter()

#     print(f"Execution time {end - start}")

#     print(doc.XX[0])
#     print(doc.YX[0])
        