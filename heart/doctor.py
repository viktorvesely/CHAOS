import numpy as np
import os

import recorder
from settings import Params
from loader import dedicate_folder, load_experiment_generator
from dictator import Dictator


class Doctor:

    def __init__(
        self, 
        name,
        architecture,
        beta, 
        washout,
        w_params,
        spectral_radius,
        d,
        path,
        log_neurons,
        u_bounds,
        y_bounds,
        pars,
        heart_pars,
        sampling_frequency,
        seed=None
    ):

        self.name = name        
        self.n_input, self.n_reservior, self.n_output = architecture
        self.n_readouts = self.n_input + self.n_reservior + 1

        self.beta = beta
        self.washout_period = washout
        self.seed = seed
        self.u_bounds = u_bounds
        self.y_bounds = y_bounds
        self.spectral_radius = spectral_radius
        self.d = d
        self.path = path
        self.pars = pars
        self.heart_pars = heart_pars
        self.fs = sampling_frequency
        self.exploit_period = 1000 / sampling_frequency
        self.dictator = Dictator(self.pars, self.heart_pars)
        self.test_time = pars.get('test_time')
        self.indicies = np.random.choice(self.n_reservior, size=log_neurons, replace=False)
        self.log_neurons = log_neurons > 0
        self.debug_neurons = []
        self.w_in_scale, self.w_in_mu, self.w_scale, self.w_mu = w_params

        np.random.seed(self.seed)
        self.w_in, self.w, self.w_out = self.construct_architecture()
        self.x = self.initial_state()
        self.train_state = None

        self.XX = np.zeros((self.n_readouts, self.n_readouts))
        self.XXC = np.zeros((self.n_readouts, self.n_readouts))

        self.YX = np.zeros((self.n_output, self.n_readouts))
        self.YXC = np.zeros((self.n_output, self.n_readouts))


    def normalize_batch(self, states, actions):
        s_min, s_max = self.u_bounds
        a_min, a_max = self.y_bounds

        s_shape = states.shape
        states = np.reshape(states, (s_shape[0], s_shape[1], 1))
        a_shape = actions.shape
        actions = np.reshape(actions, (a_shape[0], a_shape[1], 1))

        n_samples = s_shape[0]

        states = (states - s_min) / (s_max - s_min)
        actions = (actions - a_min) / (a_max - a_min)

        return states, actions, n_samples

    def exploit(self, u_now, t):
        
        n = round(t / self.exploit_period)
        u_desired = self.dictator.u_ref(n)

        u_desired = np.reshape(u_desired, (u_desired.size, 1))
        u_now = np.reshape(u_now, (u_now.size, 1))

        yhat = self(u_now, u_desired)

        # TODO un-normalization + clipping

        return yhat.flatten()

    def test(self):

        heart_name = self.pars.get("dataset")
        hearts_path = os.path.join(os.getcwd(), "hearts")
        doctor_heart_name = f"TEST_{self.name}_{heart_name}"
        doctor_heart_name, path = dedicate_folder(doctor_heart_name, hearts_path)
        test_time = self.pars.get("test_time")
    
        print(f"Generating test '{doctor_heart_name}' for {test_time} simulation seconds")

        # Override the duration of the simulation
        original_t_start = self.heart_pars.get("t_start")
        original_t_end = self.heart_pars.get("t_end")
        t_end = self.pars.get("test_time") * self.fs
        t_end *= 1000 # Convert to ms
        t_start = 0
        self.heart_pars.override("t_start", t_start)
        self.heart_pars.override("t_end", t_end)        

        # Start the model
        heart = recorder.Recorder(
            doctor_heart_name,
            0,
            path,
            self.heart_pars
        ).setup_interactive_mode(self.exploit)

        heart.record()
        self.dictator.save_ref_sequence(os.path.join(path, 'data', 'ref.npy'))

        # Restore heart params just in case : )
        self.heart_pars.override("t_start", original_t_start)
        self.heart_pars.override("t_end", original_t_end)
    
        
        
        
    def train(self, generator):
        
        print("Training network")

        for states, actions in generator:
                            
            states, actions, n_samples = self.normalize_batch(states, actions)
            

            for i in range(n_samples):


                if i + self.d >= n_samples:
                    break

                if self.log_neurons:
                    self.debug_neurons.append(
                        np.squeeze(self.x[self.indicies])
                    )

                u_now = states[i]
                y = actions[i]

                if i < 2:
                    print(u_now)
                    print(y)

                
                u_future = states[i + self.d]

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
        self.test()
        

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
    

def get_architecture(pars, heart_pars):

    state_size = len(heart_pars.get("detectors"))
    n_input = 2 * state_size

    n_reservior = pars.get("n_reservior")
    
    n_output = len(heart_pars.get("injectors"))

    return [n_input, n_reservior, n_output]
    
    
    
def boot_doctor_train(name, doc_pars):
    
    name, path = dedicate_folder(name, os.path.join(os.getcwd(), "doctors"))
    
    # Copy settings
    doc_pars.save(os.path.join(path, "doctor_params.json"))

    heart_path = os.path.join(os.getcwd(), "hearts", doc_pars.get("dataset"))
    heart_pars = Params(os.path.join(heart_path, "params.json"))
    architecture = get_architecture(doc_pars, heart_pars)

    doctor = Doctor(
        name,
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
        doc_pars.get('log_neurons'),
        [   
            heart_pars.get('min_Vm'),
            heart_pars.get('max_Vm')
        ],
        [
            heart_pars.get('min_action'),
            heart_pars.get('max_action')
        ],
        doc_pars,
        heart_pars,
        heart_pars.get('sampling_frequency')
    )

    doctor.train(
        load_experiment_generator(
            doc_pars.get('dataset'), 
            os.path.join(os.getcwd(), 'hearts')
        )
    )



if __name__ == "__main__":
    
    boot_doctor_train("bigboi", Params("./doctor_params.json"))


    

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
        