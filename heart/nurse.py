import numpy as np
import os

class Nurse:

    def __init__(self, doctor):
        self.doctor = doctor
        self.doc_pars = doctor.pars
        self.heart_pars = doctor.heart_pars

        log_neurons = 10

        n_local = self.heart_pars.get("gridx") * self.heart_pars.get("gridy")
        n_other = self.doc_pars.get("local_n_other")

        self.other_i = np.random.choice(
            np.arange(n_local),
            size=log_neurons,
            replace=False
        )
        self.local_i = np.random.choice(
            np.arange(n_local, n_local + n_other),
            size=log_neurons,
            replace=False
        )
        self.other_neurons = []
        self.local_neurons = []
        

    def on_training_tick(self, u_now, u_future, y):

        self.local_neurons.append(
            np.squeeze(self.doctor.x[self.local_i])
        )
        self.other_neurons.append(
            np.squeeze(self.doctor.x[self.other_i])
        )

    def on_save(self, core, path):
        
        l_n = np.array(self.local_neurons)
        o_n = np.array(self.other_neurons)

        np.save(os.path.join(path, f"neurons_other_{core}.npy"), o_n)
        np.save(os.path.join(path, f"neurons_local_{core}.npy"), l_n)
        np.save(os.path.join(path, f"neurons_local_i_{core}.npy"), self.local_i)
        np.save(os.path.join(path, f"neurons_other_i_{core}.npy"), self.other_i)