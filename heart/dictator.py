import os
import numpy as np

class Dictator:


    def __init__(self, doctor_pars, heart_pars):
        self.doctor_pars = doctor_pars
        self.heart_pars = heart_pars

        self.d = doctor_pars.get("d")
        self.begin = doctor_pars.get("reference_loop_start")
        self.end = doctor_pars.get("reference_loop_end")
        self.range = self.end - self.begin
        
        state_path = os.path.join(
            os.getcwd(),
            'hearts',
            doctor_pars.get("reference_signal"),
            'data',
            'states_0_0.npy'
        )
        self.states = np.load(state_path)

    
    # TODO produce a reference sequence for saving

    def u_ref(self, n):
        i = n + self.d
        
        i = self.begin + (i % self.range)

        return self.states[i]

    def u_ref_sequence(self):
        steps = np.arange(
            self.doctor_pars.get("test_time") * self.heart_pars.get("sampling_frequency"),
            dtype=int
        )

        reference_signal = np.array([self.u_ref(n) for n in steps])

        return reference_signal

        
    def save_ref_sequence(self, path):
        np.save(
            path,
            self.u_ref_sequence()
        )


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    from settings import Params

    pars = Params("./doctor_params.json")
    dictator = Dictator(pars)

    seconds = 40
    sampling_frequency = 20
    ns = np.arange(seconds * sampling_frequency)
    reference = np.array([dictator.u_ref(n) for n in ns])
    s = reference.T

    plt.plot(ns, s[0])
    plt.plot(ns, s[1])
    plt.plot(ns, s[2])
    plt.show()
        