import numpy as np
import os

class Nurse:


    def pick_random_non_zero_value_indices(self, avail, n, axis=None):
        """
        DO NOT USE
        """
        hack = np.min(avail) - 1
        indices = np.array(np.where(avail > hack))
        shuffle_indices = np.arange(indices.shape[1])
        np.random.shuffle(shuffle_indices)

        indices[0,:] = indices[0, shuffle_indices]
        indices[1,:] = indices[0, shuffle_indices]
        
        chosen = []

        for pos in indices.T:
            i, j = pos
            if avail[i, j] != 0 and len(chosen) < n:
                if axis is None:
                    chosen.append(pos)
                else:
                    chosen.append(pos[axis])

        if len(chosen) < n:
            raise ValueError(f"Not enough non-zero elements in the avail matrix. Got: {len(chosen)} Needed: {n}")

        return np.array(chosen)
        
        

    def __init__(self, doctor, u_size, x_size, y_size):
        self.doctor = doctor
        self.doc_pars = doctor.pars
        self.heart_pars = doctor.heart_pars

        log_stuff = 10

        # n_local = self.heart_pars.get("gridx") * self.heart_pars.get("gridy")
        # n_other = self.doc_pars.get("local_n_other")

        # self.other_i = np.random.choice(
        #     np.arange(n_local),
        #     size=log_neurons,
        #     replace=False
        # )
        # self.local_i = np.random.choice(
        #     np.arange(n_local, n_local + n_other),
        #     size=log_neurons,
        #     replace=False
        # )

        # self.other_neurons = []
        # self.local_neurons = []


        self.n_i = np.random.choice(
            x_size,
            size=min(x_size, log_stuff),
            replace=False
        )

        self.u_i = np.random.choice(
            u_size,
            size=min(u_size, log_stuff),
            replace=False
        )
        
        self.y_i = np.random.choice(
            y_size,
            size=min(y_size, log_stuff),
            replace=False
        )
        
        self.neurons = []
        self.u = []
        self.y = []


        self.neurons_test = []
        self.u_test = []
        self.yhat_test = []
        self.y_test = []


    def on_test_tick(self, u_now, u_future, yhat, y):

        u = np.ones(u_now.size + u_future.size + 1)
        u[:u_now.size] = np.squeeze(u_now)
        u[u_now.size:-1] = np.squeeze(u_future)

        self.neurons_test.append(
            np.squeeze(self.doctor.x[self.n_i])
        )

        self.u_test.append(
            u[self.u_i]
        )


        self.y_test.append(
            np.squeeze(y[self.y_i])
        )

        self.yhat_test.append(
            np.squeeze(yhat[self.y_i])
        )
        

    def on_training_tick(self, u_now, u_future, y):

        # self.local_neurons.append(
        #     np.squeeze(self.doctor.x[self.local_i])
        # )
        # self.other_neurons.append(
        #     np.squeeze(self.doctor.x[self.other_i])
        # )

        u = np.ones(u_now.size + u_future.size + 1)
        u[:u_now.size] = np.squeeze(u_now)
        u[u_now.size:-1] = np.squeeze(u_future)

        self.neurons.append(
            np.squeeze(self.doctor.x[self.n_i])
        )

        self.u.append(
            u[self.u_i]
        )

        self.y.append(
            np.squeeze(y[self.y_i])
        )

    def on_save(self, core, path):
        
        # l_n = np.array(self.local_neurons)
        # o_n = np.array(self.other_neurons)
        # np.save(os.path.join(path, f"neurons_other_{core}.npy"), o_n)
        # np.save(os.path.join(path, f"neurons_local_{core}.npy"), l_n)
        # np.save(os.path.join(path, f"neurons_local_i_{core}.npy"), self.local_i)
        # np.save(os.path.join(path, f"neurons_other_i_{core}.npy"), self.other_i)

        u = np.array(self.u)
        y = np.array(self.y)
        neurons = np.array(self.neurons)

        np.save(os.path.join(path, f"u_{core}.npy"), u)
        np.save(os.path.join(path, f"neurons_{core}.npy"), neurons)
        np.save(os.path.join(path, f"y_{core}.npy"), y)

    def on_test_finish(self, core, path):
        u_test = np.array(self.u_test)
        y_test = np.array(self.y_test)
        yhat_test = np.array(self.yhat_test)
        neurons_test = np.array(self.neurons_test)

        np.save(os.path.join(path, f"u_test_{core}.npy"), u_test)
        np.save(os.path.join(path, f"neurons_test_{core}.npy"), neurons_test)
        np.save(os.path.join(path, f"y_test_{core}.npy"), y_test)
        np.save(os.path.join(path, f"yhat_test_{core}.npy"), yhat_test)