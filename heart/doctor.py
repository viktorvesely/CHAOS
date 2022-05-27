from re import S
import numpy as np
import os
from scipy import sparse as sp
import time
import numba
from sklearn.decomposition import PCA

import recorder
from loader import dedicate_folder, get_state_size, load_experiment_generator
from dictator import Dictator
from reservoir import get_architecture
from nurse import Nurse

@numba.njit(parallel=True)
def fadd(a1, a2, shape):
    result = np.zeros(shape)
    for i in numba.prange(shape[0]):
        for j in numba.prange(shape[1]):
            result[i, j] = a1[i, j] + a2[i, j]
    return result

PARALLEL_ADD = False


class Doctor:

    def __init__(
        self,
        name,
        beta, 
        washout,
        d,
        path,
        u_bounds,
        y_bounds,
        pars,
        heart_pars,
        sampling_frequency
    ):       

        self.name = name 
        self.beta = beta
        self.washout_period = washout
        self.u_bounds = u_bounds
        self.y_bounds = y_bounds
        self.d = d
        self.path = path
        self.pars = pars
        self.heart_pars = heart_pars
        self.fs = sampling_frequency
        self.exploit_period = 1000 / sampling_frequency
        self.dictator = Dictator(self.pars, self.heart_pars)
        self.test_time = pars.get('test_time')
        self.kahan = pars.get('kahan')
        self.core = None
        self.pca = None
        self.__is_pca = False
        self.w_in_pca_penalty = None

        self.w_in, self.w, self.w_out, self.leaky_mask = get_architecture(pars, heart_pars)
        self.n_input = self.w_in.shape[1]
        self.n_reservior = self.w.shape[0]
        self.n_output, self.n_readouts = self.w_out.shape

        self.x = self.initial_state()
        self.train_state = None
        self.u_size = get_state_size(os.path.join(os.getcwd(), 'hearts', pars.get("dataset")))
        self.u_now = np.zeros((self.u_size, 1))
        self.V_now = np.zeros((self.u_size, 1))
        self.non_pca_states = None
        self.u_future = np.zeros((self.u_size, 1))

        self.extenders = []
        if pars.get("x^2"):
            self.extend_readouts(self.x_squared)
        if pars.get("state_sub"):
            self.extend_readouts(self.sub_input)
        pca_dim = pars.get("pca_dim")
        if pca_dim > 0:
            self.init_pca(pca_dim)

        self.XX = np.zeros((self.n_readouts, self.n_readouts))
        self.XXC = np.zeros((self.n_readouts, self.n_readouts))
        
        self.YX = np.zeros((self.n_output, self.n_readouts))
        self.YXC = np.zeros((self.n_output, self.n_readouts))

        self.nurse = Nurse(self, 
            self.w_in.shape[1],
            self.w.shape[0],
            self.w_out.shape[0]
        )
    

    def init_pca(self, pca_dim):
        self.__is_pca = True
        self.pca = PCA(n_components=pca_dim)
        
        generator = load_experiment_generator(
            self.pars.get("dataset"),
            os.path.join(os.getcwd(), 'hearts')
        )

        states, _ = next(generator)

        states = self.normalize_states(states)
        self.pca.fit(states)

        #self.extend_readouts(self.heart)
        
        penalty = np.array(self.pca.explained_variance_)
        penalty = penalty / (np.max(penalty) / 2)
        penalty = np.reshape(penalty, (-1, 1))
        self.w_in_pca_penalty = np.ones((pca_dim * 2 + 1, 1))
        self.w_in_pca_penalty[:pca_dim, :] = np.copy(penalty)
        self.w_in_pca_penalty[pca_dim:-1, :] = np.copy(penalty)
        
    def heart(self):
        return self.V_now    

    def extend_readouts(self, blueprint):
        self.extenders.append(blueprint)
        self.n_readouts += blueprint().size
        self.w_out = np.random.random((self.n_output, self.n_readouts)) 

    def x_squared(self):
        return self.x * self.x
    
    def sub_input(self):
        return self.u_future - self.u_now

    def normalize_states(self, states):
        s_min, s_max = self.u_bounds
        states = (states - s_min) / (s_max - s_min)
        states = states * 2 - 1
        return states

    def normalize_actions(self, actions):
        a_min, a_max = self.y_bounds
        actions = (actions - a_min) / (a_max - a_min)
        return actions

    def normalize_batch(self, states, actions):
        
        states = self.normalize_states(states)
        actions = self.normalize_actions(actions)

        a_shape = actions.shape
        actions = np.reshape(actions, (a_shape[0], a_shape[1], 1))
        s_shape = states.shape
        states = np.reshape(states, (s_shape[0], s_shape[1], 1))

        n_samples = s_shape[0]

        if self.__is_pca:
            self.non_pca_states = states
            states = self.pca.transform(np.squeeze(states))
            shape = states.shape
            states = np.reshape(states, (shape[0], shape[1], 1))

        return states, actions, n_samples

    def exploit(self, u_now, t):
        
        n = round(t / self.exploit_period)
        u_desired = self.dictator.u_ref(n)

        # TODO normalization

        u_desired = np.reshape(u_desired, (u_desired.size, 1))
        u_now = np.reshape(u_now, (u_now.size, 1))

        yhat = self(u_now, u_desired)

        # TODO un-normalization + clipping

        return yhat.flatten()

    def test(self):

        self.x = self.initial_state()

        heart_name = self.pars.get("dataset")
        hearts_path = os.path.join(os.getcwd(), "hearts")
        doctor_heart_name = f"TEST_{self.name}_{heart_name}"
        doctor_heart_name, path = dedicate_folder(doctor_heart_name, hearts_path)
        test_time = self.pars.get("test_time")
    
        print(f"Generating test '{doctor_heart_name}' for {test_time} simulation seconds")

        # Override the duration of the simulation
        original_t_start = self.heart_pars.get("t_start")
        original_t_end = self.heart_pars.get("t_end")
        t_end = self.pars.get("test_time")
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
    
        

    def test_train_data(self, generator, cores=1):
        
        print("Testing on train data")

        ys = []
        yhats = []

        for _ in range(cores):

            states, actions = next(generator)

            states, actions, n_samples = self.normalize_batch(states, actions)
            self.x = self.initial_state()

            for i in range(n_samples - self.d):

                u_now = states[i]
                y = actions[i]
                u_future = states[i + self.d]

                if self.__is_pca:
                    yhat = self(u_now, u_future, self.non_pca_states[i])
                else:
                    yhat = self(u_now, u_future)

                self.nurse.on_test_tick(u_now, u_future, yhat, y)

                if i < self.washout_period:
                    continue

                yhats.append(yhat)
                ys.append(y)
        
        ys = np.squeeze(np.array(ys))
        yhats = np.squeeze(np.array(yhats))

        self.nurse.on_test_finish(self.core, self.path)
     
        return ys, yhats
        


    def train(self, generator, save=True, verbal=True, parts=-1):
        
        if verbal:
            print("Training network")
        
        X = []
        Y = []

        core = 1
        for states, actions in generator:
            
            if parts > 0 and (core - 1) >= parts:
                break

            start = time.perf_counter()
            states, actions, n_samples = self.normalize_batch(states, actions)
            self.x = self.initial_state()

            for i in range(n_samples - self.d):
                
                u_now =  states[i]
                y = actions[i]            
                u_future = states[i + self.d]

                self.nurse.on_training_tick(u_now, u_future, y)

                if self.__is_pca:
                    self(u_now, u_future, self.non_pca_states[i])
                else:
                    self(u_now, u_future)

                if i < self.washout_period:
                    continue
                
                X.append(np.squeeze(self.train_state))
                Y.append(np.squeeze(y))
            
            end = time.perf_counter()
            if verbal:
                print(f"Core {core} done in {end - start}")
            core += 1
  
        X_T = np.array(X)
        X = X_T.T
        Y = np.array(Y).T
        
        self.calc_w_out(X, X_T, Y)

        if save:
            self.save_model()
        

    def calc_w_out(self, X, X_T, Y):
        inversed = np.linalg.inv(np.matmul(X, X_T) + self.beta * np.identity(self.n_readouts))
        self.w_out = np.matmul(np.matmul(Y, X_T), inversed)

    def save_model(self, core=0):
        p = self.path
        self.core = core

        if sp.issparse(self.w):
            self.w = self.w.toarray()
            
        np.save(os.path.join(p, f"w_in_{core}.npy"), self.w_in)
        np.save(os.path.join(p, f"w_{core}.npy"), self.w)
        np.save(os.path.join(p, f"w_out_{core}.npy"), self.w_out)
        np.save(os.path.join(p, f"leaky_mask_{core}.npy"), self.leaky_mask)
        if self.__is_pca:
            np.save(os.path.join(p, f"pca.npy"), self.pca.components_)

        self.nurse.on_save(core, p)
        
    def load_model(self, core=0):
        p = self.path

        raise NotImplementedError("Implement loading PCA matrix first")

        self.w_in = np.load(os.path.join(p, f"w_in_{core}.npz"))
        self.w = np.load(os.path.join(p, f"w_{core}.npz"))
        self.w_out = np.load(os.path.join(p, f"w_out_{core}.npy"))
        self.leaky_mask = np.load(os.path.join(p, f"leaky_mask_{core}.npy"))


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


    def __call__(self, u_now, u_future, non_pca_state=None):

        self.u_now = u_now
        self.u_future = u_future
        u = self.fast_append_and_insert_one(u_now, u_future)

        

        self.x = self.x * (1 - self.leaky_mask) + self.leaky_mask * np.tanh(
            self.w_in.dot(u * self.w_in_pca_penalty) +
            self.w.dot(self.x)
        )

        if self.__is_pca:
            self.V_now = non_pca_state

        self.train_state = self.compile_readouts(u)

        return np.matmul(self.w_out, self.train_state)
    


    

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
        