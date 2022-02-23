import os
from os.path import isfile, join
import numpy as np

from settings import Params

def load_experiment(name, newest=False):
    core = 0
    batch = 0

    if newest:
        children = os.listdir("../hearts/")
        
        new_name = None
        suffix = -1
        for child in children:
            child_name = child.split("-")

            if len(child_name) == 1:
                if new_name is None and child_name[0] == name:
                    new_name = child_name[0]
                    suffix = 0
                continue

            child_suffix = int(child_name[1])
            if child_name[0] == name and child_suffix > suffix:
                new_name = child
                suffix = child_suffix
        
        if new_name is None:
            raise ValueError(f"Heart wiht name {name} does not exist")
        
        name = new_name
        
    experiment_path = f"../hearts/{name}/data/"
    data_files = [f for f in os.listdir(experiment_path) if isfile(join(experiment_path, f))]
    
    cores = {}

    for f in data_files:
        params = f.replace(".npy", "").split("_")
        n_core = int(params[1])
        number = int(params[2])
        
        if n_core not in cores:
            cores[n_core] = -1
        
        if cores[n_core] < number:
            cores[n_core] = number
    
    all_actions = []
    all_states = []
    
    for core, n_files in cores.items():
        actions_batch = np.load(f"../hearts/{name}/data/actions_{core}_{0}.npy")
        states_batch = np.load(f"../hearts/{name}/data/states_{core}_{0}.npy")
        for n_file in range(1, n_files + 1):
            actions = np.load(f"../hearts/{name}/data/actions_{core}_{n_file}.npy")
            actions_batch = np.concatenate((actions_batch, actions), axis=0)
            states = np.load(f"../hearts/{name}/data/states_{core}_{n_file}.npy")
            states_batch = np.concatenate((states_batch, states), axis=0)
        
        all_actions.append(actions_batch)
        all_states.append(states_batch)
            
    
    par = Params(f"../hearts/{name}/params.json")
    t_start = par.get("t_start")
    t_end = par.get("t_end")

    return all_states, all_actions, t_start, t_end, par, name