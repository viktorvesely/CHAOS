import os
from os.path import isfile, join
import numpy as np

from settings import Params

def get_newest_name(name, path_to_dir="../hearts/"):
    children = os.listdir(path_to_dir)
        
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
        raise ValueError(f"Heart with name {name} does not exist")

    return new_name

def get_state_size(path_to_heart):
    path = join(path_to_heart, 'data', 'states_0_0.npy')
    states = np.load(path)
    return states[0].size
    
def get_cores_and_batch(path):
    experiment_path = path
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

    return cores


def load_experiment_core(name, path, core, n_files):
    ptd = join(path, name, 'data')
    
    actions_batch = np.load(join(ptd, f"actions_{core}_{0}.npy"))
    states_batch = np.load(join(ptd, f"states_{core}_{0}.npy"))
    for n_file in range(1, n_files + 1):
        actions = np.load(join(ptd, f"actions_{core}_{n_file}.npy"))
        actions_batch = np.concatenate((actions_batch, actions), axis=0)
        states = np.load(join(ptd, f"states_{core}_{n_file}.npy"))
        states_batch = np.concatenate((states_batch, states), axis=0)

    return (states_batch, actions_batch)

def load_experiment_generator(name, path):

    ptd = join(path, name, 'data')
    cores = get_cores_and_batch(ptd)


    for core, n_files in cores.items():
        actions_batch = np.load(join(ptd, f"actions_{core}_{0}.npy"))
        states_batch = np.load(join(ptd, f"states_{core}_{0}.npy"))
        for n_file in range(1, n_files + 1):
            actions = np.load(join(ptd, f"actions_{core}_{n_file}.npy"))
            actions_batch = np.concatenate((actions_batch, actions), axis=0)
            states = np.load(join(ptd, f"states_{core}_{n_file}.npy"))
            states_batch = np.concatenate((states_batch, states), axis=0)
        
        yield (states_batch, actions_batch)

def load_experiment(name, path, newest=False):

    if newest:
        name = get_newest_name(name)
    
    all_actions = []
    all_states = []
    
    par = Params(os.path.join(path, name, "params.json"))
    t_start = par.get("t_start")
    t_end = par.get("t_end")
        
    for states_batch, actions_batch in load_experiment_generator(name, path):
        all_states.append(states_batch)
        all_actions.append(actions_batch)

    return all_states, all_actions, t_start, t_end, par, name


def dedicate_folder(name, experiments_path):

    original_name = name  
    
    suffix = 1
    while os.path.isdir(os.path.join(experiments_path, name)):
        name = f"{original_name}-{suffix}"
        suffix += 1 

    path = os.path.join(experiments_path, name)
    os.mkdir(path)

    return name, path