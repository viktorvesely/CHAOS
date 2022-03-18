from multiprocessing.pool import Pool, ThreadPool
import numpy as np
import os
import json

import loader
from loader import dedicate_folder, load_experiment_generator
from settings import Params
from doctor import Doctor

def get_parser():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--name', type=str, required=True)
    parser.add_argument('-hc', '--hypercores', type=int, default=0)
    parser.add_argument('-tc', '--traincores', type=int, default=1)
    parser.add_argument('-p', '--parts', type=int, default=-1)

    return parser  

def get_heart_path(doc_pars):
    return os.path.join(os.getcwd(), "hearts", doc_pars.get("dataset"))
    
def boot_doctor_train(name, path, doc_pars):
    
    # Copy settings
    doc_pars.save(os.path.join(path, "doctor_params.json"))

    heart_pars = Params(os.path.join(get_heart_path(doc_pars), "params.json"))

    doctor = Doctor(
        name,
        doc_pars.get('beta'),
        doc_pars.get('washout'),
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

    return doctor


def test(doctor, save=True):
    doc_pars = doctor.pars

    ys, yhats = doctor.test_train_data(
        load_experiment_generator(
            doc_pars.get('dataset'), 
            os.path.join(os.getcwd(), 'hearts')
        ),
        cores=1
    )

    delta = np.mean((yhats - ys) * (yhats - ys), axis=0)
    variances = np.var(ys, axis=0)
    NMSE = delta / variances
    NRMSE = np.sqrt(NMSE)
    NRMSE = np.mean(NRMSE)

    if save:
        np.save("./trash/ys.npy", ys)
        np.save("./trash/yhats.npy", yhats)
    
    return NRMSE

def train_single_thread(name, path, doctor_pars, verbal=True, save=True, parts=-1):

    doctor = boot_doctor_train(name, path, doctor_pars)

    doctor.train(
        load_experiment_generator(
            doctor_pars.get('dataset'),
            os.path.join(os.getcwd(), 'hearts')
        ),
        save=save,
        verbal=verbal,
        parts=parts
    )

    return test(doctor)

def train_single_thread_pool_wrapper(args):
    name, path, doctor_pars, parts = args
    NRMSE = train_single_thread(
        name,
        path,
        doctor_pars,
        verbal=False,
        save=False,
        parts=parts
    )

    hyper_params = doctor_pars.get('__hyper_params')
    print(f"{NRMSE} : {hyper_params}")
    return (
        NRMSE,
        hyper_params
    )

def generator(heart_name, core, n_files):
    yield loader.load_experiment_core(
        heart_name,
        os.path.join(os.getcwd(), "hearts"),
        core,
        n_files
    )

def boot_and_train(options):
    name, path, doctor_pars, core, n_files = options

    doctor = boot_doctor_train(name, path, doctor_pars)

    doctor.train(generator(
        doctor.pars.get("dataset"),
        core,
        n_files
    ), save=False, verbal=False)

    print(f"Core {core} finished!")
    return doctor
    

def train_multi_threaded(name, path, doctor_pars, n_cores):
    
    heart_data = os.path.join(get_heart_path(doctor_pars), 'data')
    cores = loader.get_cores_and_batch(heart_data)
    
    pool_args = [ [ name, path, doctor_pars, core, n_files ] for core, n_files in cores.items() ]

    with Pool(n_cores) as p:
        doctors = p.map(boot_and_train, pool_args)
    
    doctor = doctors[0]

    for other in doctors[1:]:
        doctor.XX += other.XX
        doctor.YX += other.YX

    doctor.calc_w_out()
    doctor.save_model()
    return test(doctor)


def set_item(run, name, item):
    import random

    if isinstance(item, list):
        run[name] = random.choice(item)
    elif isinstance(item, dict):

        run[name] = {}
        for k, v in item.items():
            run[name][k] = random.choice(v)
    else:
        raise TypeError(f"Incorrect value type got: {type(item)}, expected: list or dict")

def generate_run(grid):
    
    run = {}

    for name, value in grid.items():
        
        args = name.split(":")

        if len(args) > 1:
            continue
        
        set_item(run, name, value)

    for name, value in grid.items():
        
        args = name.split(":")

        if len(args) == 1:
            continue

        name = args[2]
        condition = args[1].split("=")
        variable = condition[0]
        desired = condition[1]
        
        if run[variable] != desired:
            continue

        set_item(run, name, value)

    return run

    
def generate_runs(n):

    with open("./grid.json", "r") as f:
        grid = json.load(f)
    
    runs = []
    

    while len(runs) < n:
        run = generate_run(grid)
        
        if run in runs:
            continue
        
        runs.append(run)
    
    return runs

def iterate_param(remaining_grid_keys, grid):
    
    if len(remaining_grid_keys) == 0:
        yield {}
        return

    current = remaining_grid_keys[:1][0]
    remaining = remaining_grid_keys[1:]
    values = grid[current]

    for value in values:
        for shallow_run in iterate_param(remaining, grid):
            run = {}
            run[current] = value
            run.update(shallow_run)
            yield run

def generate_grid(grid):

    keys = list(grid.keys())
    runs = [ run for run in iterate_param(keys, grid) ]

    return runs

def update_params(run, pars):

    for key, value in run.items():
        
        if key not in pars:
            raise ValueError(f"Run contains key {key} which is not a valid parameter")    

        if isinstance(value, dict):
            update_params(value, pars[key])
            continue
        
        pars[key] = value


def hyper_optimization_single_thread_training(name, path, hyper_cores, original_pars, parts=-1):
    import copy
    import time

    with open("./grid.json", "r") as f:
        grid = json.load(f)

    runs = generate_grid(grid)

    pool_args = []

    for run in runs:
        doctor_pars_dict = copy.deepcopy(original_pars.params())
        update_params(run, doctor_pars_dict)
        doctor_pars = Params().from_dict(doctor_pars_dict)
        doctor_pars.params()["__hyper_params"] = run
        pool_args.append([name, path, doctor_pars, parts])
    n_runs = len(runs)

    print(f"{n_runs} run(s) generated!")

    start = time.perf_counter()
    with Pool(hyper_cores) as pool:
        results = pool.map(train_single_thread_pool_wrapper, pool_args)
    end = time.perf_counter()

    duration = end - start
    print(f"Hyperoptimization finished in {duration}")
    print(f"Time per run: {duration / n_runs}")

    ranked = sorted(results, key=lambda result: result[0])

    keys = grid.keys()
    export = "nrmse"

    for key in keys:
        value = ranked[0][1][key]
        if isinstance(value, list) and len(value) == 2:
            export += f",{key}_mu,{key}_scale"
        else:
            export += f",{key}"
    export += "\n"

    for run in ranked:
        NRMSE, hp = run
        export += str(NRMSE)
        for key in keys:
            value = hp[key]
            if isinstance(value, list) and len(value) == 2:
                export += f",{value[0]},{value[1]}"
            else:
                export += f",{value}"

        export += "\n"    
    

    with open(os.path.join(path, 'results.csv'), "w", encoding='utf-8') as f:
        f.write(export)
    
    

if __name__ == '__main__':
    import time

    parser = get_parser()
    args = parser.parse_args()

    name, path = dedicate_folder(
        args.name,
        os.path.join(os.getcwd(), 'doctors')
    )

    doctor_params = Params("./doctor_params.json")


    if args.hypercores > 0:

        if args.traincores > 1:
            print(f"""[{name}] Hyperoptimization using {args.hypercores} core(s) for hyper-optimization and {args.traincores} cores for training""")
            print("Not implemented right now .... exiting")
        else:
            print(f"""[{name}] Hyperoptimization runs using {args.hypercores} core(s) for hyper-optimization and single-threaded training""")
            hyper_optimization_single_thread_training(
                args.name,
                path,
                args.hypercores,
                doctor_params,
                parts=args.parts
            )
  
    elif args.traincores > 1:
        print(f"[{name}] Multithreaded training using {args.traincores} cores")
        start = time.perf_counter()
        NRMSE = train_multi_threaded(name, path, doctor_params, args.traincores)
        end = time.perf_counter()
        print(f"NRMSE: {NRMSE}")
        print(f"Multithreaded training took {end - start}")

    else:
        print(f"[{name}] Singlethreaded training")
        start = time.perf_counter()
        NRMSE = train_single_thread(name, path, doctor_params, parts=args.parts)
        end = time.perf_counter()
        print(f"NRMSE: {NRMSE}")
        print(f"Singlethreaded training took {end - start}")
    

    

    
