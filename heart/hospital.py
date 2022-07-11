from multiprocessing.pool import Pool, ThreadPool
import numpy as np
import os
import json

import loader
from loader import dedicate_folder, load_experiment_generator
from settings import Params
from doctor import Doctor
from simple import architecture as simple_architecture
from simple import test as test_model
from recorder import Recorder

import cProfile
import pstats

def get_parser():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--name', type=str, required=True)
    parser.add_argument('-hc', '--hypercores', type=int, default=0)
    parser.add_argument('-tc', '--traincores', type=int, default=1)
    parser.add_argument('-p', '--parts', type=int, default=-1)
    parser.add_argument('-l', '--limit', type=int, default=-1)
    parser.add_argument('-nh', '--nonheart', action='store_true')
    parser.add_argument('-ttl', '--testload', action='store_true')
    parser.add_argument('-t', '--test', action='store_true')

    return parser  

def get_heart_path(doc_pars):
    return os.path.join(os.getcwd(), "hearts", doc_pars.get("dataset"))

def boot_doctor_train_non_heart(
    name,
    path,
    doc_pars,
    min_u,
    max_u,
    min_y,
    max_y,
    architecture,
    core=0,
    save=True
    ):

    if save:
        doc_pars.save(os.path.join(path, f"doctor_params_{core}.json"))

    doctor = Doctor(
        name,
        doc_pars.get('beta'),
        doc_pars.get('washout'),
        doc_pars.get('d'),
        path,
        [   
            min_u, max_u
        ],
        [
            min_y, max_y
        ],
        doc_pars,
        None,
        1,
        backup_architecture=architecture
    )
    
    return doctor

def boot_doctor_train(name, path, doc_pars, core=0, save=True):
    
    # Copy settings
    if save:
        doc_pars.save(os.path.join(path, f"doctor_params_{core}.json"))

    heart_pars = Params(os.path.join(get_heart_path(doc_pars), "params.json"))

    doctor = Doctor(
        name,
        doc_pars.get('beta'),
        doc_pars.get('washout'),
        doc_pars.get('d'),
        path,
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

def calc_NRMSE(ys, yhats):
    # Axis 0: time
    # Axis 1: features

    delta = np.mean((yhats - ys) * (yhats - ys), axis=1)
    variances = np.var(ys, axis=1)
    NMSE = delta / variances
    NRMSE = np.sqrt(NMSE)
    NRMSE = np.mean(NRMSE)

    return NRMSE
    

def test(doctor):
    doc_pars = doctor.pars

    ys, yhats = doctor.test_train_data(
        load_experiment_generator(
            doc_pars.get('dataset'), 
            os.path.join(os.getcwd(), 'hearts')
        ),
        cores=1
    )

    ys = ys.T
    yhats = yhats.T

    return calc_NRMSE(ys, yhats)

def load_model(name, core=0, non_heart_args=None):

    path = os.path.join(os.getcwd(), 'doctors', name)

    doctor_pars = Params(os.path.join(path, f'doctor_params_{core}.json'))

    if non_heart_args is not None:

        args = non_heart_args
        doctor = boot_doctor_train_non_heart(
            name,
            path,
            doctor_pars,
            core=core,
            min_u=args["min_u"],
            max_u=args["max_u"],
            min_y=args["min_y"],
            max_y=args["max_y"],
            architecture=args["architecture"],
            save=False
        )
    else:
        doctor = boot_doctor_train(name, path, doctor_pars, core=core, save=False)

    doctor.load_model(core=core)

    return doctor

def train_single_thread(
    name,
    path,
    doctor_pars,
    verbal=True,
    save=True,
    parts=-1,
    core=0,
    non_heart_args=None
    ):

    if non_heart_args is not None:

        args = non_heart_args
        doctor = boot_doctor_train_non_heart(
            name,
            path,
            doctor_pars,
            core=core,
            min_u=args["min_u"],
            max_u=args["max_u"],
            min_y=args["min_y"],
            max_y=args["max_y"],
            architecture=args["architecture"]
        )
    else:

        doctor = boot_doctor_train(name, path, doctor_pars, core=core)

    doctor.train(
        load_experiment_generator(
            doctor_pars.get('dataset'),
            os.path.join(os.getcwd(), 'hearts')
        ),
        save=save,
        verbal=verbal,
        parts=parts
    )

    return test(doctor), doctor


def train_single_thread_pool_wrapper(args):
    name, path, doctor_pars, parts, core, non_heart_args = args
    _, doctor = train_single_thread(
        name,
        path,
        doctor_pars,
        verbal=False,
        save=False,
        parts=parts,
        core=core,
        non_heart_args=non_heart_args
    )

    doctor.save_model(core=core)

    print(f"Testing {core}")
    NRMSE = real_test(doctor, verbal=False)

    hyper_params = doctor_pars.get('__hyper_params')
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

    if "__runs" in grid:
        return grid["__runs"]

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


def hyper_optimization_single_thread_training(
    name,
    path,
    hyper_cores,
    original_pars,
    parts=-1,
    non_heart_args=None
    ):

    import copy
    import time

    with open("./grid.json", "r") as f:
        grid = json.load(f)

    runs = generate_grid(grid)

    pool_args = []

    for i, run in enumerate(runs):
        doctor_pars_dict = copy.deepcopy(original_pars.params())
        update_params(run, doctor_pars_dict)
        doctor_pars = Params().from_dict(doctor_pars_dict)
        doctor_pars.params()["__hyper_params"] = run
        pool_args.append([name, path, doctor_pars, parts, i, non_heart_args])
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

    keys = runs[0].keys()
    export = "nrmse"

    for key in keys:
        value = ranked[0][1][key]
        if isinstance(value, list):
            value = np.array(value).flatten()
            
            key_labels = f"{key}_labels"
            labels_found = False
            if key_labels in original_pars.params():
                labels = original_pars.get(labels)
                if value.size == len(labels):
                    labels_found = True
                    for label in labels:
                        export += f",{label}"
                else:
                    print(f"Labels found in {key_labels} with size {len(labels)} are not matching the size of the sample with size {value.size}")

            if not labels_found:
                for i in range(value.size):
                    export += f",{key}_{i}"
        else:
            export += f",{key}"
    export += "\n"

    for run in ranked:
        NRMSE, hp = run
        print(f"{NRMSE} : {hp}")
        export += str(NRMSE)
        for key in keys:
            value = hp[key]
            if isinstance(value, list):
                value = np.array(value).flatten()
            
                for i in range(value.size):
                    export += f",{value[i]}"
            else:
                export += f",{value}"

        export += "\n"    
    

    with open(os.path.join(path, 'results.csv'), "w", encoding='utf-8') as f:
        f.write(export)
    
def real_test(doctor, verbal=True):
    trajectory, actions, reference = test_model(doctor, t_end = 1000, verbal=verbal)
    
    NRMSE = calc_NRMSE(reference[doctor.washout_period:, :].T, trajectory[doctor.washout_period:, :].T)
    
    print(trajectory.shape)

    with open("./oscillator/discussion.js", 'w') as fi:
        jString = "var reference_traj = {}; var real_traj = {};".format(
            json.dumps(reference[:10_000,0:2].tolist()),
            json.dumps(trajectory[:10_000,0:2].tolist())
        )
        fi.write(jString)

    if verbal:
        print(f"Test NRMSE: {NRMSE}")
    return NRMSE


if __name__ == '__main__':
    import time

    parser = get_parser()
    args = parser.parse_args()

    non_heart_args = None
    if args.nonheart:
        non_heart_args = {
            "min_u": -1, "max_u": 1,
            "min_y": -1, "max_y": 1,
            "architecture": simple_architecture
        }

    if args.testload:
        print(f"[{args.name}] Loading and real test")
        doctor = load_model(args.name, core=0, non_heart_args=non_heart_args)
        doctor.test()
        exit()

    name, path = dedicate_folder(
        args.name,
        os.path.join(os.getcwd(), 'doctors')
    )

    doctor_params = Params("./doctor_params.json")

    doctor = None
    if args.hypercores > 0:

        if args.traincores > 1:
            print(f"""[{name}] Hyperoptimization using {args.hypercores} core(s) for hyper-optimization and {args.traincores} cores for training""")
            print("Not implemented right now .... exiting")
            exit()
        else:
            print(f"""[{name}] Hyperoptimization runs using {args.hypercores} core(s) for hyper-optimization and single-threaded training""")
            hyper_optimization_single_thread_training(
                args.name,
                path,
                args.hypercores,
                doctor_params,
                parts=args.parts,
                non_heart_args=non_heart_args
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
        NRMSE, doctor = train_single_thread(
            name,
            path,
            doctor_params,
            parts=args.parts,
            non_heart_args=non_heart_args
        )
        end = time.perf_counter()
        print(f"NRMSE: {NRMSE}")
        print(f"Singlethreaded training took {end - start}")
    
    if args.test:
        print(f"[{name}] Real test time!")
        real_test(doctor)



    

    
