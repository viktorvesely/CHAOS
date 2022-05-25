from multiprocessing.pool import Pool, ThreadPool
import numpy as np
import os
import json
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from regex import F

import loader
from loader import dedicate_folder, load_experiment_generator
from settings import Params
from doctor import Doctor
from pendulum import dsdt, arctan2ToPendulum


import cProfile
import pstats

def get_parser():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--name', type=str, required=True)
    parser.add_argument('-hc', '--hypercores', type=int, default=0)
    parser.add_argument('-tc', '--traincores', type=int, default=1)
    parser.add_argument('-p', '--parts', type=int, default=-1)
    parser.add_argument('-t', '--test', action="store_true", default=False)
    parser.add_argument('-tat', '--testtrain', action="store_true", default=False)
    parser.add_argument('-nhr', '--nhyperruns', type=int, default=1)

    return parser  


def diplay_hyper_optimization(path):
    
    df = pd.read_csv(os.path.join(path, "results.csv"))
    n_ivs = len(df.columns) - 2
    ivs = list(df.columns.values)
    del ivs[ivs.index("nrmse")]
    del ivs[ivs.index("var")]
    dv ="nrmse"
    var = "var"

    if n_ivs == 2:
        fig = plt.figure(figsize=(7, 7))
        ax = plt.axes(projection='3d')
        
        ax.scatter3D(df[ivs[0]], df[ivs[1]], df[dv], c=df[dv], cmap="Reds")
        return
    
    # Determin the number of columns and rows for the figure
    n_cols = 3 if n_ivs > 3 else n_ivs
    n_rows = int(np.ceil(n_ivs / n_cols))
    # How many figures need to be deleted
    extra = n_rows * n_cols - n_ivs
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    # Case of only one independent variable
    if n_cols == 1:
        iv = ivs[0]
        ax.scatter(df[iv], df[dv], marker="x")
        ax.errorbar(df[iv], df[dv], yerr=df[var], fmt="o")
        ax.set_xlabel(iv)
        ax.set_ylabel(dv)
    # Case when there is only one row
    elif n_rows == 1:

        # Delete extra axes
        for i in range(1, extra + 1):
            fig.delaxes(ax[-i])
    
        for index, iv in enumerate(ivs):
            ax[index].scatter(df[iv], df[dv], marker="x")
            ax[index].errorbar(df[iv], df[dv], yerr=df[var], fmt="o")
            ax[index].set_xlabel(iv)
            ax[index].set_ylabel(dv)
    # Case of multiple rows
    else:
        # Delete extra axes
        for i in range(1, extra + 1):
            fig.delaxes(ax[-1, -i])

        for index, iv in enumerate(ivs):
            # Calculate the indicies of the figure
            i = int(np.floor(index / n_cols))
            j = index % n_cols

            ax[i, j].scatter(df[iv], df[dv], marker="x")
            ax[i, j].errorbar(df[iv], df[dv], yerr=df[var], fmt="o")
            ax[i, j].set_xlabel(iv)
            ax[i, j].set_ylabel(dv)
    
    plt.show()
    

def get_heart_path(doc_pars):
    return os.path.join(os.getcwd(), "hearts", doc_pars.get("dataset"))
    
def boot_doctor_train(name, path, doc_pars, core=0):
    
    # Copy settings
    doc_pars.save(os.path.join(path, f"doctor_params_{core}.json"))

    heart_pars = None

    doctor = Doctor(
        name,
        doc_pars.get('beta'),
        doc_pars.get('washout'),
        doc_pars.get('d'),
        path,
        None,
        None,
        doc_pars,
        heart_pars,
        None
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

    ys = ys.T
    yhats.T
    ys = np.reshape(ys, (1, -1))
    yhats = np.reshape(yhats, (1, -1))


    if save:
        np.save("./trash/ys.npy", ys)
        np.save("./trash/yhats.npy", yhats)
    
    return calc_NRMSE(yhats, ys)

def train_single_thread(
    name,
    path,
    doctor_pars,
    verbal=True,
    save=True, 
    parts=-1, 
    core=0, 
    perform_test=True
):

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

    if perform_test:
        return test(doctor), doctor
    else:
        return None, doctor

def train_single_thread_pool_wrapper(args):
    name, path, doctor_pars, parts, core = args
    NRMSE, doctor = train_single_thread(
        name,
        path,
        doctor_pars,
        verbal=False,
        save=False,
        parts=parts,
        core=core
    )

    doctor.save_model(core=core)

    hyper_params = doctor_pars.get('__hyper_params')
    return (
        NRMSE,
        hyper_params
    )

def train_single_thread_test_pool_wrapper(args):
    name, path, doctor_pars, parts, core, N = args

    scores = []

    for i in range(N):

        _, doctor = train_single_thread(
            name,
            path,
            doctor_pars,
            verbal=False,
            save=False,
            parts=parts,
            core=core,
            perform_test=False
        )

        NRMSE = real_test(name, path, doctor_pars, verbal=False, doctor=doctor)
        scores.append(NRMSE)

        print(f"[{name}] Core {core}: {i + 1} / {N}")

    scores = np.array(scores)
    doctor.save_model(core=core)

    hyper_params = doctor_pars.get('__hyper_params')
    return (
        np.mean(scores),
        hyper_params,
        np.var(scores)
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
    nRuns=1
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
        pool_args.append([name, path, doctor_pars, parts, i, nRuns])
    n_runs = len(runs)

    print(f"{n_runs} run(s) generated!")

    start = time.perf_counter()
    with Pool(hyper_cores) as pool:
        results = pool.map(train_single_thread_test_pool_wrapper, pool_args)
    end = time.perf_counter()

    duration = end - start
    print(f"Hyperoptimization finished in {duration}")
    print(f"Time per run: {duration / n_runs}")

    
    ranked = sorted(results, key=lambda result: np.mean(result[0]))

    keys = runs[0].keys()
    export = "nrmse,var"

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
        NRMSE, hp, var = run
        print(f"{NRMSE} : {hp}")
        export +=  f"{NRMSE},{var}"
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

    diplay_hyper_optimization(path)
    
    
def real_test(name, path, pars, verbal=False, doctor=None):
    global trajectory, actions

    from pendulum import get_targets

    timesteps = 10_000
    washout = 50
    rescaling = 28.696154954529547
    # t = np.arange(timesteps)

    targets = get_targets(timesteps)
    # targets = np.load("./hearts/pendulum/data/states_0_0.npy")
    # oractions = np.load("./hearts/pendulum/data/actions_0_0.npy")
    targets = np.reshape(targets, (targets.shape[0], targets.shape[1], 1))

    # state = np.array([
    #     [targets[0, 0, 0]],
    #     [np.arctan2(targets[0, 2, 0], targets[0, 1, 0])]
    # ])

    phi0 = np.arctan2(targets[0, 2], targets[0, 1])[0]
    if phi0 < 0:
        phi0 = 2 * np.pi - phi0

    state = np.array([
        [0],
        [phi0]
    ])

    

    actions = [ ]
    trajectory = [ ]
    
    d = pars.get("d")

    if doctor is None:
        doctor = Doctor(
            name,
            pars.get('beta'),
            pars.get('washout'),
            d,
            path,
            None,
            None,
            pars,
            None,
            None
        )

        doctor.load_model()
    
    doctor.x = doctor.initial_state()

    for i in range(timesteps):
        phi = state[1, 0]
        u_now = np.array([
            [state[0, 0]],
            [np.cos(phi)],
            [np.sin(phi)]
        ])

        if i + d >= targets.shape[0]:
            break

        u_ref = targets[i + d]

        action = doctor(u_now, u_ref)
        
        torque = action[0, 0] * rescaling

        if i < washout:
            torque = 0
        
        torque = np.clip(torque, -500.0, 500.0)
        #torque = oractions[i, 0]

        trajectory.append(state)
        state = dsdt(state, torque)
        actions.append(torque)
    
    trajectory = np.squeeze(np.array(trajectory))
    trajectory = trajectory.T

    targets = np.squeeze(targets)
    targets = targets[:-d,:].T

    trajectoryCart = np.array([
        np.cos(trajectory[1]),
        np.sin(trajectory[1])
    ])

    targetCart = np.array([
        targets[1, :], targets[2, :]
    ])

    targetRad = np.arctan2(targets[2, :], targets[1, :])
    targetRad = arctan2ToPendulum(targetRad)

    actions = np.array(actions).T
      
    if verbal:
        print(f"RMSE state diff: {calc_RMSE(trajectoryCart, targetCart)}")

        begin = 0
        window = 10_000
        end = begin + window

        fig, ax = plt.subplots(2, 2, figsize=(18, 6), dpi=90)
        ax[0, 0].plot(trajectoryCart[0][begin:end], label="x real", linewidth=1)
        ax[0, 0].plot(trajectoryCart[1][begin:end], label="y real", linewidth=1)
        ax[0, 0].plot(targets[1][begin:end], label="x target", linewidth=1)
        ax[0, 0].plot(targets[2][begin:end], label="y target", linewidth=1)
        ax[0, 0].legend()

        ax[0, 1].plot(trajectory[1][begin:end], label="phi real", linewidth=1)
        ax[0, 1].plot(targetRad[begin:end], label="phi target", linewidth=1)
        ax[0, 1].legend()

        ax[1, 0].plot(np.squeeze(actions), label="actions")
        ax[1, 0].legend()
        #plt.plot(actions[begin:end], label="actions")
        plt.show()

    return calc_RMSE(trajectoryCart, targetCart)


def calc_NRMSE(yhat, y):
    """
    Axis 0: Features
    Axis 1: Time
    """
    MSE = np.mean((yhat - y) * (yhat - y), axis=1)
    variances = np.var(y, axis=1)
    NMSE = MSE / variances
    NRMSE = np.sqrt(NMSE)
    NRMSE = np.mean(NRMSE)

    return NRMSE

def calc_RMSE(yhat, y):
    """
    Axis 0: Features
    Axis 1: Time
    """
    MSE = np.mean((yhat - y) * (yhat - y), axis=1)
    RMSE = np.sqrt(MSE)
    RMSE = np.mean(RMSE)

    return RMSE

def test_load(name, path, pars):
    d = pars.get("d")
    doctor = Doctor(
        name,
        pars.get('beta'),
        pars.get('washout'),
        d,
        path,
        None,
        None,
        pars,
        None,
        None
    )

    doctor.load_model()

    NRMSE = test(doctor, save=False)

    print(f"NRMSE: {NRMSE}")


if __name__ == '__main__':
    import time

    parser = get_parser()
    args = parser.parse_args()

    if args.test:
        name = args.name
        print(f"[{name}] performing the real boi test")
        path = os.path.join(os.getcwd(), 'doctors', name)
        doctor_params = Params(os.path.join(path, 'doctor_params_0.json'))
        real_test(name, path, doctor_params, verbal=True)
        exit()

    if args.testtrain:
        name = args.name
        print(f"[{name}] performing load test")
        path = os.path.join(os.getcwd(), 'doctors', name)
        doctor_params = Params(os.path.join(path, 'doctor_params_0.json'))
        test_load(name, path, doctor_params)
        exit()

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
                nRuns=args.nhyperruns,
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
        NRMSE, _ = train_single_thread(name, path, doctor_params, parts=args.parts)
        end = time.perf_counter()
        print(f"NRMSE: {NRMSE}")
        print(f"Singlethreaded training took {end - start}")
        print(f"[{name}] Performing the real boi test")
        path = os.path.join(os.getcwd(), 'doctors', name)
        real_test(name, path, doctor_params, verbal=True)
        exit()
    

    

    
