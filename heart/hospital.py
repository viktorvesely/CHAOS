from multiprocessing.pool import Pool, ThreadPool
import numpy as np
import os

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

    return parser


def get_architecture(pars, heart_pars):

    state_size = len(heart_pars.get("detectors"))
    n_input = 2 * state_size

    n_reservior = pars.get("n_reservior")
    
    n_output = len(heart_pars.get("injectors"))

    return [n_input, n_reservior, n_output]
    

def get_heart_path(doc_pars):
    return os.path.join(os.getcwd(), "hearts", doc_pars.get("dataset"))
    
def boot_doctor_train(name, path, doc_pars):
    
    # Copy settings
    doc_pars.save(os.path.join(path, "doctor_params.json"))

    heart_pars = Params(os.path.join(get_heart_path(doc_pars), "params.json"))
    architecture = get_architecture(doc_pars, heart_pars)

    doctor = Doctor(
        name,
        architecture,
        doc_pars.get('beta'),
        doc_pars.get('washout'),
        [
            doc_pars.get('w_in_scale'),
            doc_pars.get('w_in_mu'),
            doc_pars.get('w_min'),
            doc_pars.get('w_max')
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

def train_single_thread(name, path, doctor_pars):
    
    doctor = boot_doctor_train(name, path, doctor_pars)

    doctor.train(
        load_experiment_generator(
            doctor_pars.get('dataset'),
            os.path.join(os.getcwd(), 'hearts')
        )
    )

    return test(doctor)


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


def hyper_optimization(name, hyper_cores, train_cores):
    pass


if __name__ == '__main__':


    parser = get_parser()
    args = parser.parse_args()

    name, path = dedicate_folder(
        args.name,
        os.path.join(os.getcwd(), 'doctors')
    )

    if args.hypercores > 0:
        print(f"[{name}] Hyperoptimization using {args.hypercores * args.traincores} cores")
        hyper_optimization(args.name, args.hypercores, args.traincores)
    
    if args.traincores > 1:
        print(f"[{name}] Multithreaded training using {args.traincores} cores")
        doctor_params = Params("./doctor_params.json")
        NRMSE = train_multi_threaded(name, path, doctor_params, args.traincores)
        print(f"NRMSE: {NRMSE}")
    else:
        print(f"[{name}] Singlethreaded training")
        doctor_params = Params("./doctor_params.json")
        NRMSE = train_single_thread(name, path, doctor_params)
        print(f"NRMSE: {NRMSE}")
    

    

    
