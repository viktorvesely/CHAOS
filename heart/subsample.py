    
if __name__ == '__main__':
    import argparse
    from loader import get_cores_and_batch
    import os
    from os.path import join
    import numpy as np
    from settings import Params
    import shutil


    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('-s', '--subsample', type=int, required=True)
    args = parser.parse_args()

    i_ratio = args.subsample
    path = join(os.getcwd(), 'hearts', args.name)
    new_name = f"{args.name}_SS_{i_ratio}"
    new_path = join(os.getcwd(), 'hearts', f"{args.name}_SS_{i_ratio}")

    if os.path.isdir(new_path):
        print(f"Heart with name {new_name} already exists")
        exit(0)
    
    print(f"Subsampling [{args.name}] to [{new_name}] with ration {i_ratio} : 1")
    os.mkdir(new_path)
    os.mkdir(join(new_path, 'data'))
    cores = get_cores_and_batch(join(path, 'data'))


    for core, n_files in cores.items():
        for n_file in range(n_files):
            action_name =f"actions_{core}_{n_file}.npy"
            state_name =f"states_{core}_{n_file}.npy"

            actions = np.load(join(path, 'data', action_name))
            states = np.load(join(path, 'data', state_name))
            states = states[:,::i_ratio]
            np.save(join(new_path, 'data', action_name), actions)
            np.save(join(new_path, 'data', state_name), states)

    
    params = Params(join(path, "params.json"))
    params.create("__ss_size", states.shape[1])
    params.save(join(new_path, "params.json"))
            


