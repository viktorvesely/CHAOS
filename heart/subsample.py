    
if __name__ == '__main__':
    import argparse
    from loader import get_cores_and_batch
    import os
    from os.path import join
    import numpy as np
    from settings import Params


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
    

    origin_pars = Params(join(path, "params.json"))
    gridx = origin_pars.get("gridx")
    gridy = origin_pars.get("gridy")
    if gridx % i_ratio != 0:
        raise ValueError(f"Cannot subsample with ratio 1:{i_ratio} since the subsampling factor {i_ratio} is not a divisor of the number of heart columns (which is {gridx})")

    print(f"Subsampling [{args.name}] to [{new_name}] with ratio {i_ratio} : 1")
    os.mkdir(new_path)
    os.mkdir(join(new_path, 'data'))
    cores = get_cores_and_batch(join(path, 'data'))

    original_size = None
    new_size = None

    for core, n_files in cores.items():
        for n_file in range(n_files):
            action_name =f"actions_{core}_{n_file}.npy"
            state_name =f"states_{core}_{n_file}.npy"

            actions = np.load(join(path, 'data', action_name))
            states = np.load(join(path, 'data', state_name))
            original_size = states.shape[1]
            
            states = states[:,::i_ratio]
            np.save(join(new_path, 'data', action_name), actions)
            np.save(join(new_path, 'data', state_name), states)

    new_size = states.shape[1]
    params = Params(join(path, "params.json"))
    
    new_gridx = int(gridx / i_ratio)
    new_gridy = int(gridy)

    if new_gridx * new_gridy == new_size:
        print(f"Original size: {original_size}")
        print(f"New size: {new_size}")
    else:
        raise ValueError(f"Non-matching new heart shape ({new_gridy}, {new_gridx}) with the state size: {new_size}. Delete the subsampled folder, fix your code, and try again.")
        
    params.create("__ss_shape", (new_gridy, new_gridx))
    params.save(join(new_path, "params.json"))
            


