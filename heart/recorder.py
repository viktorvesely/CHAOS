import numpy as np
import os
import shutil
import time
from enum import Enum
from multiprocessing import Pool

from heart import solve
from noise import SinusNoise, RectNoise, WhiteNoise
from loader import dedicate_folder


class Recorder:

    MAX_BUFFEE_SIZE = 10_000
    MINIMAL_DISTURBANCE = 20

    def __init__(self, name, core, path, pars, lineArgs=None):
        self.name = name
        self.core = core
        self.lineArgs = lineArgs
        self.pars = pars
        
        self.period = 1000 / self.pars.get("sampling_frequency")
        self.next_sample = 0
        self.grid = (self.pars.get("gridy"), self.pars.get("gridx"))
        
        self.detectors = tuple(
            np.array(self.pars.get("detectors")).T
        )

        self.injectors = tuple(
            np.array(self.pars.get("injectors")).T
        )

        self.num_actions = len(self.injectors[0])

        self.states = []
        self.actions = []

        self.save_n_batch = 0
        
        self.path = path

        self.noise = None
        self.get_action = self.resolve_action_mode()
        self.last_action = 0
        self.interactive = False

    def setup_interactive_mode(self, get_action_callback):
        self.lineArgs = get_parser_defaults(self.name)
        self.lineArgs.verbal = True
        self.get_action = get_action_callback
        self.interactive = True
        return self

    def resolve_action_mode(self):
        actor = self.pars.get("actor")

        if actor == "":
            return self.get_action_zeros

        if actor == "sinus":
            settings = self.pars.get("noise_sinus_settings")
            self.noise = SinusNoise(
                self.pars.get("min_action"),
                self.pars.get("max_action"),
                self.num_actions,
                settings
            )
            return self.get_action_noise

        if actor == "rect":
            settings = self.pars.get("noise_rect_settings")
            self.noise = RectNoise(
                self.pars.get("min_action"),
                self.pars.get("max_action"),
                self.num_actions,
                settings
            )
            return self.get_action_noise

        if actor == "white":
            settings = self.pars.get("noise_white_settings")
            self.noise = WhiteNoise(
                self.pars.get("min_action"),
                self.pars.get("max_action"),
                self.num_actions,
                settings
            )
            return self.get_action_noise
            

        raise ValueError(f"Invalid actor: Got {actor} check the code for expected")

    def print(self, arg):
        if self.lineArgs.verbal:
            print(arg)

    def get_state(self, V):
        return V.flatten()

    def save(self):

        if self.interactive:
            return

        np.save(
            os.path.join(self.path, 'data', f'states_{self.core}_{self.save_n_batch}.npy'),
            np.array(self.states)
        )

        np.save(
            os.path.join(self.path, 'data', f'actions_{self.core}_{self.save_n_batch}.npy'),
            np.array(self.actions)
        )

        self.states = []
        self.actions = []

        self.save_n_batch += 1

    def onTick(self, V, t):

        if t >= self.next_sample:
            self.next_sample += self.period

            state = self.get_state(V)
            action = self.get_action(state, t)
            self.states.append(state)
            self.actions.append(action)

            if len(self.states) >= Recorder.MAX_BUFFEE_SIZE:
                self.save()
            
            self.last_action = action
            return self.map_action_to_heart(action) 

        return self.map_action_to_heart(self.last_action)
        #return self.map_action_to_heart(self.get_action_zeros(None, t))

    def map_action_to_heart(self, action):
        stimuli_map = np.zeros(self.grid)
        stimuli_map[self.injectors] = action
        return stimuli_map


    def get_action_noise(self, state, t):
        return self.noise(t)

    def get_action_zeros(self, state, t):
        return np.zeros(self.num_actions)

    def get_disturbance(self):
        if not self.lineArgs.disrupt:
            return 0
        
        dist = np.random.random(self.grid) * Recorder.MINIMAL_DISTURBANCE

        return dist

    def record(self):

        disturbance = self.get_disturbance()

        perf_start = time.perf_counter()
        solve(
            self.pars,
            videoOut=self.lineArgs.record,
            verbal=self.lineArgs.verbal,
            onTick=self.onTick,
            s0_disturbance=disturbance
        )
        perf_end = time.perf_counter()

        print(f"Solve time core {self.core}: {perf_end - perf_start}")

        if len(self.states) > 0:
            self.save()


def create_heart(name, pars):

    name, path = dedicate_folder(name, os.path.join(os.getcwd(), 'hearts'))

    mask_path = pars.get("resistivity_path")
    mask_name = mask_path.split('/')[-1]

    # Copy resistivity mask
    shutil.copyfile(
        mask_path,
        os.path.join(path, mask_name)
    )

    # Copy settings
    pars.override(
        "resistivity_path",
        f"./hearts/{name}/{mask_name}"
    )
    pars.save(os.path.join(path, 'params.json'))

    os.mkdir(os.path.join(path, 'data'))

    return name, path


def setup_recorder(options):
    name, core, args, path, pars = options

    recorder = Recorder(
        name,
        core,
        path,
        pars,
        lineArgs=args
    )

    recorder.record()
    

def get_parser_defaults(name):
    class AttributeDict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
    

    parser = get_parser()
    args = parser.parse_args(["--name", "placeholder"])

    all_defaults = {}
    for key in vars(args):
        all_defaults[key] = parser.get_default(key)
    
    all_defaults["name"] = name

    return AttributeDict(all_defaults)

def get_parser():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--name', type=str, required=True)
    parser.add_argument('-v', '--verbal', action="store_true", default=False)
    parser.add_argument('-c', '--cores', type=int, default=1)
    parser.add_argument('-r', '--record', action="store_true", default=False)
    parser.add_argument('-disr', '--disrupt', action="store_true", default=False)

    return parser


if __name__ == '__main__':

    from settings import Params

    parser = get_parser()
    args = parser.parse_args()

    if args.record and args.cores > 1:
        raise ValueError(f"It would be unsafe to record the heart with multiprocessing ; ). Change the number of cores from {args.cores} to 0")

    if args.verbal and args.cores > 1:
        print("Verbal flag won't work properly with multiple cores!")

    pars = Params("./params.json")
    name, path = create_heart(args.name, pars)


    if args.cores > 1:
        duration = (pars.get("t_end") - pars.get("t_start")) / 1000
        print(f"Simulating {args.cores} heart(s) simultaneously for {duration} seconds")
        pool_args = [ [name, core, args, path, pars] for core in range(args.cores)]
        
        with Pool(args.cores) as p:
            p.map(setup_recorder, pool_args)
    else:
        recorder = Recorder(
            name,
            0,
            path,
            pars,
            lineArgs=args
        )

        recorder.record()