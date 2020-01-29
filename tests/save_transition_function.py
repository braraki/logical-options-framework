import time, os
from simulator.rendering import Viewer
from simulator.balldrop import BallDropSim
import scipy.sparse as sparse
from pathlib import Path

plot = False

# for an environment and a list of transition functions,
# save the transition functions in a folder
# /transitions/env_name/Ti.npz
def save_transitions(env_name, T):
    directory = Path(__file__).parent.parent / 'transitions' / env_name
    # if directory doesn't exist, create it
    Path(directory).mkdir(parents=True, exist_ok=True)
    # delete contents of the file so that they can be replaced
    # (just in case the number of actions changed)
    for file in directory.iterdir():
        Path.unlink(file)
    for i, t in enumerate(T):
        file_name = 'T' + str(i) + '.npz'
        path_name = directory / file_name
        sparse.save_npz(path_name, t)


def make_transitions():
    sim = BallDropSim()
    sim.reset()
    T = sim.env.make_transition_function(plot)
    save_transitions(sim.env.name, T)

if __name__ == '__main__':
    make_transitions()