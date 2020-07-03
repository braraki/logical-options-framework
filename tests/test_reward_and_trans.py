import time, os
from simulator.rendering import Viewer
from simulator.balldrop import BallDropSim
from simulator.lineworld import LineWorldSim
from simulator.gridworld import GridWorldSim
import scipy.sparse as sparse
from pathlib import Path
import numpy as np

plot = False

# for an environment and a list of transition functions,
# save the transition functions in a folder
# /transitions/env_name/Ti.npz
def save_transitions(env_name, T):
    directory = Path(__file__).parent.parent / 'storage' / env_name / 'transitions'
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

def save_reward_function(env_name, R):
    directory = Path(__file__).parent.parent / 'storage' / env_name / 'reward'
    # if directory doesn't exist, create it
    Path(directory).mkdir(parents=True, exist_ok=True)
    # delete contents of the file so that they can be replaced
    # (just in case the number of actions changed)
    for file in directory.iterdir():
        Path.unlink(file)
    file_name = 'R.npz'
    path_name = directory / file_name
    np.savez(path_name, R=R)


def make_transitions_balldrop():
    sim = BallDropSim()
    sim.reset()

    R = sim.env.make_reward_function()
    save_reward_function(sim.env.name, R)
    
    # T = sim.env.make_transition_function(plot)
    # save_transitions(sim.env.name, T)

def make_transitions_lineworld():
    sim = LineWorldSim()
    sim.reset()

    R = sim.env.make_reward_function()

    # save_reward_function(sim.env.name, R)
    
    T = sim.env.make_transition_function(plot)
    # save_transitions(sim.env.name, T)

    return R, T

def make_transitions_gridworld():
    sim = GridWorldSim()
    sim.reset()

    R = sim.env.make_reward_function()

    # save_reward_function(sim.env.name, R)
    
    T = sim.env.make_transition_function(plot)
    # save_transitions(sim.env.name, T)

    return R, T

if __name__ == '__main__':
    R, T = make_transitions_gridworld()
    print(R)
    print(T[1].toarray())