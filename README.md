# The Logical Options Framework

Note: This repository contains code for the discrete delivery domain. A separate repository contains code for the continuous reacher domain.

 ## Installation
 
 `conda env create -f environment.yml`
 
 will create a conda environment called 'lof'.

 Add `/path/to/lof/` to your `PYTHONPATH` variable in your `~/.bashrc` file and `source ~/.bashrc`

 ## Running Satisfaction and Composability Experiments
 
 For satisfaction experiments, run: `python tests/satisfaction_delivery.py`
 
 For composability epxeriments, run: `python tests/composability_delivery.py`
 
 The data generated from these experiments is saved in `dataset/{satisfaction|composability}/{task_name}/{test_num}.npz`

 ## Plot Experiments
 
 To plot satisfaction results: `python tests/load_and_plot_results_satisfaction.py`
 
 To plot composability results: `python tests/load_and_plot_results_composability.py`

 Plots are saved in `dataset/satisfaction` and `dataset/composability`

## Visualize Algorithm

To make animated gifs of any of the algorithms run the associated test file in `tests/test_qlearning_{alg_name}.py`

## Overview of the Code

The code for the LOF algorithm is stored in the `simulator` folder. The code can be divided into three main parts:

### Environment

The `Env` class and its children define environment MDPs with states, actions, and transitions. These classes are in `environment.py`.

### Simulator

The `Sim` class and its children define "simulators" for environments, where you can initialize an environment, input actions, step the environment forward, and get the reward and state of the environment. The `Sim` class is defined in `simulator.py`. `GridWorldSim` is defined in `gridworld.py` and defines a discete 2D gridworld. `DeliverySim` is defined in `delivery.py` and defines a specific type of `GridWorldSim` that mimics a package delivery setting.

### Objects and Propositions

In this code base, "objects" are things that are included in the environment that have their own state and dynamics, such as the agent, a falling block, or a goal. Objects are defined in `object.py`.
Propositions are logical true/false events in an environment, and they are used to determine transitions in the logical automation. They are defined in `proposition.py`.

### Options and Meta-policies

The core of the LOF algorithm is defined in `options.py`. This file contains all of the classes critical for LOF, including definitions of subgoals, task specs, and safety specs; classes for low-level options; and classes for high-level meta-policies. The main LOF algorithm is found in `options.QLearningMetaPolicy`. This class learns low-level options to achieve subgoals using Q-learning and find a meta-policy over the options using Logical Value Iteration.