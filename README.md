# The Logical Options Framework

Note: This repository contains code for the discrete delivery domain. A separate repository contains code for the continuous reacher domain.

 ## Installation
 
 `conda create --name <env> --file requirements.txt`
 
 ## Running Satisfaction and Composability Experiments
 
 `python tests/satisfaction_delivery.py`
 
 `python tests/composability_delivery.py`
 
 The data generated from these experiments is saved in `dataset/{satisfaction|composability}/{task_name}/{test_num}.npz`

 ## Plot Experiments
 
 `python tests/load_and_plot_results_satisfaction.py`
 
 `python tests/load_and_plot_results_composability.py`

 Plots are saved in `dataset/satisfaction` and `dataset/composability`

## Visualize Algorithm

To make animated gifs of any of the algorithms run the associated test file in `tests/test_qlearning_{alg_name}.py`