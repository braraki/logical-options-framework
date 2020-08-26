import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# def load_dataset(exp_name, exp_num):
#     directory = Path(__file__).parent.parent / 'dataset' / exp_name
#     # if directory doesn't exist, create it
#     Path(directory).mkdir(parents=True, exist_ok=True)
#     file_name = 'results_' + str(exp_num) + '.npz'
#     path_name = directory / file_name
    
#     data = np.load(path_name, allow_pickle=True)

#     return data

def load_dataset(exp_name, method_name, task_name, exp_num):
    directory = Path(__file__).parent.parent / 'dataset' / exp_name / method_name / task_name
    # if directory doesn't exist, create it
    Path(directory).mkdir(parents=True, exist_ok=True)
    file_name = str(exp_num) + '.npz'
    path_name = directory / file_name
    
    data = np.load(path_name, allow_pickle=True)

    return data['arr_0'][()]

task_reward_bounds = [(-100, -26), (-100, -62), (-100, -21), (-100, -19)]
# task_reward_bounds = [(-100, -38), (-100, -62), (-100, -21), (-100, -25)]
task_names = ['complex', 'sequential', 'OR', 'IF']
method_names = ['lof', 'flat', 'fsa', 'rm', 'greedy']
method_plot_names = ['Logical Options', 'Options', 'Options+FSA', 'Reward Machines', 'Greedy']
method_colors = ['b', 'y', 'r', 'c', 'g']

num_exp = 10

def get_plot_data_for_task(task_num, task_name):

    method_max_rewards = []
    method_min_rewards = []
    method_ave_rewards = []
    method_steps = []

    for method_name in method_names:
        first_data = load_dataset('satisfaction', method_name, task_name, 0)
        num_data = len(first_data['reward'])
        steps = first_data['steps']
        method_steps.append(steps)
        max_rewards = [-np.inf]*num_data
        min_rewards = [np.inf]*num_data
        ave_rewards = [0]*num_data
        for i in range(num_exp):
            results = load_dataset('satisfaction', method_name, task_name, i)
            # for each experiment, average reward over the tasks
            bounds = task_reward_bounds[task_num]
            for k, reward in enumerate(results['reward']):
                reward = (reward - bounds[0])/(bounds[1]-bounds[0])
                if reward > max_rewards[k]:
                    max_rewards[k] = reward
                if reward < min_rewards[k]:
                    min_rewards[k] = reward

                ave_rewards[k] += reward / num_exp

        method_max_rewards.append(max_rewards)
        method_min_rewards.append(min_rewards)
        method_ave_rewards.append(ave_rewards)

    return method_ave_rewards, method_min_rewards, method_max_rewards, method_steps


def get_plot_data_over_tasks():

    method_max_rewards = []
    method_min_rewards = []
    method_ave_rewards = []
    method_steps = []

    for method_name in method_names:
        first_data = load_dataset('satisfaction', method_name, task_names[0], 0)
        num_data = len(first_data['reward'])
        steps = first_data['steps']
        method_steps.append(steps)
        max_rewards = [-np.inf]*num_data
        min_rewards = [np.inf]*num_data
        ave_rewards = [0]*num_data
        for i in range(num_exp):
            # for each experiment, average reward over the tasks
            task_ave_rewards = [0]*num_data
            for j, task_name in enumerate(task_names):
                results = load_dataset('satisfaction', method_name, task_name, i)
                bounds = task_reward_bounds[j]
                for k, reward in enumerate(results['reward']):
                    reward = (reward - bounds[0])/(bounds[1]-bounds[0])
                    task_ave_rewards[k] += reward/len(task_names)
                
            for k, reward in enumerate(task_ave_rewards):
                if reward > max_rewards[k]:
                    max_rewards[k] = reward
                if reward < min_rewards[k]:
                    min_rewards[k] = reward

                ave_rewards[k] += reward / num_exp

        method_max_rewards.append(max_rewards)
        method_min_rewards.append(min_rewards)
        method_ave_rewards.append(ave_rewards)

    return method_ave_rewards, method_min_rewards, method_max_rewards, method_steps

def plot_data_over_tasks():
    method_ave_rewards, method_min_rewards, \
        method_max_rewards, method_steps = get_plot_data_over_tasks()


    for i, (method_name, method_color, ave_reward, min_reward, max_reward, steps) in enumerate(zip(
            method_plot_names, method_colors, method_ave_rewards, method_min_rewards, method_max_rewards, method_steps)):
        plt.plot(steps, ave_reward, color=method_color, label=method_name)    
        plt.fill_between(steps, min_reward, max_reward, color=method_color, alpha=0.2)

    plt.ylim(0, 1)
    plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.2))
    plt.tight_layout()
    plt.title('Reward Averaged over Tasks')
    plt.xlim((0, 150000))

    directory = Path(__file__).parent.parent / 'dataset' / 'satisfaction'
    # if directory doesn't exist, create it
    Path(directory).mkdir(parents=True, exist_ok=True)
    file_name = 'results_averaged_over_tasks.png'
    path_name = directory / file_name

    plt.savefig(path_name)

def plot_data_per_task():
    for i, task_name in enumerate(task_names):
        fig = plt.figure()
        method_ave_rewards, method_min_rewards, \
        method_max_rewards, method_steps = get_plot_data_for_task(i, task_name)

        for i, (method_name, method_color, ave_reward, min_reward, max_reward, steps) in enumerate(zip(
                method_plot_names, method_colors, method_ave_rewards, method_min_rewards, method_max_rewards, method_steps)):
            plt.plot(steps, ave_reward, color=method_color, label=method_name)    
            plt.fill_between(steps, min_reward, max_reward, color=method_color, alpha=0.2)

        plt.ylim(0, 1)
        plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.2))
        plt.tight_layout()
        plt.title("Reward for {} task".format(task_name))
        plt.xlim((0, 150000))

        directory = Path(__file__).parent.parent / 'dataset' / 'satisfaction'
        # if directory doesn't exist, create it
        Path(directory).mkdir(parents=True, exist_ok=True)
        file_name = 'results_{}.png'.format(task_name)
        path_name = directory / file_name

        plt.savefig(path_name)

if __name__ == '__main__':
    plot_data_over_tasks()
    plot_data_per_task()