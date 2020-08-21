import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_dataset(exp_name, exp_num):
    directory = Path(__file__).parent.parent / 'dataset' / exp_name
    # if directory doesn't exist, create it
    Path(directory).mkdir(parents=True, exist_ok=True)
    file_name = 'results_' + str(exp_num) + '.npz'
    path_name = directory / file_name
    
    data = np.load(path_name, allow_pickle=True)

    return data

task_reward_bounds = [(-100, -38), (-100, -62), (-100, -7), (-100, -25)]

num_exp = 10

for i in range(num_exp):

    data = load_dataset('satisfaction', exp_num=i)

    lof_results = data['lof'][()]
    flat_results = data['flat'][()]
    fsa_results = data['fsa'][()]
    rm_results = data['rm'][()]

    all_results = [lof_results, flat_results, fsa_results, rm_results]

    if i == 0:
        max_lof_rewards = [-np.inf]*len(lof_results[0]['reward'])
        max_flat_rewards = [-np.inf]*len(flat_results[0]['reward'])
        max_fsa_rewards = [-np.inf]*len(fsa_results[0]['reward'])
        max_rm_rewards = [-np.inf]*len(rm_results[0]['reward'])
        max_rewards = [max_lof_rewards, max_flat_rewards, max_fsa_rewards, max_rm_rewards]
        min_lof_rewards = [np.inf]*len(lof_results[0]['reward'])
        min_flat_rewards = [np.inf]*len(flat_results[0]['reward'])
        min_fsa_rewards = [np.inf]*len(fsa_results[0]['reward'])
        min_rm_rewards = [np.inf]*len(rm_results[0]['reward'])
        min_rewards = [min_lof_rewards, min_flat_rewards, min_fsa_rewards, min_rm_rewards]
        ave_lof_rewards = [0]*len(lof_results[0]['reward'])
        ave_flat_rewards = [0]*len(flat_results[0]['reward'])
        ave_fsa_rewards = [0]*len(fsa_results[0]['reward'])
        ave_rm_rewards = [0]*len(rm_results[0]['reward'])
        ave_rewards = [ave_lof_rewards, ave_flat_rewards, ave_fsa_rewards, ave_rm_rewards]

    ########## Take the results from the 4 tasks and average them ##########

    ave_lof_results = [0]*len(lof_results[0]['reward'])
    ave_flat_results = [0]*len(flat_results[0]['reward'])
    ave_fsa_results = [0]*len(fsa_results[0]['reward'])
    ave_rm_results = [0]*len(rm_results[0]['reward'])

    all_ave_results = [ave_lof_results, ave_flat_results, ave_fsa_results, ave_rm_results]

    for i, method_results in enumerate(all_results):
        for task_result, bounds in zip(method_results, task_reward_bounds):
            for j, reward in enumerate(task_result['reward']):
                task_result['reward'][j] = (reward - bounds[0])/(bounds[1]-bounds[0])
                all_ave_results[i][j] += task_result['reward'][j]/len(all_ave_results)

        # plt.plot(all_results[i][0]['steps'], all_ave_results[i])

    for i, method_ave_rewards in enumerate(all_ave_results):
        for j, ave_reward in enumerate(method_ave_rewards):
            ave_rewards[i][j] += ave_reward/num_exp

            if ave_reward > max_rewards[i][j]:
                max_rewards[i][j] = ave_reward
            if ave_reward < min_rewards[i][j]:
                min_rewards[i][j] = ave_reward
    
    
task_names = ['Logical Options', 'Options', 'Options+FSA', 'Reward Machines']
task_colors = ['b', 'r', 'y', 'g']
for i, (task_name, task_color, ave_reward, min_reward, max_reward) in enumerate(zip(task_names, task_colors, ave_rewards, min_rewards, max_rewards)):
    plt.plot(all_results[i][0]['steps'], ave_reward, color=task_color, label=task_name)    
    plt.fill_between(all_results[i][0]['steps'], min_reward, max_reward, color=task_color, alpha=0.2)

plt.ylim(0, 1)
plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.2))
plt.tight_layout()
plt.savefig('results.png')

# plt.plot(flat_results[0]['steps'], ave_flat_rewards, 'b', label='Options')
# plt.fill_between(flat_results[0]['steps'], min_flat)
# plt.plot(fsa_results[0]['steps'], ave_fsa_rewards, 'r', label='Options+FSA')
# plt.plot(rm_results[0]['steps'], ave_rm_rewards, 'y', label='Reward Machines')
# plt.plot(lof_results[0]['steps'], ave_lof_rewards, 'g', label='Logical Options')
# plt.ylim(0, 1)
# plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.2))
# plt.tight_layout()
# plt.savefig('results.png')