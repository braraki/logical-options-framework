import time
from simulator.rendering import Viewer
from simulator.delivery import DeliverySim
from simulator.options import *
from simulator.reward_machines import *
from celluloid import Camera
from specs import *

# human: show plots
# anim: don't show plots but save a gif
# note: the first call to 'render' sets the render mode
render_mode = 'anim'

def run_experiment(sim, exp_num=1):
    sim.reset()

    training_task_name = 'composite'
    training_taskspec, safety_props = make_taskspec_delivery1()

    task_names = ['composite', 'sequential', 'OR', 'IF']
    make_taskspecs = [make_taskspec_delivery1, make_taskspec_delivery2, make_taskspec_delivery3, make_taskspec_delivery4]
    task_spec_and_safety_props = [make_taskspec() for make_taskspec in make_taskspecs] 


    safety_specs = make_safetyspecs_delivery()
    subgoals = make_subgoals_delivery(sim.env)

    num_episodes = 1001

    # train the options/policy for a single task
    print("--------- Training LOF on Task {} ---------".format(training_task_name))
    lof_policy = QLearningMetaPolicy(subgoals, training_taskspec, safety_props, safety_specs, sim.env,
            record_training=False, num_episodes=num_episodes, experiment_num=exp_num)

    print("----- Training FSA Options -----")
    fsa_policy = FSAQLearningMetaPolicy(subgoals, training_taskspec, safety_props, safety_specs, sim.env,
            record_training=False, num_episodes=num_episodes, experiment_num=exp_num)

    print("--- Training Greedy Options ----")
    greedy_policy = GreedyQLearningMetaPolicy(subgoals, training_taskspec, safety_props, safety_specs, sim.env,
            record_training=False, num_episodes=num_episodes, experiment_num=exp_num)

    num_iter = 300
    # test the trained options on other tasks
    for i, task_name in enumerate(task_names):
        task_spec, safety_props = task_spec_and_safety_props[i]

        # print("######## TASK SPEC {} ##########".format(task_name))
        print("--------- LOF Results ---------")
        lof_policy.record_composability_results(sim.env, task_spec, num_iter)
        lof_results = lof_policy.get_composability_results(task_name)
        save_dataset('composability', 'lof', task_name, exp_num, lof_results)
        # print("reward: {} | success: {} | final f: {}".format(reward, success, final_f))

        print("----- Training FSA Options -----")
        fsa_policy.record_composability_results(sim.env, task_spec, num_iter)
        fsa_results = fsa_policy.get_composability_results(task_name)
        save_dataset('composability', 'fsa', task_name, exp_num, fsa_results)
        # print("reward: {} | success: {} | final f: {}".format(reward, success, final_f))

        print("--- Training Greedy Options ----")
        greedy_policy.record_composability_results(sim.env, task_spec, num_iter)
        greedy_results = greedy_policy.get_composability_results(task_name)
        save_dataset('composability', 'greedy', task_name, exp_num, greedy_results)
        # print("reward: {} | success: {} | final f: {}".format(reward, success, final_f))
        print('f')

        save_dataset('composability', 'lof', task_name, exp_num, lof_results)

def save_dataset(exp_name, method_name, task_name, exp_num, results):
    directory = Path(__file__).parent.parent / 'dataset' / exp_name / method_name /task_name
    # if directory doesn't exist, create it
    Path(directory).mkdir(parents=True, exist_ok=True)
    file_name = str(exp_num) + '.npz'
    path_name = directory / file_name
    np.savez(path_name, results)

def run_multiple_experiments(sim, num_exp=10):
    for i in range(num_exp):
        print("$$$$$$$$ EXP NUMBER {} $$$$$$$$$$".format(i))
        run_experiment(sim, exp_num=i)

if __name__ == '__main__':
    sim = DeliverySim()
    run_multiple_experiments(sim)