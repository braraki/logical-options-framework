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

    task_names = ['composite', 'sequential', 'OR', 'IF']
    # task_names = ['composite']
    make_taskspecs = [make_taskspec_delivery1, make_taskspec_delivery2, make_taskspec_delivery3, make_taskspec_delivery4]
    # make_taskspecs = [make_taskspec_delivery1]
    make_taskspecs_rm = [make_taskspec_delivery1_rm, make_taskspec_delivery2_rm, make_taskspec_delivery3_rm, make_taskspec_delivery4_rm]
    # make_taskspecs_rm = [make_taskspec_delivery1_rm]
    task_spec_and_safety_props = [make_taskspec() for make_taskspec in make_taskspecs] 
    task_spec_and_safety_props_rm = [make_taskspec() for make_taskspec in make_taskspecs_rm] 

    safety_specs = make_safetyspecs_delivery()
    subgoals = make_subgoals_delivery(sim.env)

    num_episodes = 1601
    recording_frequency = 20

    for i, task_name in enumerate(task_names):
        task_spec_rm, safety_props = task_spec_and_safety_props_rm[i]
        print("--------- Training RMs ---------")
        rm_policy = RewardMachineMetaPolicy(subgoals, task_spec_rm, safety_props, sim.env,
                record_training=True, recording_frequency=recording_frequency, num_episodes=num_episodes, experiment_num=exp_num)
        save_dataset('satisfaction', 'rm', task_name, exp_num, rm_policy.get_results())

    for i, task_name in enumerate(task_names):
        task_spec, safety_props = task_spec_and_safety_props[i]

        print("######## TASK SPEC {} ##########".format(i+1))
        print("--------- Training LOF ---------")
        lof_policy = QLearningMetaPolicy(subgoals, task_spec, safety_props, safety_specs, sim.env,
                record_training=True, recording_frequency=recording_frequency, num_episodes=num_episodes, experiment_num=exp_num)
        # lof_results.append(lof_policy.get_results())
        save_dataset('satisfaction', 'lof', task_name, exp_num, lof_policy.get_results())

        print("---- Training Flat Options -----")
        flat_policy = FlatQLearningMetaPolicy(subgoals, task_spec, safety_props, safety_specs, sim.env,
                record_training=True, recording_frequency=recording_frequency, num_episodes=num_episodes, experiment_num=exp_num)
        # flat_results.append(flat_policy.get_results())
        save_dataset('satisfaction', 'flat', task_name, exp_num, flat_policy.get_results())

        print("----- Training FSA Options -----")
        fsa_policy = FSAQLearningMetaPolicy(subgoals, task_spec, safety_props, safety_specs, sim.env,
                record_training=True, recording_frequency=recording_frequency, num_episodes=num_episodes, experiment_num=exp_num)
        # fsa_results.append(fsa_policy.get_results())
        save_dataset('satisfaction', 'fsa', task_name, exp_num, fsa_policy.get_results())

        print("--- Training Greedy Options ----")
        greedy_policy = GreedyQLearningMetaPolicy(subgoals, task_spec, safety_props, safety_specs, sim.env,
                record_training=True, recording_frequency=recording_frequency, num_episodes=num_episodes, experiment_num=exp_num)
        # fsa_results.append(fsa_policy.get_results())
        save_dataset('satisfaction', 'greedy', task_name, exp_num, greedy_policy.get_results())

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