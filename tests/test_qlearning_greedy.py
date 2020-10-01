# this is basically the model-free version of LOF with a TM
# aka, this version has no TM, just FSA states, and it uses
# q learning instead of value iteration to learn the high-level policy

import time
from simulator.rendering import Viewer
from simulator.delivery import DeliverySim
from simulator.options import *
from celluloid import Camera
import matplotlib.pyplot as plt
from pathlib import Path
from specs import *

# human: show plots
# anim: don't show plots but save a gif
# note: the first call to 'render' sets the render mode
render_mode = 'anim'

def test_qlearning(sim, task_spec=None):
    directory = Path(__file__).parent.parent / 'dataset' / 'tests'
    # if directory doesn't exist, create it
    Path(directory).mkdir(parents=True, exist_ok=True)

    sim.reset()

    # delivery1: composite task
    # delivery2: sequential task
    # delivery3: OR task
    # delivery4: IF task
    task_spec, safety_props = make_taskspec_delivery1()
    safety_specs = make_safetyspecs_delivery()
    subgoals = make_subgoals_delivery(sim.env)

    policy = GreedyQLearningMetaPolicy(subgoals, task_spec, safety_props, safety_specs, sim.env,
                                    record_training=True, recording_frequency=20, num_episodes=1000)

    results = policy.get_results()

    plt.plot(results['steps'], results['reward'])
    results_path = directory / 'results_greedy.png'
    plt.savefig(results_path)

    f = 0
    goal_state = task_spec.nF - 1
    max_steps_in_option = 30

    for i in range(5):
        sim.render(mode=render_mode)
        camera = sim.render()
        option = policy.get_option(sim.env, f)
        f_prev = f
        steps_in_option = 0
        while not policy.is_terminated(sim.env, option) and f_prev == f and steps_in_option < max_steps_in_option:
            action = policy.get_action(sim.env, option)
            obs = sim.step(action)
            f_prev = f
            f = policy.get_fsa_state(sim.env, f)
            print("option: {} | FSA state: {} | state: {}".format(option, f, sim.env.get_state()))
            camera = sim.render()
            steps_in_option += 1
            if f == 1:
                if np.random.uniform() < 1.0:
                    sim.env.prop_dict['canceled'].value = True
                else:
                    sim.env.prop_dict['canceled'].value = False
            else:
                sim.env.prop_dict['canceled'].value = False
        if f == goal_state:
            break
        

    if render_mode == 'anim':
        animation = camera.animate()
        animation_file = sim.env.name + '_greedy.gif'
        animation_path = directory / animation_file
        animation.save(animation_path, writer='imagemagick')
    return 0

if __name__ == '__main__':
    sim = DeliverySim()
    test_qlearning(sim)