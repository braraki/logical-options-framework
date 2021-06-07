# this is basically the model-free version of LOF with a TM
# aka, this version has no TM, just FSA states, and it uses
# q learning instead of value iteration to learn the high-level policy

import time
from simulator.rendering import Viewer
from simulator.delivery import DeliverySim
from simulator.options import *
from simulator.reward_machines import *
from celluloid import Camera
import matplotlib.pyplot as plt
from pathlib import Path
from specs import *

# human: show plots
# anim: don't show plots but save a gif
# note: the first call to 'render' sets the render mode
render_mode = 'anim'

def test_rewardmachine(sim, task_spec=None):
    directory = Path(__file__).parent.parent / 'dataset' / 'tests'
    # if directory doesn't exist, create it
    Path(directory).mkdir(parents=True, exist_ok=True)

    sim.reset()

    # delivery1_rm: composite task
    # delivery2_rm: sequential task
    # delivery3_rm: OR task
    # delivery4_rm: IF task
    task_spec, safety_props = make_taskspec_delivery1_rm()
    subgoals = make_subgoals_delivery(sim.env)

    policy = RewardMachineMetaPolicy(subgoals, task_spec, safety_props, sim.env,
                                    record_training=True, recording_frequency=20, num_episodes=2000)
    results = policy.get_results()

    plt.plot(results['steps'], results['reward'])
    results_path = directory / 'results_rm.png'
    plt.savefig(results_path)

    f = 0
    goal_state = task_spec.nF - 2
    trap_state = task_spec.nF - 1

    for i in range(100):
        sim.render(mode=render_mode)
        camera = sim.render()

        action = policy.get_action(sim.env, f)
        obs = sim.step(action)
        f = policy.get_fsa_state(sim.env, f)

        if f == 1:
            if np.random.uniform() < 0.0:
                sim.env.prop_dict['canceled'].value = True
            else:
                sim.env.prop_dict['canceled'].value = False
        else:
            sim.env.prop_dict['canceled'].value = False

        if f == goal_state or f == trap_state:
            break

    if render_mode == 'anim':
        animation = camera.animate()
        animation_file = sim.env.name + '_rm.gif'
        animation_path = directory / animation_file
        animation.save(animation_path, writer='imagemagick')
    return 0

if __name__ == '__main__':
    sim = DeliverySim()
    test_rewardmachine(sim)