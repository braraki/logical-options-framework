import time
from simulator.rendering import Viewer
from simulator.driveworld import DriveWorldSim
from simulator.policy import *
from celluloid import Camera

from .options import *

# human: show plots
# anim: don't show plots but save a gif
# note: the first call to 'render' sets the render mode
render_mode = 'anim'

# 'vi', 'lvi', 'hardcoded', 'options'
policy_mode = 'options'

def make_tm_driveworld():
    
    # go to goal A, then B, then A

    # prop order:
    # goal_a, goal_b

    nF = 5
    nP = 6
    tm = np.zeros((nF, nF, nP))

    # initial state
    #   a b c o e
    # 0 0 1 1 0 1
    # 1 1 0 0 0 0
    # 2 0 0 0 0 0
    # 3 0 0 0 0 0
    # 4 0 0 0 1 0
    tm[0, 1, 0] = 1
    tm[0, 0, 1] = 1
    tm[0, 0, 2] = 1
    tm[0, 4, 3] = 1
    tm[0, 0, 4] = 1
    # S1
    #   a b c o e
    # 0 0 0 0 0 0
    # 1 1 0 1 0 1
    # 2 0 1 0 0 0
    # 3 0 0 0 0 0
    # 4 0 0 0 1 0
    tm[1, 1, 0] = 1
    tm[1, 2, 1] = 1
    tm[1, 1, 2] = 1
    tm[1, 4, 3] = 1
    tm[1, 1, 4] = 1
    # S2
    #   a b c o e
    # 0 0 0 0 0 0
    # 1 0 0 0 0 0
    # 2 1 1 0 0 1
    # 3 0 0 1 0 0
    # 4 0 0 0 1 0
    tm[2, 2, 0] = 1
    tm[2, 2, 1] = 1
    tm[2, 3, 2] = 1
    tm[2, 4, 3] = 1
    tm[2, 2, 4] = 1
    # G
    #   a b c o e
    # 0 0 0 0 0 0
    # 1 0 0 0 0 0
    # 2 0 0 0 0 0
    # 3 1 1 1 1 1
    # 4 0 0 0 0 0
    tm[3, 3, :] = 1
    # T
    #   a b c o e
    # 0 0 0 0 0 0
    # 1 0 0 0 0 0
    # 2 0 0 0 0 0
    # 3 0 0 0 0 0
    # 4 1 1 1 1 1
    tm[4, 4, :] = 1

    return tm

def test_options(sim, tm=None):
    sim.reset()

    policy = RRTOptionsPolicy()
    policy.make_policy(sim.env, tm)
    all_paths = policy.make_rrt_policy(sim.env)
    f = 0

    # iterate over options
    for i in range(4):
        sim.render(mode=render_mode)
        option = policy.get_option(sim.env, f)
        s_idx = sim.env.state_to_idx(sim.env.get_state())
        path = all_paths[option][s_idx]
        print("option: {}, state: {}, s_idx: {}, g_idx: {} path: {}".format(option, sim.env.get_state(), s_idx, sim.env.state_to_idx(path[-1]), path))
        old_f = f
        camera = sim.render_rrt(path)
        # iterate over actions in each option
        for j in range(10):
            # this isn't actually the best way to detect
            # if a policy is over. the policy should tell you
            # when it has terminated (reached its goal)
            if old_f != f:
                break
            action = policy.get_action(sim.env, option)
            obs = sim.step(action)
            f = policy.get_fsa_state(sim.env, f, tm)
            print("option: {} | FSA state: {} | state: {}".format(option, f, sim.env.get_state()))
            camera = sim.render_rrt(path)
        if f == 3:
            break

    if render_mode == 'anim':
        animation = camera.animate()
        animation.save(sim.env.name + '_options.gif', writer='imagemagick')
    return 0

# fix the goal so that it is to put ball a THEN ball b into the basket

if __name__ == '__main__':
    sim = DriveWorldSim()
    tm = make_tm_driveworld()
    test_options(sim, tm)