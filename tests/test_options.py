import time
from simulator.rendering import Viewer
from simulator.balldrop import BallDropSim
from simulator.lineworld import LineWorldSim
from simulator.gridworld import GridWorldSim
from simulator.policy import *
from celluloid import Camera

# human: show plots
# anim: don't show plots but save a gif
# note: the first call to 'render' sets the render mode
render_mode = 'anim'

# 'vi', 'lvi', 'hardcoded', 'options'
policy_mode = 'options'

def make_tm_balldrop():
    # drop ball A in basket, THEN drop ball B in basket

    # prop order:
    # ainb, ainb, abinb, hba, hbb, ainbhbb, binbhba

    nF = 4
    nP = 8
    tm = np.zeros((nF, nF, nP))

    # initial state
    #   a b c d e f g
    # 0 0 0 1 1 1 0 1
    # 1 1 0 0 0 0 1 0
    # 2 0 0 0 0 0 0 0
    # 3 0 0 0 0 0 0 0
    tm[0, 0, :] = 1
    tm[0, 0, 0] = 0
    tm[0, 1, 0] = 1
    tm[0, 0, 5] = 0
    tm[0, 1, 5] = 1
    # S1
    #   a b c d e f g
    # 0 0 0 0 0 0 0 0
    # 1 1 0 0 1 1 1 0
    # 2 0 1 1 0 0 0 1
    # 3 0 0 0 0 0 0 0
    tm[1, 1, :] = 1
    tm[1, 1, 1] = 0
    tm[1, 2, 1] = 1
    tm[1, 1, 2] = 0
    tm[1, 2, 2] = 1
    tm[1, 1, 6] = 0
    tm[1, 2, 6] = 1
    # S2
    #   a b c d e f g
    # 0 0 0 0 0 0 0 0
    # 1 0 0 0 0 0 0 0
    # 2 1 1 1 1 1 1 1
    # 3 0 0 0 0 0 0 0
    tm[2, 2, :] = 1
    tm[3, 3, :] = 1

    return tm

def make_tm_lineworld():
    # go to goal A, then B, then A

    # prop order:
    # goal_a, goal_b

    nF = 5
    nP = 3
    tm = np.zeros((nF, nF, nP))

    # initial state
    #   a b e
    # 0 0 1 1
    # 1 1 0 0
    # 2 0 0 0
    # 3 0 0 0
    tm[0, 0, :] = 1
    tm[0, 0, 0] = 0
    tm[0, 1, 0] = 1
    # S1
    #   a b e
    # 0 0 0 0
    # 1 1 0 1
    # 2 0 1 0
    # 3 0 0 0
    tm[1, 1, :] = 1
    tm[1, 1, 1] = 0
    tm[1, 2, 1] = 1
    # S2
    #   a b e
    # 0 0 0 0
    # 1 0 0 0
    # 2 0 1 1
    # 3 1 0 0
    tm[2, 2, :] = 1
    tm[2, 2, 0] = 0
    tm[2, 3, 0] = 1
    # G
    #   a b e
    # 0 0 0 0
    # 1 0 0 0
    # 2 0 0 0
    # 3 1 1 1
    tm[3, 3, :] = 1
    # T
    #   a b e
    # 0 0 0 0
    # 1 0 0 0
    # 2 0 0 0
    # 3 0 0 0
    # 4 1 1 1
    tm[4, 4, :] = 1

    return tm

def make_tm_gridworld():
    # go to goal A, then B, then A

    # prop order:
    # goal_a, goal_b

    nF = 5
    nP = 5
    tm = np.zeros((nF, nF, nP))

    # initial state
    #   a b c o e
    # 0 0 1 0 0 1
    # 1 1 0 0 0 0
    # 2 0 0 0 0 0
    # 3 0 0 1 0 0
    # 4 0 0 0 1 0
    tm[0, 1, 0] = 1
    tm[0, 0, 1] = 1
    tm[0, 3, 2] = 1
    tm[0, 4, 3] = 1
    tm[0, 0, 4] = 1
    # S1
    #   a b c o e
    # 0 0 0 0 0 0
    # 1 1 0 1 0 1
    # 2 0 0 0 0 0
    # 3 0 1 0 0 0
    # 4 0 0 0 1 0
    tm[1, 1, 0] = 1
    tm[1, 3, 1] = 1
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
    tm[2, 2, 2] = 1
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

    policy = OptionsPolicy()
    policy.make_policy(sim.env, tm)
    f = 0

    for i in range(30):
        sim.render(mode=render_mode)
        option = policy.get_option(sim.env, f)
        action = policy.get_action(sim.env, option)
        obs = sim.step(action)
        f = policy.get_fsa_state(sim.env, f, tm)
        print("option: {} | FSA state: {}".format(option, f))
    camera = sim.render()

    if render_mode == 'anim':
        animation = camera.animate()
        animation.save(sim.env.name + '_options.gif', writer='imagemagick')
    return 0

# fix the goal so that it is to put ball a THEN ball b into the basket

if __name__ == '__main__':
    # sim = BallDropSim()
    # tm = make_tm_balldrop()

    # sim = LineWorldSim()
    # tm = make_tm_lineworld()

    sim = GridWorldSim()
    tm = make_tm_gridworld()
    test_options(sim, tm)