import time
from simulator.rendering import Viewer
from simulator.balldrop import BallDropSim
from simulator.lineworld import LineWorldSim
from simulator.gridworld import GridWorldSim
from simulator.driveworld import DriveWorldSim
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
    # 2 0 1 0 0 1
    # 3 1 0 1 0 0
    # 4 0 0 0 1 0
    tm[2, 3, 0] = 1
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

def make_tm_driveworld():
    # go to goal A, then B, then A

    # prop order:
    # goal_a, goal_b

    nF = 10
    nP = 9
    tm = np.zeros((nF, nF, nP))

    # initial state
    #   go gs gc  o  s  c  l ob  e
    # 0  1  1  1  0  0  0  1  0  1
    # 1  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  1  0  0  0  0  0
    # 4  0  0  0  0  1  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0
    # 6  0  0  0  0  0  1  0  0  0
    # 7  0  0  0  0  0  0  0  0  0
    # 8  0  0  0  0  0  0  0  0  0
    # 9  0  0  0  0  0  0  0  1  0
    tm[0, 0, 0] = 1 # go
    tm[0, 0, 1] = 1 # gs
    tm[0, 0, 2] = 1 # gc
    tm[0, 3, 3] = 1 #  o
    tm[0, 4, 4] = 1 #  s
    tm[0, 6, 5] = 1 #  c
    tm[0, 0, 6] = 1 #  l
    tm[0, 9, 7] = 1 # ob
    tm[0, 0, 8] = 1 #  e
    # S1 [overtaking and in the left lane]
    #   go gs gc  o  s  c  l ob  e
    # 0  0  0  0  0  0  0  0  0  0
    # 1  0  1  1  1  1  1  1  0  0
    # 2  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  1
    # 4  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0
    # 6  0  0  0  0  0  0  0  0  0
    # 7  0  0  0  0  0  0  0  0  0
    # 8  1  0  0  0  0  0  0  0  0
    # 9  0  0  0  0  0  0  0  1  0
    tm[1, 8, 0] = 1 # go
    tm[1, 1, 1:7] = 1 # gs, gc, o, s, c, l
    tm[1, 9, 7] = 1 # ob
    tm[1, 3, 8] = 1 #  e
    # S2 [overtaking and reached goal but in the left lane] [currently doesn't happen in practice]
    #   go gs gc  o  s  c  l ob  e
    # 0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0
    # 2  1  1  1  1  1  1  1  0  1
    # 3  0  0  0  0  0  0  0  0  0
    # 4  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0
    # 6  0  0  0  0  0  0  0  0  0
    # 7  0  0  0  0  0  0  0  0  0
    # 8  0  0  0  0  0  0  0  0  0
    # 9  0  0  0  0  0  0  0  1  0
    tm[2, 2, 0:7] = 1 # go, gs, gc, o, s, c, l
    tm[2, 9, 7] = 1 # ob
    tm[2, 2, 8] = 1 #  e
    # S3 [overtaking and not in the left lane]
    #   go gs gc  o  s  c  l ob  e
    # 0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  1  0  0
    # 2  0  0  0  0  0  0  0  0  0
    # 3  0  1  1  1  1  1  0  0  1
    # 4  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0
    # 6  0  0  0  0  0  0  0  0  0
    # 7  0  0  0  0  0  0  0  0  0
    # 8  1  0  0  0  0  0  0  0  0
    # 9  0  0  0  0  0  0  0  1  0
    tm[3, 8, 0] = 1 # go
    tm[3, 3, 1:6] = 1 # gs, gc, o, s, c
    tm[3, 1, 6] = 1 #  l
    tm[3, 9, 7] = 1 # ob
    tm[3, 3, 8] = 1 #  e
    # S4 [straight and the left lane is bad]
    #   go gs gc  o  s  c  l ob  e
    # 0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0
    # 4  1  0  1  1  1  1  0  0  1
    # 5  0  0  0  0  0  0  0  0  0
    # 6  0  0  0  0  0  0  0  0  0
    # 7  0  0  0  0  0  0  0  0  0
    # 8  0  1  0  0  0  0  0  0  0
    # 9  0  0  0  0  0  0  1  1  0
    tm[4, 4, 0] = 1 # go
    tm[4, 8, 1] = 1 # gs
    tm[4, 4, 2] = 1 # gc
    tm[4, 4, 3] = 1 #  o
    tm[4, 4, 4] = 1 #  s
    tm[4, 4, 5] = 1 #  c
    tm[4, 9, 6] = 1 #  l
    tm[4, 9, 7] = 1 # ob
    tm[4, 4, 8] = 1 #  e
    # S5 [changing lanes and in the left lane]
    #   go gs gc  o  s  c  l ob  e
    # 0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0
    # 4  0  0  0  0  0  0  0  0  0
    # 5  1  1  0  1  1  1  1  0  0
    # 6  0  0  0  0  0  0  0  0  1
    # 7  0  0  0  0  0  0  0  0  0
    # 8  0  0  1  0  0  0  0  0  0
    # 9  0  0  0  0  0  0  0  1  0
    tm[5, 5, 0] = 1 # go
    tm[5, 5, 1] = 1 # gs
    tm[5, 8, 2] = 1 # gc
    tm[5, 5, 3] = 1 #  o
    tm[5, 5, 4] = 1 #  s
    tm[5, 5, 5] = 1 #  c
    tm[5, 5, 6] = 1 #  l
    tm[5, 9, 7] = 1 # ob
    tm[5, 6, 8] = 1 #  e
    # S6 [changing lanes and not in the left lane]
    #   go gs gc  o  s  c  l ob  e
    # 0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0
    # 4  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  1  0  0
    # 6  1  1  0  1  1  1  0  0  1
    # 7  0  0  0  0  0  0  0  0  0
    # 8  0  0  1  0  0  0  0  0  0
    # 9  0  0  0  0  0  0  0  1  0
    tm[6, 6, 0] = 1 # go
    tm[6, 6, 1] = 1 # gs
    tm[6, 8, 2] = 1 # gc
    tm[6, 6, 3] = 1 #  o
    tm[6, 6, 4] = 1 #  s
    tm[6, 6, 5] = 1 #  c
    tm[6, 5, 6] = 1 #  l
    tm[6, 9, 7] = 1 # ob
    tm[6, 6, 8] = 1 #  e
    # S7 [changing lanes and reached goal but not in left lane] [currently doesn't happen in practice]
    #   go gs gc  o  s  c  l ob  e
    # 0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0
    # 4  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0
    # 6  0  0  0  0  0  0  0  0  0
    # 7  1  1  1  1  1  1  1  0  1
    # 8  0  0  0  0  0  0  0  0  0
    # 9  0  0  0  0  0  0  0  1  0
    tm[7, 7, 0] = 1 # go
    tm[7, 7, 1] = 1 # gs
    tm[7, 7, 2] = 1 # gc
    tm[7, 7, 3] = 1 #  o
    tm[7, 7, 4] = 1 #  s
    tm[7, 7, 5] = 1 #  c
    tm[7, 7, 6] = 1 #  l
    tm[7, 9, 7] = 1 # ob
    tm[7, 7, 8] = 1 #  e
    # G
    #   go gs gc  o  s  c  l ob  e
    # 0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0
    # 4  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0
    # 6  0  0  0  0  0  0  0  0  0
    # 7  0  0  0  0  0  0  0  0  0
    # 8  1  1  1  1  1  1  1  1  1
    # 9  0  0  0  0  0  0  0  0  0
    tm[8, 8, :] = 1
    # T
    #   go gs gc  o  s  c  l ob  e
    # 0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0
    # 4  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0
    # 6  0  0  0  0  0  0  0  0  0
    # 7  0  0  0  0  0  0  0  0  0
    # 8  0  0  0  0  0  0  0  0  0
    # 9  1  1  1  1  1  1  1  1  1
    tm[9, 9, :] = 1

    return tm

def test_options(sim, tm=None):
    sim.reset()

    policy = OptionsPolicy()
    policy.make_policy(sim.env, tm)
    f = 0
    goal_state = 8


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
        if f == goal_state:
            break

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

    # sim = DriveWorldSim()
    # tm = make_tm_driveworld()
    # test_options(sim, tm)