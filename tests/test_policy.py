import time
from simulator.rendering import Viewer
from simulator.balldrop import BallDropSim
from simulator.policy import *
from celluloid import Camera

# human: show plots
# anim: don't show plots but save a gif
# note: the first call to 'render' sets the render mode
render_mode = 'human'

# 'vi', 'lvi', 'hardcoded'
policy_mode = 'lvi'

def make_tm():
    # drop ball A in basket, THEN drop ball B in basket

    # prop order:
    # bainb, bbinb, hba, hbb

    nF = 4
    nP = 5
    tm = np.zeros((nF, nF, nP))

    # initial state
    #   a b c d e
    # 0 0 1 1 1 1
    # 1 1 0 0 0 0
    # 2 0 0 0 0 0
    # 3 0 0 0 0 0
    tm[0, 0, 1:] = 1
    tm[0, 1, 0] = 1
    # S1
    #   a b c d e
    # 0 0 0 0 0 0
    # 1 1 0 1 1 1
    # 2 0 1 0 0 0
    # 3 0 0 0 0 0
    tm[1, 1, :] = 1
    tm[1, 1, 1] = 0
    tm[1, 2, 1] = 1
    # S2
    #   a b c d e
    # 0 0 0 0 0 0
    # 1 0 0 0 0 0
    # 2 1 1 1 1 1
    # 3 0 0 0 0 0
    tm[2, 2, :] = 1
    tm[3, 3, :] = 1

    return tm

def test_rendering():
    sim = BallDropSim()
    sim.reset()
    if policy_mode == 'vi':
        policy = VIPolicy()
        policy.make_policy(sim.env)
    elif policy_mode == 'hardcoded':
        policy = HardCodedPolicy()
    elif policy_mode == 'lvi':
        policy = LVIPolicy()
        tm = make_tm()
        policy.make_policy(sim.env, tm)
        f = 0

    for i in range(20):
        sim.render(mode=render_mode)
        if policy_mode == 'lvi':
            action = policy.get_action(sim.env, f)
        else:
            action = policy.get_action(sim.env)
        obs = sim.step(action)
        f = policy.get_fsa_state(sim.env, f, tm)
        print(f)
    camera = sim.render()

    if render_mode == 'anim':
        animation = camera.animate()
        animation.save('balldrop.gif', writer='imagemagick')
    return 0

# fix the goal so that it is to put ball a THEN ball b into the basket

if __name__ == '__main__':
    test_rendering()