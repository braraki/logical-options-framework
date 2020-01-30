import time
from simulator.rendering import Viewer
from simulator.balldrop import BallDropSim
from simulator.policy import *
from celluloid import Camera

# human: show plots
# anim: don't show plots but save a gif
# note: the first call to 'render' sets the render mode
render_mode = 'human'

def test_rendering():
    sim = BallDropSim()
    sim.reset()
    initial_state = sim.env.get_state()
    policy = VIPolicy()
    policy.make_transition_function(sim.env)
    sim.env.set_state(initial_state)
    policy.make_policy(sim.env)
    # policy = HardCodedPolicy()
    for i in range(20):
        sim.render(mode=render_mode)
        action = policy.get_action(sim.env)
        obs = sim.step(action)
    camera = sim.render()

    if render_mode == 'anim':
        animation = camera.animate()
        animation.save('balldrop.gif', writer='imagemagick')
    return 0

# fix the goal so that it is to put ball a THEN ball b into the basket

if __name__ == '__main__':
    test_rendering()