import time
from simulator.rendering import Viewer
from simulator.balldrop import BallDropSim
from simulator.policy import Policy
from celluloid import Camera

# human: show plots
# anim: don't show plots but save a gif
# note: the first call to 'render' sets the render mode
render_mode = 'anim'

def test_rendering():
    sim = BallDropSim()
    policy = Policy()
    sim.reset()
    for i in range(20):
        sim.render(mode=render_mode)
        action = policy.get_action(sim.env)
        obs = sim.step(action)
    camera = sim.render()

    if render_mode == 'anim':
        animation = camera.animate()
        animation.save('balldrop.gif', writer='imagemagick')
    return 0

if __name__ == '__main__':
    test_rendering()