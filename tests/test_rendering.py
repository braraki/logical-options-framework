import time
from simulator.rendering import Viewer
from simulator.balldrop import BallDropSim

def test_rendering():
    sim = BallDropSim()
    sim.reset()
    sim.env.props[0].eval(sim.env.obj_dict)
    for i in range(10):
        sim.render()
        obs = sim.step(None)
    sim.render()
    return 0

if __name__ == '__main__':
    test_rendering()