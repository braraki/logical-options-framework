import time
from simulator.rendering import Viewer
from simulator.balldrop import BallDropSim
from simulator.policy import Policy

def test_rendering():
    sim = BallDropSim()
    policy = Policy()
    sim.reset()
    # sim.env.props[0].eval(sim.env.obj_dict, 0)
    for i in range(20):
        sim.render()
        action = policy.get_action(sim.env)
        obs = sim.step(action)
    sim.render()
    return 0

if __name__ == '__main__':
    test_rendering()