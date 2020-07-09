"""1D arm picks up and drops balls into a basket"""
import numpy as np
from numpy.random import randint
from collections import OrderedDict
# efficient sparse matrix construction:
from scipy.sparse import dok_matrix
# efficient matrix-vector multiplication:
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from celluloid import Camera

from .simulator import Sim
from .rendering import Viewer
from .object import *
from .proposition import *
from .environment import *

class DriveWorldSim(Sim):

    def __init__(self, dom_size=[8, 8], n_domains=100, n_traj=2):
        self.dom_size = dom_size
        self.viewer = None
        self.camera = None
        self.env = None # Env is defined in reset()
        self.obj_dict = OrderedDict()
        self.action_dict = {
            'nothing': 0, 'left': 1, 'right': 2, 'up': 3, 'down': 4
        }

    def reset(self):
        self.obj_dict.clear()
        self.env =DriveWorldEnv(
            name='DriveWorldEnv',
            dom_size=self.dom_size,
            action_dict=self.action_dict
        )

        self.obj_dict = self.add_objects()
        self.init_objects()
        self.env.add_objects(self.obj_dict)

        self.prop_dict = self.add_props()
        self.env.add_props(self.prop_dict)

    # this allows you to access propositions as attributes of this class
    # so for example instead of doing self.prop_dict['ball_a'], you can
    # just do self.ball_a
    def __getattr__(self, name):
        return self.obj_dict[name]

    def init_objects(self):
        self.make_agent()
        self.make_goals()
        self.make_obstacles()

    def add_props(self):
        prop_dict = OrderedDict()

        # "Agent on Goal A"
        prop_dict['ona'] = SameLocationProp(
            name='ona',
            obj_name1='agent',
            obj_name2='goal_a'
            )

        # "Agent on Goal B"
        prop_dict['onb'] = SameLocationProp(
            name='onb',
            obj_name1='agent',
            obj_name2='goal_b'
            )

        # "Agenet on Goal C"
        prop_dict['onc'] = SameLocationProp(
            name='onc',
            obj_name1='agent',
            obj_name2='goal_c'
        )

        # "Agent on Obstacle"
        prop_dict['onobstacle'] = OnObstacleProp(
            name='onobstacle',
            obj_name='agent',
            obstacle_name='obstacles'
        )

        return prop_dict

    # create and initialize objects
    # and add them to the env
    def add_objects(self):
        obj_dict = OrderedDict()

        obj_dict['agent'] = GridAgentObj(name='agent', color=[.7, .7, .7],
            state_space=self.dom_size)
        obj_dict['goal_a'] = GridGoalObj(name='a', color=[1, 0, 0], 
            state_space=self.dom_size)
        obj_dict['goal_b'] = GridGoalObj(name='b', color=[0, 1, 0], 
            state_space=self.dom_size)
        obj_dict['goal_c'] = GridGoalObj(name='c', color=[0, 0, 1], 
            state_space=self.dom_size)
        obj_dict['obstacles'] = ObstacleObj(name='obstacles', color=[0, 0, 0],
            dom_size=self.dom_size)

        return obj_dict

    def make_agent(self):
        self.agent.create(
            x_range=[0, self.dom_size[0] - 1],
            y_range=[0, self.dom_size[1] - 1]
        )

    def make_goals(self):
        self.goal_a.create(
            x_range=[0, self.dom_size[0] - 1],
            y_range=[0, self.dom_size[1] - 1]
        )

        self.goal_b.create(
            x_range=[0, self.dom_size[0] - 1],
            exclude_from_x_range=[self.goal_a.state[0]], # don't place goal b on top of goal a
            exclude_from_y_range=[self.goal_a.state[1]],
            y_range=[0, self.dom_size[1] - 1]
        )

        self.goal_c.create(
            x_range=[0, self.dom_size[0] - 1],
            exclude_from_x_range=[self.goal_a.state[0], self.goal_b.state[0]], # don't place goal b on top of goal a
            exclude_from_y_range=[self.goal_a.state[1], self.goal_b.state[1]],
            y_range=[0, self.dom_size[1] - 1]
        )

    def make_obstacles(self):
        mask = np.array([self.agent.state, self.goal_a.state, self.goal_b.state, self.goal_c.state])

        self.obstacles.create_n_rand_obs(
            mask=mask,
            size_max=1,
            num_obs=10
            )


    # save proposition state of the environment
    # move objects
    # update proposition state based on object states
    def step(self, action_name):
        self.env.step(action_name)

        return self.env

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        if self.viewer == None:
            self.viewer = Viewer(mode=mode)

        return self.viewer.render(self.env)

    def render_rrt(self, path, mode='human'):
        if self.viewer == None:
            self.viewer = Viewer(mode=mode)

        return self.viewer.render_rrt(self.env, path)