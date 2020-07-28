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
from .gridworld import GridWorldSim
from .rendering import Viewer
from .object import *
from .proposition import *
from .environment import *

class DeliverySim(GridWorldSim):

    def __init__(self, dom_size=[15, 15], n_domains=100, n_traj=2):
        super().__init__(dom_size, n_domains, n_traj)

    def reset(self):
        self.obj_dict.clear()
        self.env =GridWorldEnv(
            name='DeliveryEnv',
            dom_size=self.dom_size,
            action_dict=self.action_dict
        )

        self.obj_dict = self.add_objects()
        self.init_objects()
        self.env.add_objects(self.obj_dict)

        self.prop_dict = self.add_props()
        self.env.add_props(self.prop_dict)

    def init_objects(self):
        # making obstacles has to come first so
        # the agent and goals can be placed around
        # the obstacles    
        self.make_obstacles()
        self.make_agent()
        self.make_goals()
        print('f')

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

        # "Agent on Goal H"
        prop_dict['onh'] = SameLocationProp(
            name='onh',
            obj_name1='agent',
            obj_name2='goal_h'
        )

        # "Delivery B Canceled"
        prop_dict['canceled'] = RandomProp(
            name='canceled',
            prob=0.05
        )

        # "Goal A and Delivery B Canceled"
        prop_dict['a_and_canceled'] = CombinedProp(
            name='a_and_canceled',
            prop1=prop_dict['ona'],
            prop2=prop_dict['canceled'],
            prop_idxs=[0,3]
        )

        # "Goal B and Delivery B Canceled"
        prop_dict['b_and_canceled'] = CombinedProp(
            name='b_and_canceled',
            prop1=prop_dict['onb'],
            prop2=prop_dict['canceled'],
            prop_idxs=[1,3]
        )

        # "Goal H and Delivery B Canceled"
        prop_dict['h_and_canceled'] = CombinedProp(
            name='h_and_canceled',
            prop1=prop_dict['onh'],
            prop2=prop_dict['canceled'],
            prop_idxs=[2,3]
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
        obj_dict['goal_h'] = GridGoalObj(name='h', color=[0, 0, 1], 
            state_space=self.dom_size)
        obj_dict['obstacles'] = ObstacleObj(name='obstacles', color=[0, 0, 0],
            dom_size=self.dom_size)
        return obj_dict

    def make_agent(self):
        # this will create the agent on either the
        # left or right side of the gridworld
        # at a random height
        mask = np.copy(self.obstacles.state)
        self.agent.create_with_mask(
            x_range=[0, self.dom_size[1]-1],
            y_range=[i for i in range(self.dom_size[1])],
            mask=mask
        )

    def make_goals(self):
        mask = np.copy(self.obstacles.state)
        mask[self.agent.state[0], self.agent.state[1]] = 1
        self.goal_a.create_with_mask(
            x_range=[i for i in range(self.dom_size[0])],
            y_range=[i for i in range(self.dom_size[1])],
            mask=mask
        )
        
        mask[self.goal_a.state[0], self.goal_a.state[1]] = 1
        self.goal_b.create_with_mask(
            x_range=[i for i in range(self.dom_size[0])],
            y_range=[i for i in range(self.dom_size[1])],
            mask=mask
        )

        # home will be placed at the same location
        # as the agent
        self.goal_h.create_with_mask(
            x_range=[self.agent.state[0]],
            y_range=[self.agent.state[1]]
        )

    def make_obstacles(self):
        self.obstacles.add_obstacle_grid(obstacle_size=3)

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