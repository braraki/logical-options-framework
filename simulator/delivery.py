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
    """DeliverySim
    Author: Brandon Araki
    Defines a discrete 2D gridworld environment that can be used for simulating
    delivery tasks. The environment is 15x15 with 3x3 blocks of 'buildings'
    arranged in a grid. There are four subgoals a, b, c, h; one event
    prop 'can' for indicating when a delivery is canceled; and one
    obstacle prop o.
    """

    def __init__(self, dom_size=[15, 15]):
        super().__init__(dom_size)

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
        """
        Create the obstacles, agent, and goals in the enviornment.
        Making obstacles has to come first so the agent and goals can
        be placed around the obstacles.
        """    
        self.make_obstacles()
        self.make_agent()
        self.make_goals()

    def add_props(self):
        """
        Create a dictionary of propositions with 
        subgoals a, b, c, h; event prop 'can';
        obstacle prop 'o', and combined props 'a&can',
        'b&can', 'c&can', 'h&can'.
        """

        prop_dict = OrderedDict()

        # "Agent on Goal A"
        prop_dict['ona'] = SameLocationProp(
            name='a',
            obj_name1='agent',
            obj_name2='goal_a'
            )

        # "Agent on Goal B"
        prop_dict['onb'] = SameLocationProp(
            name='b',
            obj_name1='agent',
            obj_name2='goal_b'
            )

        # "Agent on Goal C"
        prop_dict['onc'] = SameLocationProp(
            name='c',
            obj_name1='agent',
            obj_name2='goal_c'
            )

        # "Agent on Goal H"
        prop_dict['onh'] = SameLocationProp(
            name='h',
            obj_name1='agent',
            obj_name2='goal_h'
        )

        # "Delivery C Canceled"
        prop_dict['canceled'] = ExternalProp(
            name='can',
            value=False
        )

        # "Goal A and Delivery C Canceled"
        prop_dict['a_and_canceled'] = CombinedProp(
            name='ca',
            prop1=prop_dict['ona'],
            prop2=prop_dict['canceled'],
            prop_idxs=[0,4]
        )

        # "Goal B and Delivery C Canceled"
        prop_dict['b_and_canceled'] = CombinedProp(
            name='cb',
            prop1=prop_dict['onb'],
            prop2=prop_dict['canceled'],
            prop_idxs=[1,4]
        )

        # "Goal C and Delivery C Canceled"
        prop_dict['c_and_canceled'] = CombinedProp(
            name='cc',
            prop1=prop_dict['onc'],
            prop2=prop_dict['canceled'],
            prop_idxs=[2,4]
        )

        # "Goal H and Delivery B Canceled"
        prop_dict['h_and_canceled'] = CombinedProp(
            name='ch',
            prop1=prop_dict['onh'],
            prop2=prop_dict['canceled'],
            prop_idxs=[3,4]
        )

        # "Agent on Obstacle"
        prop_dict['onobstacle'] = OnObstacleProp(
            name='o',
            obj_name='agent',
            obstacle_name='obstacles'
        )

        return prop_dict

    def add_objects(self):
        """
        Create and initialize objects and add them
        to the environment.
        """
        obj_dict = OrderedDict()

        obj_dict['agent'] = GridAgentObj(name='agent', color=[.7, .7, .7],
            state_space=self.dom_size)
        obj_dict['goal_a'] = GridGoalObj(name='a', color=[1, 0, 0], 
            state_space=self.dom_size)
        obj_dict['goal_b'] = GridGoalObj(name='b', color=[0, 1, 0], 
            state_space=self.dom_size)
        obj_dict['goal_c'] = GridGoalObj(name='c', color=[0, 0, 1], 
            state_space=self.dom_size)
        obj_dict['goal_h'] = GridGoalObj(name='h', color=[0, 1, 1], 
            state_space=self.dom_size)
        obj_dict['obstacles'] = ObstacleObj(name='obstacles', color=[0, 0, 0],
            dom_size=self.dom_size)
        return obj_dict

    def make_agent(self):
        """
        Create the agent on either the left or right side of
        the gridworld at a random height.
        """
        mask = np.copy(self.obstacles.state)
        # self.agent.create_with_mask(
        #     x_range=[0, self.dom_size[1]-1],
        #     y_range=[i for i in range(self.dom_size[1])],
        #     mask=mask
        # )
        self.agent.create_with_mask(
            x_range=[7],
            y_range=[0],
            mask=mask
        )

    def make_goals(self):
        """
        Add subgoals a, b, c, h to the environment as objects.
        """
        mask = np.copy(self.obstacles.state)
        mask[self.agent.state[0], self.agent.state[1]] = 1
        self.goal_a.create_with_mask(
            x_range=[1],
            y_range=[7],
            mask=mask
        )
        
        mask[self.goal_a.state[0], self.goal_a.state[1]] = 1
        self.goal_b.create_with_mask(
            x_range=[11],
            y_range=[3],
            mask=mask
        )

        mask[self.goal_b.state[0], self.goal_b.state[1]] = 1
        self.goal_c.create_with_mask(
            x_range=[3],
            y_range=[13],
            mask=mask
        )

        # home will be placed at the same location
        # as the agent
        self.goal_h.create_with_mask(
            x_range=[self.agent.state[0]],
            y_range=[self.agent.state[1]]
        )

    def make_obstacles(self):
        """
        Add a grid of 3x3 obstacles to the environment
        """
        self.obstacles.add_obstacle_grid(obstacle_size=3)