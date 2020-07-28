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

    def __init__(self, dom_size=[6, 12], n_domains=100, n_traj=2):
        self.dom_size = dom_size
        self.viewer = None
        self.camera = None
        self.env = None # Env is defined in reset()
        self.obj_dict = OrderedDict()
        self.action_dict = {
            'nothing': 0, 'left': 1, 'right': 2, 'up': 3, 'overtake' : 4, 'straight' : 5, 'change': 6
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
        self.make_leftlane()

    def add_props(self):
        prop_dict = OrderedDict()

        # "Agent on Goal O(vertake)"
        prop_dict['ongo'] = SameLocationProp(
            name='ongo',
            obj_name1='agent',
            obj_name2='goal_o'
            )

        # "Agent on Goal S(traight)"
        prop_dict['ongs'] = SameLocationProp(
            name='ongs',
            obj_name1='agent',
            obj_name2='goal_s'
            )

        # "Agenet on Goal C(hange Lane)"
        prop_dict['ongc'] = SameLocationProp(
            name='ongc',
            obj_name1='agent',
            obj_name2='goal_c'
        )

        # # "Overtake Maneuver Chosen"
        # prop_dict['o'] = ManeuverProp(
        #     name='o',
        #     obj_name='agent',
        #     option_number=1
        # )

        # # "Straight Maneuver Chosen"
        # prop_dict['s'] = ManeuverProp(
        #     name='s',
        #     obj_name='agent',
        #     option_number=2
        # )

        # # "Change Maneuver Chosen"
        # prop_dict['c'] = ManeuverProp(
        #     name='c',
        #     obj_name='agent',
        #     option_number=3
        # )

        # "Agent in Left Lane"
        prop_dict['inleftlane'] = OnObstacleProp(
            name='inleftlane',
            obj_name='agent',
            obstacle_name='leftlane'
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

        obj_dict['agent'] = CarAgentObj(name='agent', color=[.9, .9, .0],
            state_space=self.dom_size)
        obj_dict['goal_o'] = GridGoalObj(name='o', color=[1, 0, 0], 
            state_space=self.dom_size)
        obj_dict['goal_s'] = GridGoalObj(name='s', color=[0, 1, 0], 
            state_space=self.dom_size)
        obj_dict['goal_c'] = GridGoalObj(name='c', color=[0, 1, 0], 
            state_space=self.dom_size)
        obj_dict['obstacles'] = ObstacleObj(name='obstacles', color=[0.1, 0.1, 0.1],
            dom_size=self.dom_size)
        obj_dict['leftlane'] = LeftLaneObj(name='leftlane', color=[0.7, 0.7, 0.7],
            dom_size=self.dom_size)

        return obj_dict

    def make_agent(self):
        twothirds = int(self.dom_size[0]*(2/3))
        self.agent.create(
            x_range=[twothirds, twothirds+1],
            y_range=[0, 1]
        )

    def make_goals(self):
        onethird = int(self.dom_size[0]/3) - 1
        twothirds = int(self.dom_size[0]*(2/3))
        self.goal_o.create(
            x_range=[twothirds, twothirds+1],
            y_range=[self.dom_size[1]-2, self.dom_size[1] - 1]
        )

        self.goal_s.create(
            x_range=[twothirds, twothirds+1],
            y_range=[self.dom_size[1]-4, self.dom_size[1] - 3]
        )

        self.goal_c.create(
            x_range=[onethird, onethird+1],
            y_range=[self.dom_size[1]-2, self.dom_size[1] - 1]
        )

    def make_obstacles(self):
        # mask = np.array([self.agent.state, self.goal_o.state, self.goal_s.state, self.goal_c.state])

        self.obstacles.add_right_lane_car()
        self.obstacles.add_left_lane_car()

    def make_leftlane(self):
        self.leftlane.add_leftlane()

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