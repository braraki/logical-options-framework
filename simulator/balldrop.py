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

class BallDropSim(Sim):

    def __init__(self, dom_size=[8, 6], n_domains=100, n_traj=2):
        self.dom_size = dom_size
        self.viewer = None
        self.camera = None
        self.env = None # Env is defined in reset()
        # self.env = Env(name='BallDropEnv', dom_size=dom_size)
        self.obj_dict = OrderedDict()
        self.action_dict = {
            'nothing': 0, 'left': 1, 'right': 2, 'grip': 3, 'drop': 4
        }

    def reset(self):
        self.obj_dict.clear()
        self.env = BallDropEnv(
            name='BallDropEnv',
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
        self.make_balls()


        self.make_basket([self.ball_a, self.ball_b])
        # these obstacles have to be made AFTER
        # making the balls
        self.make_obstacle_under_ball(self.ball_a)
        self.make_obstacle_under_ball(self.ball_b)

    def add_props(self):
        prop_dict = OrderedDict()

        # "Ball A in Basket"
        prop_dict['ainb'] = SameLocationProp(
            name='ainb',
            obj_name1='ball_a',
            obj_name2='basket'
            )

        # "Ball B in Basket"
        prop_dict['binb'] = SameLocationProp(
            name='binb',
            obj_name1='ball_b',
            obj_name2='basket'
            )

        # "Ball A and Ball B in Basket"
        prop_dict['abinb'] = CombinedProp(
            name='abinb',
            prop1=prop_dict['ainb'],
            prop2=prop_dict['binb'],
            prop_idxs=[0,1]
        )

        # "Holding Ball A"
        prop_dict['hba'] = HoldingBallProp(
            name='hba',
            ball_name='ball_a'
        )

        # "Holding Ball B"
        prop_dict['hbb'] = HoldingBallProp(
            name='hbb',
            ball_name='ball_b'
        )

        return prop_dict

    # create and initialize objects
    # and add them to the env
    def add_objects(self):
        obj_dict = OrderedDict()

        # ball is never in the top row of the state space
        # so that row is reserved for indicating if the ball
        # is being held or not
        # (I decided to not add a 3rd dimension because it's too
        # space inefficient)
        ball_state_space = [self.dom_size[0], self.dom_size[1]]

        obj_dict['agent'] = AgentObj(name='agent', color=[1, 1, 0],
            state_space=[self.dom_size[0]])
        obj_dict['ball_a'] = BallObj(name='ball_a', color=[1, 0, 0], 
            state_space=ball_state_space)
        obj_dict['ball_b'] = BallObj(name='ball_b', color=[0, 1, 0], 
            state_space=ball_state_space)
        obj_dict['basket'] = BasketObj(name='basket', color=[0, 0, 1], 
            state_space=[self.dom_size[0]])
        obj_dict['obstacles'] = ObstacleObj(name='obstacles', color=[0, 0, 0], dom_size=self.dom_size)

        return obj_dict

    def make_obstacles(self):
        self.obstacles.add_random_obstacle(
            x_range=[1, self.dom_size[0] - 1],
            y_range=[1, self.dom_size[1]- 3]
            )

    def make_obstacle_under_ball(self, ball):
        x = ball.state[0]
        self.obstacles.add_random_obstacle(
            x_range=x,
            y_range=self.dom_size[1]- 3
            )

    def make_agent(self):
        self.agent.create(
            x_range=[1, self.dom_size[0] - 1],
            y_range=self.dom_size[1] - 1
        )

    def make_balls(self):
        ball_a_offset_from_top = 1 # the agent is at the very top, right above the ball?
        ball_b_offset_from_top = 1

        self.ball_a.create(
            x_range=[1, self.dom_size[0] - 1],
            y_range=self.dom_size[1] - 1 - ball_a_offset_from_top
        )
        self.ball_b.create(
            x_range=[1, self.dom_size[0] - 1],
            y_range=self.dom_size[1] - 1 - ball_b_offset_from_top,
            exclude_from_x_range=[self.ball_a.state[0]] # don't place ball b on top of ball a
        )

    def make_basket(self, balls):
        basket_y = 0 # y location of the basket (bottom of the env)

        self.basket.create(
            x_range=[1, self.dom_size[0] - 1],
            y_range=basket_y,
            exclude_from_x_range=[ball.state[0] for ball in balls]
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