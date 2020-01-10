"""1D arm picks up and drops balls into a basket"""
import numpy as np
from numpy.random import randint
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .simulator import Sim
from .object import *
from .proposition import *

class Env(object):

    def __init__(self, name, dom_size):
        self.name = name
        self.dom_size = dom_size
        self.obj_state = np.zeros([*dom_size, 0])
        self.color_array = np.zeros([0, 3])
        self.objects = [] # may not be necessary
        self.obj_dict = None
        self.obj_idx_dict = {} # may not be necessary
        self.props = [] # may not be necessary
        self.prop_dict = None
        self.prop_idx_dict = {} # may not be necessary

    def add_props(self, prop_dict):
        self.prop_dict = prop_dict
        for prop in prop_dict.values():
            self.add_prop(prop)

    def add_prop(self, prop):
        self.props.append(prop)
        self.prop_idx_dict[prop.name] = len(self.props) - 1

    # given an OrderedDict of objects, add each object to the env
    def add_objects(self, obj_dict):
        self.obj_dict = obj_dict
        for obj in obj_dict.values():
            self.add_object(obj)

    # add an object to the env by adding it to the object list
    # and by adding a dimension for the object to the object state
    def add_object(self, obj):
        self.objects.append(obj)
        self.obj_idx_dict[obj.name] = len(self.objects) - 1
        self.color_array = np.append(self.color_array, obj.color[None], axis=0)
        self.add_obj_state(obj)

    # add object to obj_state
    def add_obj_state(self, obj):
        # if the object is a StaticObj, its state is a 2D array
        # of the entire domain already
        if type(obj).mro()[1].__name__ == 'StaticObj':
            new_obj_array = obj.state[...,None] # need to add extra singleton dim to state
        # if the object is not a StaticObj, its state needs to be
        # converted into the representation of a 2D array
        else:
            new_obj_array = np.zeros([*self.dom_size, 1])
            new_obj_array[obj.state[0], obj.state[1], 0] = 1
        self.obj_state = np.append(self.obj_state, new_obj_array, axis=-1)

    # update object in obj_state
    def update_obj_state(self):
        for i, obj in enumerate(self.objects):
            # if StaticObj, set obj_state to match the StaticObj's state
            if type(obj).mro()[1].__name__ == 'StaticObj':
                self.obj_state[..., i] = obj.state
            # else, wipe obj_state (set it to 0) and set the obj's position to 1
            else:
                self.obj_state[..., i] = 0
                self.obj_state[obj.state[0], obj.state[1], i] = 1


class BallDropSim(Sim):

    def __init__(self, dom_size=[8, 6], n_domains=100, n_traj=2):
        self.dom_size = dom_size
        self.viewer = None
        self.env = None
        # self.env = Env(name='BallDropEnv', dom_size=dom_size)
        self.obj_dict = OrderedDict()

    def reset(self):
        self.obj_dict.clear()
        self.env = Env(name='BallDropEnv', dom_size=self.dom_size)

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
        self.make_balls()
        self.make_basket()
        self.make_obstacles()

    def add_props(self):
        prop_dict = OrderedDict()

        # "Ball A in Basket"
        prop_dict['bainb'] = SameLocationProp(
            name='bainb',
            obj_name1='ball_a',
            obj_name2='basket'
            )

        return prop_dict

    # create and initialize objects
    # and add them to the env
    def add_objects(self):
        obj_dict = OrderedDict()

        obj_dict['ball_a'] = BallObj(name='ball_a', color=[1, 0, 0])
        obj_dict['ball_b'] = BallObj(name='ball_b', color=[0, 1, 0])
        obj_dict['basket'] = BasketObj(name='basket', color=[0, 0, 1])
        obj_dict['obstacles'] = ObstacleObj(name='obstacles', color=[0, 0, 0], dom_size=self.dom_size)

        return obj_dict

    def make_obstacles(self):
        self.obstacles.add_random_obstacle(
            x_range=[1, self.dom_size[0] - 1],
            y_range=[1, self.dom_size[0]- 3]
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

    def make_basket(self):
        basket_y = 0 # y location of the basket (bottom of the env)

        self.basket.create(
            x_range=[1, self.dom_size[0] - 1],
            y_range=basket_y
            )

    # save proposition state of the environment
    # move objects
    # update proposition state based on object states
    def step(self, action):
        # 1. update the props
        for prop in self.prop_dict.values():
            prop.eval(self.obj_dict)

        # 2. the agent steps

        # 3. the objects step
        for obj in self.obj_dict.values():
            obj.step(self.env)
        
        # 4. update obj_state
        self.env.update_obj_state()

        return self.env

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        if mode == 'human':
            fig, ax = plt.subplots()

            # get non-whitespace in the domain
            nonwhitespace = (np.sum(self.env.obj_state, -1) > 0).astype(float)
            # make a map of the whitespace
            whitespace = np.ones(self.dom_size) - nonwhitespace
            # add whitespace as an extra layer to env.obj_state
            obj_state = np.append(self.env.obj_state, whitespace[...,None], axis=-1)
            # add whitespace color as an extra layer to env.color_array
            white = np.array([[1, 1, 1]])
            color_array = np.append(self.env.color_array, white, axis=0)

            # multiplying the domain, (X x Y x O) with (O x 3)
            # to get an RGB array (X x Y x 3)
            image = np.matmul(obj_state, color_array)
            # flip the y axis to graph it naturally
            image = np.flip(image, axis=1)
            # transpose x and y coordinates to graph naturally
            image = np.transpose(image, [1, 0, 2])

            implot = plt.imshow(image)

            # make string of proposition names
            textstr = ''
            for prop in self.prop_dict.values():
                textstr += prop.name + ' = ' + str(prop.value) + '\n'
            text_properties = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=text_properties)

            legend_keys = []
            for obj in self.obj_dict.values():
                legend_keys.append(mpatches.Patch(color=obj.color, label=obj.name))
            plt.legend(
                loc='upper right', 
                shadow=False,
                handles=legend_keys,
                bbox_to_anchor=(1.3, 1.)
                )
                

            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close(fig)


        # from simulator.rendering import Viewer

        # if self.viewer is None:
        #     self.viewer = Viewer(500, 500)
