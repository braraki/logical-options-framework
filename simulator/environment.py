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

class Env(object):

    def __init__(self, name, dom_size, action_dict):
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

        self.viewer = None

        self.action_dict = action_dict

    def get_state(self):
        state = []
        for obj in self.objects:
            # don't include obstacles
            if type(obj).mro()[1].__name__ != 'StaticObj':
                state.extend(list(obj.get_state()))
        for prop in self.props:
            state.append(int(prop.value))
        return state

    # ie [8, 6, 8, 6, 2, 2, 2, 2]
    #     x, y, x, y, p, p, p, p
    def get_full_state_space(self):
        state_space = []
        for obj in self.objects:
            # don't include obstacles
            if type(obj).mro()[1].__name__ != 'StaticObj':
                state_space.extend(obj.state_space)
        for prop in self.props:
            state_space.append(2)
        return state_space

    # state of the form [3, 2, 4, 8, 1, 0, 0, 0]
    def state_to_idx(self, state):
        state_space = self.get_full_state_space()
        return np.ravel_multi_index(tuple(state), tuple(state_space))

    def idx_to_state(self, idx):
        state_space = self.get_full_state_space()
        return np.unravel_index(idx, tuple(state_space))

    # given state of form [3, 2, 4, 8, 1, 0, 0, 0]
    # set object and prop states to match
    def set_state(self, state):
        idx = 0 # index along the whole state
        for obj in self.objects:
            # don't include obstacles
            if type(obj).mro()[1].__name__ != 'StaticObj':
                ndim = len(obj.state_space)
                obj_state = state[idx : idx+ndim]
                obj.set_state(obj_state)
                idx += ndim
        for prop in self.props:
            prop.value = bool(state[idx])
            idx += 1
        self.update_obj_state()

    def make_transition_function(self, plot=False):
        # find the size of the state space
        full_state_space = self.get_full_state_space()
        ss_size = np.prod(full_state_space)

        # define a transition matrix
        T = [dok_matrix((ss_size, ss_size)) for a in self.action_dict]
        # iterate through every state
        idx = 0
        for state in np.ndindex(*full_state_space):
            state = list(state)
            s_idx = self.state_to_idx(state)
            # iterate through every action
            for action_name in self.action_dict:
                self.set_state(state)
                self.step(action_name)
                new_state = self.get_state()
                
                ns_idx = self.state_to_idx(new_state)

                action = self.action_dict[action_name]
                T[action][s_idx, ns_idx] = 1
            idx += 1
            if idx % 10000 == 0:
                print(idx)
            if plot:
                self.render(mode='fast')

        T = [t.tocsr() for t in T]

        return T

    def step(self, action_name):
        action = self.action_dict[action_name]

        # 1. update the props
        for prop in self.props:
            prop.eval(self.obj_dict, action)

        # 2. the objects step
        for obj in self.objects:
            obj.step(self, action)
        
        # 3. update obj_state
        self.update_obj_state()                
                    
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

    def render(self, mode='human'):
        if self.viewer == None:
            self.viewer = Viewer(mode=mode)

        return self.viewer.render(self)

class BallDropEnv(Env):

    def get_state(self):
        state = []
        for obj in self.objects:
            # don't include obstacles
            if type(obj).mro()[1].__name__ != 'StaticObj':
                if obj.name == 'ball_a':
                    # if ball a is in basket, set ball a's
                    # y value to be state_space[1]-1
                    # (aka dom_size[1])
                    state.append(obj.state[0])
                    if self.prop_dict['bainb'].value:
                        state.append(obj.state_space[1]-1)
                    # if ball a is beling held, set ball a's
                    # y value to be state_space[1]-2
                    # (aka dom_size[1]-1)
                    elif self.prop_dict['hba'].value:
                        state.append(obj.state_space[1]-2)
                    else:
                        state.append(obj.state[1])
                elif obj.name == 'ball_b':
                    state.append(obj.state[0])
                    if self.prop_dict['bbinb'].value:
                        state.append(obj.state_space[1]-1)
                    elif self.prop_dict['hbb'].value:
                        state.append(obj.state_space[1]-2)
                    else:
                        state.append(obj.state[1])
                else:
                    state.extend(list(obj.get_state()))

        return state

    # given state of form [3, 2, 4, 8, 1, 0, 0, 0]
    # set object and prop states to match
    def set_state(self, state):
        idx = 0 # index along the whole state
        for obj in self.objects:
            # don't include obstacles
            if type(obj).mro()[1].__name__ != 'StaticObj':
                ndim = len(obj.state_space)
                obj_state = state[idx : idx+ndim]
                if obj.name == 'ball_a':
                    # if ball a's y value is state_space[1]-1,
                    # then it is the basket (bainb is True)
                    if obj_state[1] == obj.state_space[1]-1:
                        obj_state[1] = 0
                        self.prop_dict['bainb'].value = True
                    else:
                        self.prop_dict['bainb'].value = False
                    # if ball a's y value is state_space[1]-2
                    # then it is being held (hba is True)
                    if obj_state[1] == obj.state_space[1]-2:
                        obj_state[1] = self.dom_size[1]-2
                        self.prop_dict['hba'].value = True
                    else:
                        self.prop_dict['hba'].value = False
                elif obj.name == 'ball_b':
                    if obj_state[1] == obj.state_space[1]-1:
                        obj_state[1] = 0
                        self.prop_dict['bbinb'].value = True
                    else:
                        self.prop_dict['bbinb'].value = False
                    if obj_state[1] == obj.state_space[1]-2:
                        obj_state[1] = self.dom_size[1]-2
                        self.prop_dict['hbb'].value = True
                    else:
                        self.prop_dict['hbb'].value = False
                obj.set_state(obj_state)
                idx += ndim
        self.update_obj_state()

    # ie [8, 6, 8, 6, 2, 2, 2, 2]
    #     x, y, x, y, p, p, p, p
    def get_full_state_space(self):
        state_space = []
        for obj in self.objects:
            # don't include obstacles
            if type(obj).mro()[1].__name__ != 'StaticObj':
                state_space.extend(obj.state_space)
        return state_space