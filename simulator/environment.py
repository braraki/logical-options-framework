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
    """Env
    Author: Brandon Araki
    A base class defining an environment MDP with a
    discrete 2D state space, a discrete action space,
    and a transition function.
    """

    def __init__(self, name, dom_size, action_dict):
        """ Initialize the environment.
        Args:
            name (str): name of the env
            dom_size (list): [x, y] dimensions of the env
            action_dict (dict): dictionary relating {action_name: action_num}
        """
        self.name = name
        self.dom_size = dom_size
        self.obj_state = np.zeros([*dom_size, 0])
        self.color_array = np.zeros([0, 3])
        self.objects = []
        self.obj_dict = None
        self.obj_idx_dict = {}
        self.props = []
        self.prop_dict = None
        self.prop_idx_dict = {}

        self.P = None

        self.viewer = None

        self.action_dict = action_dict

    def get_state(self):
        """
        Return the state of the environment by appending
        the states of every object and proposition.
        """
        state = []
        for obj in self.objects:
            # don't include obstacles
            if type(obj).mro()[1].__name__ != 'StaticObj':
                state.extend(list(obj.get_state()))
        for prop in self.props:
            state.append(int(prop.value))
        return state


    def get_full_state_space(self):
        """
        Return the state space of the environment by appending
        the state spaces of the objects and propositions.
        Example:
             obj1  obj2  p1 p2
            [8, 6, 8, 6, 2,  2]
             x, y, x, y, p,  p
        """
        state_space = []
        for obj in self.objects:
            # don't include obstacles
            if type(obj).mro()[1].__name__ != 'StaticObj':
                state_space.extend(obj.state_space)
        for prop in self.props:
            state_space.append(2)
        return state_space

    def state_to_idx(self, state):
        """
        Convert a state to an index
        Args:
            state (list)
        """
        state_space = self.get_full_state_space()
        return np.ravel_multi_index(tuple(state), tuple(state_space))

    def idx_to_state(self, idx):
        """
        Convert an index to a state
        Args:
            idx (int)
        """
        state_space = self.get_full_state_space()
        return np.unravel_index(idx, tuple(state_space))

    
    def set_state(self, state):
        """
        Given a state such as [3, 2, 4, 8, 1, 0, 0, 0],
        set object and prop states to match.
        Args:
            state (list)
        """
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

    def make_prop_map(self):
        """
        Matrix/function that stores this info:
        given a prop, returns in which states it is true
        given a state, returns which props are true
        """
        # find the size of the state space
        full_state_space = self.get_full_state_space()
        ss_size = np.prod(full_state_space)

        # include empty prop
        nP = len(self.props) + 1

        # define a prop map
        P = np.zeros((nP, ss_size))
        # iterate through every state
        idx = 0
        # ndindex iterates through every state in (x, y, ...)
        for state in np.ndindex(*full_state_space):
            state = list(state)
            s_idx = self.state_to_idx(state)
            self.set_state(state)

            prop_state = []
            for prop in self.props:
                try:
                    prop_state.append(int(prop.value))
                except:
                    print(prop.value)
                if type(prop).__name__ == 'CombinedProp' and prop.value:
                    prop_state[prop.prop_idxs[0]] = 0
                    prop_state[prop.prop_idxs[1]] = 0
                    
            if 1 not in prop_state:
                prop_state.append(1)
            else:
                prop_state.append(0)

            # this is hack for the driving env; if the 'goal c' prop and left lane prop
            # are both true, make it so that only the goal c prop is true
            if prop_state[2] == 1 and type(self).__name__ == 'DriveWorldEnv':
                prop_state[3] = 0

            P[:, s_idx] = prop_state

            idx += 1
            if idx % 10000 == 0:
                print(idx)

        self.P = P

        return P

    def get_proposition(self):
        """
        Returns currently active propositions in the form
        of a vector of 0/1s
        """
        prop_state = []
        for prop in self.props:
            try:
                prop_state.append(int(prop.value))
            except:
                print(prop.value)
            if type(prop).__name__ == 'CombinedProp' and prop.value:
                prop_state[prop.prop_idxs[0]] = 0
                prop_state[prop.prop_idxs[1]] = 0
                
        if 1 not in prop_state:
            prop_state.append(1)
        else:
            prop_state.append(0)

        return np.argmax(prop_state)

    def make_transition_function(self, plot=False):
        """
        Creates a transition function T. T[a][s, s'] = 1
        if action a causes a transition from s to s', and
        = 0 if not.
        """
        initial_state = self.get_state()

        # find the size of the state space
        full_state_space = self.get_full_state_space()
        ss_size = np.prod(full_state_space)

        # define a transition matrix
        T = [dok_matrix((ss_size, ss_size)) for a in self.action_dict]
        # iterate through every state
        idx = 0
        # ndindex iterates through every state in (x, y, ...)
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

                # if action_name == 'grip' and state[0] == state[1]:
                #     if state[2] == 4:
                #         print('hi')
                T[action][s_idx, ns_idx] = 1
            idx += 1
            if idx % 10000 == 0:
                print(idx)
            if plot:
                self.render(mode='fast')

        T = [t.tocsr() for t in T]

        self.set_state(initial_state)

        return T

    def make_reward_function(self):
        raise NotImplementedError()

    def step(self, action_name):
        """
        Steps the environment forward by applying the action
        first to the objects, then to the propositions, and then
        updating their values in the environment.
        """
        if isinstance(action_name, list):
            action = tuple(action_name)
        elif isinstance(action_name, tuple):
            action = action_name
        else:
            action = self.action_dict[action_name]

        # 1. the objects step
        for obj in self.objects:
            obj.step(self, action)

        # 2. update the props
        for prop in self.props:
            prop.eval(self.obj_dict)
        
        # 3. update obj_state
        self.update_obj_state()                
                    
    def add_props(self, prop_dict):
        """
        Given an OrderedDict of propositions, add each prop to the env.
        Args:
            prop_dict (OrderedDict): {prop_name: prop}
        """
        self.prop_dict = prop_dict
        for prop in prop_dict.values():
            self.add_prop(prop)

    def add_prop(self, prop):
        """
        Add a proposition to the env by adding it to the prop list
        and also updating a dictionary that relates prop_name to the
        index of the proposition in the list.
        Args:
            prop (proposition.*): an instantiation of one of the proposition
                classes from simulator.proposition.
        """
        self.props.append(prop)
        self.prop_idx_dict[prop.name] = len(self.props) - 1

    def add_objects(self, obj_dict):
        """
        Given an OrderedDict of objects, add each object to the env.
        Args:
            obj_dict (OrderedDict): {obj_name: obj}
        """
        self.obj_dict = obj_dict
        for obj in obj_dict.values():
            self.add_object(obj)

    def add_object(self, obj):
        """
        Add an object to the env by adding it to the object list
        and by adding a dimension for the object to the object state
        Args:
            obj (object.*): an instantiation of one of the object classes
                from simulator.object.
        """
        self.objects.append(obj)
        self.obj_idx_dict[obj.name] = len(self.objects) - 1
        self.color_array = np.append(self.color_array, obj.color[None], axis=0)
        self.add_obj_state(obj)

    def add_obj_state(self, obj):
        """
        Add object to self.obj_state.
        Args:
            obj (object.*): an instantiation of one of the object classes
                from simulator.object.
        """
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

    def update_obj_state(self):
        """
        Update objects in self.obj_state.
        """
        for i, obj in enumerate(self.objects):
            # if StaticObj, set obj_state to match the StaticObj's state
            if type(obj).mro()[1].__name__ == 'StaticObj':
                self.obj_state[..., i] = obj.state
            # else, wipe obj_state (set it to 0) and set the obj's position to 1
            else:
                self.obj_state[..., i] = 0
                state = obj.get_state()
                self.obj_state[state[0], state[1], i] = 1

    def render(self, mode='human'):
        if self.viewer == None:
            self.viewer = Viewer(mode=mode)

        return self.viewer.render(self)

class GridWorldEnv(Env):
    """
    Extends the Env class to describe a 2D gridworld environment.
    It mainly assumes that the environment has an 'agent' object
    whose state is describe by [x, y] coordinates.
    """

    def __init__(self, name, dom_size, action_dict):
        super().__init__(name, dom_size, action_dict)

    def make_reward_function(self):
        initial_state = self.get_state()

        # find the size of the state space
        full_state_space = self.get_full_state_space()
        ss_size = np.prod(full_state_space)

        # define a reward function (aka a vector storing 
        # reward for each state)
        R = np.zeros((ss_size,))
        # iterate through every state
        idx = 0
        # ndindex iterates through every state in (x, y, ...)
        for state in np.ndindex(*full_state_space):
            state = list(state)
            s_idx = self.state_to_idx(state)
            self.set_state(state)

            # the goal condition
            if self.prop_dict['ona'].value or self.prop_dict['onb'].value or self.prop_dict['onc'].value:
                R[s_idx] = 10
            elif self.prop_dict['onobstacle'].value:
                R[s_idx] = -1000
            idx += 1
            if idx % 10000 == 0:
                print(idx)

        self.set_state(initial_state)

        return R

    def get_state(self):
        state = self.obj_dict['agent'].get_state()

        return state

    def set_state(self, state):
        self.obj_dict['agent'].set_state(state[0:2])
        # self.obj_dict['goal_a'].set_state(state[1])
        # self.obj_dict['goal_b'].set_state(state[2])
        self.update_obj_state()
        for prop in self.props:
            prop.eval(self.obj_dict)

    def get_full_state_space(self):
        state_space = self.dom_size
        return state_space