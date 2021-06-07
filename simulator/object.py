import numpy as np
from numpy.random import randint
from collections import OrderedDict
import random

"""
Author: Brandon Araki

This file contains classes that describe "objects" in the environment.
Objects have a state space, a state, and update rules.
"""

# describe the Agent as an object
class AgentObj(object):
    """AgentObj
    Describe the agent as an object. 
    """

    def __init__(self, name, color, state_space):
        """
        Initialize the agent.
        Args:
            name (str): name of the agent object (e.g., 'agent')
            color (list): [r, g, b] array describing the color of the agent for plotting, 0-255
            state_space (list): [x, y, ...] describing the state space of the agent
        """
        self.name = name
        self.color = np.array(color)
        self.state = None # [x, y, h] --> h for if it's holding something
        # state space must be a list
        self.state_space = state_space

    def create(self, x_range, y_range, exclude_from_x_range=[], exclude_from_y_range=[]):
        def random_num(i_range, exclude_from_range):
            i = None
            if isinstance(i_range, int):
                # randint will return i if list is [i, i+1]
                i_range = [i_range, i_range+1]
            i = randint(*i_range)
            while i in exclude_from_range:
                i = randint(*i_range)
            return i

        x = random_num(x_range, exclude_from_x_range)
        y = random_num(y_range, exclude_from_y_range)
        h = 0

        state = [x, y, h]
        self.state = state
        return state

    def create_with_mask(self, x_range, y_range, mask=None):
        x = random.choice(x_range)
        y = random.choice(y_range)
        if mask is not None:
            while mask[x, y] == 1:
                x = random.choice(x_range)
                y = random.choice(y_range)

        h = 0

        state = [x, y, h]
        self.state = state
        return state

    def set_state(self, x):
        if type(x).__name__ == 'list':
            x = x[0]
        self.state[0] = x

    def get_state(self):
        return [self.state[0]]

    def step(self, env, action):
        self.apply_action(env, action)
        self.apply_dynamics(env)

    def apply_action(self, env, action):
        if action == 0: # do nothing
            pass
        elif action == 1: # move left
            if self.state[0] > 0:
                self.state[0] = self.state[0] - 1
        elif action == 2: # move right
            if self.state[0] < env.dom_size[0] - 1:
                self.state[0] = self.state[0] + 1
        elif action == 3: # grip
            pass
        elif action == 4: # drop
            pass
        else:
            raise ValueError('action {} is not a valid action'.format(action))

    def apply_dynamics(self, env):
        return

class GridAgentObj(AgentObj):
    """GridAgentObj
    Describes an agent object for gridworld environments.
    In particular, the state is [x, y] and the actions are
    'nothing', 'left', 'right', 'up', and 'down'.
    """

    def __init__(self, name, color, state_space):
        super().__init__(name, color, state_space)

    def set_state(self, x):
        self.state[:2] = x

    def get_state(self):
        return self.state[:2]

    def apply_action(self, env, action):
        if action == 0: # do nothing
            pass
        # note: the agent cannot wrap around the environment
        elif action == 1: # move left
            if self.state[0] > 0:
                self.state[0] = self.state[0] - 1
            else:
                self.state[0] = self.state[0] # env.dom_size[0] - 1
        elif action == 2: # move right
            if self.state[0] < env.dom_size[0] - 1:
                self.state[0] = self.state[0] + 1
            else:
                self.state[0] = self.state[0] # 0
        elif action == 3: # up
            if self.state[1] > 0:
                self.state[1] = self.state[1] - 1
            else:
                self.state[1] = self.state[1] # env.dom_size[0] - 1
        elif action == 4: # down
            if self.state[1] < env.dom_size[1] - 1:
                self.state[1] = self.state[1] + 1
            else:
                self.state[1] = self.state[1] # 0
        else:
            raise ValueError('action {} is not a valid action'.format(action))

class LocObj(object):
    """LocObj
    Describes a stationary object associated with a single state.
    """

    def __init__(self, name, color, state_space):
        self.name = name
        self.color = np.array(color) # [r, g, b] 0-255
        self.state = None # [x, y, h] --> h is whether or not it's being held
        self.state_space = state_space

    def __repr__(self):
        return self.name + ", " + type(self).__name__ + ", LocObj, " + hex(id(self))

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state


    def create(self, x_range, y_range, exclude_from_x_range=[], exclude_from_y_range=[]):
        '''
        x_range, y_range: int or [int, int] from which x and y will be randomly selected
        (optional) exclude_from_x/y_range: list of ints which will be passed over

        '''
        def random_num(i_range, exclude_from_range):
            i = None
            if isinstance(i_range, int):
                # randint will return i if list is [i, i+1]
                i_range = [i_range, i_range+1]
            i = randint(*i_range)
            while i in exclude_from_range:
                i = randint(*i_range)
            return i

        x = random_num(x_range, exclude_from_x_range)
        y = random_num(y_range, exclude_from_y_range)
        h = 0

        state = [x, y, h]
        self.state = state
        return state

    def create_with_mask(self, x_range, y_range, mask=None):
        x = random.choice(x_range)
        y = random.choice(y_range)
        if mask is not None:
            while mask[x, y] == 1:
                x = random.choice(x_range)
                y = random.choice(y_range)

        h = 0

        state = [x, y, h]
        self.state = state
        return state

    def step(self, env, action):
        self.apply_action(env, action)
        self.apply_dynamics(env)

    def apply_action(self, env, action):
        return NotImplementedError
    
    def apply_dynamics(self, env):
        return NotImplementedError

class GridGoalObj(LocObj):
    """GridGoalObj
    Describes LocObjs with [x, y] states.
    """

    def apply_action(self, env, action):
        return

    def apply_dynamics(self, env):
        return

    def set_state(self, x):
        self.state[:2] = x

    def get_state(self):
        return self.state[:2]

class StaticObj(object):
    """StaticObj
    Static objects are ones with trivial dynamics that don't change.
    Instead of being associated with a single state, for convenience they
    can be associated with many states in the state space. (e.g., obstacles)
    """

    def __init__(self, name, color, dom_size):
        self.name = name
        self.color = np.array(color) # [r, g, b] 0-255
        self.dom_size = dom_size
        self.state = np.zeros(self.dom_size)

    def __repr__(self):
        return self.name + ", " + type(self).__name__ + ", StaticObj, " + hex(id(self))

    def step(self, env, action):
        self.apply_action(env, action)
        self.apply_dynamics(env)

    def apply_action(self, env, action):
        return

    def apply_dynamics(self, env):
        return

class ObstacleObj(StaticObj):
    """ObstacleObj
    A StaticObj that represents obstacles in the environment.
    Has a number of helper functions for creating obstacles.
    """

    def __init__(self, name, color, dom_size, mask=None, size_max=None):
        super().__init__(name, color, dom_size)

        self.mask = None
        self.size_max = None

    def add_random_obstacle(self, x_range, y_range, exclude_from_x_range=[], exclude_from_y_range=[]):
        def random_num(i_range, exclude_from_range=[]):
            i = None
            if isinstance(i_range, int):
                # randint will return i if list is [i, i+1]
                i_range = [i_range, i_range+1]
            i = randint(*i_range)
            while i in exclude_from_range:
                i = randint(*i_range)
            return i

        x = random_num(x_range, exclude_from_x_range)
        y = random_num(y_range, exclude_from_y_range)

        self.state[x, y] = 1

    def add_obstacle_grid(self, obstacle_size=1, offset_from_edge=0):

        for i in range(obstacle_size):
            for j in range(obstacle_size):
                self.state[(i+offset_from_edge)::(obstacle_size+1),
                           (j+offset_from_edge)::(obstacle_size+1)] = 1

    def check_mask(self, dom=None):
        # Ensure goal is in free space
        if dom is not None:
            return np.any(dom[self.mask[:, 0], self.mask[:, 1]])
        else:
            return np.any(self.dom[self.mask[:, 0], self.mask[:, 1]])

    def insert_rect(self, x, y, height, width):
        # Insert a rectangular obstacle into map
        state_try = np.copy(self.state)
        state_try[x:x+height, y:y+width] = 1
        return state_try
    
    def add_rand_obs(self):
        # Add random (valid) obstacle to map
        rand_height = int(np.ceil(np.random.rand() * self.size_max))
        rand_width = int(np.ceil(np.random.rand() * self.size_max))
        randx = int(np.ceil(np.random.rand() * (self.dom_size[1]-1)))
        randy = int(np.ceil(np.random.rand() * (self.dom_size[1]-1)))
        state_try = self.insert_rect(randx, randy, rand_height, rand_width)
        if self.check_mask(state_try):
            return False
        else:
            self.state = state_try
            return True

    def create_n_rand_obs(self, mask=None, size_max=None, num_obs=1):
        # each row of the mask is a different goal
        if mask is None:
            self.mask = np.array([])
        else:
            if mask.ndim == 1:  #  handle the 1-goal case
                mask = mask[None, :]
            self.mask = mask

        self.size_max = size_max or np.max(self.dom_size) / 4

        # Add random (valid) obstacles to map
        count = 0
        for i in range(num_obs):
            if self.add_rand_obs():
                count += 1