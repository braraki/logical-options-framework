import numpy as np
from numpy.random import randint
from collections import OrderedDict

# describe the Agent as an object
class AgentObj(object):

    def __init__(self, name, color, state_space):
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

# object with a location
class LocObj(object):

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

    '''
        x_range, y_range: int or [int, int] from which x and y will be randomly selected
        (optional) exclude_from_x/y_range: list of ints which will be passed over

    '''
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

    def step(self, env, action):
        self.apply_action(env, action)
        self.apply_dynamics(env)

    def apply_action(self, env, action):
        return NotImplementedError
    
    def apply_dynamics(self, env):
        return NotImplementedError

class BasketObj(LocObj):

    def apply_action(self, env, action):
        return

    def apply_dynamics(self, env):
        return

    def set_state(self, x):
        if type(x).__name__ == 'list':
            x = x[0]
        self.state[0] = x

    def get_state(self):
        return [self.state[0]]

class BallObj(LocObj):

    def apply_action(self, env, action):
        being_held = self.state[2]
        
        # if the ball is being held, it should move with the agent
        if being_held:
            if action == 1: # move left
                if self.state[0] > 0:
                    self.state[0] = self.state[0] - 1
            elif action == 2: # move right
                if self.state[0] < env.dom_size[0] - 1:
                    self.state[0] = self.state[0] + 1
            elif action == 4: # drop
                env.obj_dict['agent'].state[2] = 0
                self.state[2] = 0
        else:
            if action == 3: # grip
                if env.obj_dict['agent'].state[0] == self.state[0]:
                    if env.obj_dict['agent'].state[1] == self.state[1] + 1:
                        if env.obj_dict['agent'].state[2] != 1:
                            env.obj_dict['agent'].state[2] = 1
                            self.state[2] = 1
    
    # do not fall down if the ball is at the bottom of the env or
    # above an obstacle
    def apply_dynamics(self, env):
        being_held = self.state[2]

        # if the ball is not being held and it's not on the ground
        # and if it's not above an obstacle, then it falls
        if not being_held:
            if self.state[1] > 0:
                if not env.obj_dict['obstacles'].state[self.state[0], self.state[1] - 1]:
                    self.state[1] = self.state[1] - 1

# static objects are ones with trivial dynamics that don't change
class StaticObj(object):

    def __init__(self, name, color, dom_size):
        self.name = name
        self.color = np.array(color) # [r, g, b] 0-255
        self.dom_size = dom_size
        self.state = np.zeros(dom_size)

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