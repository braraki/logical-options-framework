import numpy as np
from numpy.random import randint
from collections import OrderedDict

# describe the Agent as an object
class AgentObj(object):

    def __init__(self, name, color, actions):
        self.name = name
        self.color = np.array(color)
        self.state = None

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

        state = [x, y]
        self.state = state
        return state

    #
    # 
    def step(self, env, action):
        if self.state[1] > 0:
            if not env.obj_dict['obstacles'].state[self.state[0], self.state[1] - 1]:        
                self.state[1] = self.state[1] - 1


# object with a location
class LocObj(object):

    def __init__(self, name, color):
        self.name = name
        self.color = np.array(color) # [r, g, b] 0-255
        self.state = None

    def __repr__(self):
        return self.name + ", " + type(self).__name__ + ", LocObj, " + hex(id(self))


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

        state = [x, y]
        self.state = state
        return state

    def step(self, env):
        raise NotImplementedError

class BasketObj(LocObj):

    def step(self, env):
        return

class BallObj(LocObj):

    # do not fall down if the ball is at the bottom of the env or
    # above an obstacle
    # (or if it is being held by the agent - need to add that)
    def step(self, env):
        if self.state[1] > 0:
            if not env.obj_dict['obstacles'].state[self.state[0], self.state[1] - 1]:        
                self.state[1] = self.state[1] - 1
        # # if the ball is not being held...
        # if not env.ball_held():
        #     # returns True if there is an obstacle at the specified location
        #     # check if there is not an obstacle beneath the ball. If so,
        #     # the ball falls
        #     beneath_ball = self.state[1] - 1
        #     if not env.check_obstacle(self.state[0], beneath_ball):
        #         self.state[1] = beneath_ball


# static objects are ones with trivial dynamics that don't change

class StaticObj(object):

    def __init__(self, name, color, dom_size):
        self.name = name
        self.color = np.array(color) # [r, g, b] 0-255
        self.dom_size = dom_size
        self.state = np.zeros(dom_size)

    def __repr__(self):
        return self.name + ", " + type(self).__name__ + ", StaticObj, " + hex(id(self))

    def step(self, env):
        return self.state


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