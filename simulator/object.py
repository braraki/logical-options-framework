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

# describe the Agent as an object
class LineAgentObj(AgentObj):

    def __init__(self, name, color, state_space):
        super().__init__(name, color, state_space)

    def apply_action(self, env, action):
        if action == 0: # do nothing
            pass
        # note: I'm allowing the agent to wrap around the environment
        elif action == 1: # move left
            if self.state[0] > 0:
                self.state[0] = self.state[0] - 1
            else:
                self.state[0] = env.dom_size[0] - 1
        elif action == 2: # move right
            if self.state[0] < env.dom_size[0] - 1:
                self.state[0] = self.state[0] + 1
            else:
                self.state[0] = 0
        else:
            raise ValueError('action {} is not a valid action'.format(action))

class GridAgentObj(AgentObj):

    def __init__(self, name, color, state_space):
        super().__init__(name, color, state_space)

    def set_state(self, x):
        self.state[:2] = x

    def get_state(self):
        return self.state[:2]

    def apply_action(self, env, action):
        if action == 0: # do nothing
            pass
        # note: I'm allowing the agent to wrap around the environment
        # note: I've decided to not allow that
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

class LineGoalObj(LocObj):

    def apply_action(self, env, action):
        return

    def apply_dynamics(self, env):
        return

    def set_state(self, x):
        if type(x).__name__ == 'list':
            x = x[0]
        self.state[0] = x

    def get_state(self):
        return self.state

class GridGoalObj(LocObj):

    def apply_action(self, env, action):
        return

    def apply_dynamics(self, env):
        return

    def set_state(self, x):
        self.state[:2] = x

    def get_state(self):
        return self.state[:2]

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