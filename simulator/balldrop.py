"""1D arm picks up and drops balls into a basket"""
import numpy as np
from np.random import randint

import simulator

# proposition with a location
class LocProp(object):

    def __init__(self, name, color):
        self.name = name
        self.color = color # [r, g, b] 0-255
        self.state = None

    '''
        x_range, y_range: int or [int, int] from which x and y will be randomly selected
        (optional) exclude_from_x/y_range: list of ints which will be passed over

    '''
    def init(x_range, y_range, exclude_from_x_range=[], exclude_from_y_range=[]):
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

    def step(state, action):
        raise NotImplementedError()

class BasketProp(LocProp):

    def step(state, action):
        return

class BallProp(LocProp):

    def step(state, action):
        # returns if there is an obstacle at that location
        if state.check_obstacle(self.state[0], self.state[1] - 1):

# static props are ones with trivial dynamics that don't change
# 
class StaticProp(object):


class ObstacleProp(LocProp):

    def 

class BallDropEnv(simulator.Sim):

    def __init__(self, dom_size=dom_size, n_domains=100, n_traj=2):
        self.viewer = None

    def init(self):
        balls = self.make_balls()
        basket = self.make_basket()

    def make_balls(self):
        ball_a_offset_from_top = 1 # the agent is at the very top, right above the ball?
        ball_b_offset_from_top = 1

        ball_a = BallProp(name='ball_a', color=[255, 0, 0])
        ball_a.init(x_range=[1, self.dom_size[0] - 1],
                    y_range=self.dom_size[1] - 1 - ball_a_offset_from_top,
                    )
        ball_b = BallProp(name='ball_b', color=[0, 255, 0])
        ball_b.init(x_range=[1, self.dom_size[0] - 1],
                    y_range=self.dom_size[1] - 1 - ball_b_offset_from_top,
                    exclude_from_x_range=[ball_a.state[0]] # don't place ball b on top of ball a
                    )

        return [ball_a, ball_b]

    def make_basket(self):
        basket_y = 0 # y location of the basket (bottom of the env)

        basket = BasketProp(name='basket', color=[0, 0, 255])
        basket.init(x_range=[1, self.dom_size[0] - 1],
                    y_range=basket_y)

        return [basket]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def random_init(self):


    def render(self, mode='human'):
        from simulator.rendering import Viewer

        if self.viewer is None:
            self.viewer = Viewer(500, 500)
