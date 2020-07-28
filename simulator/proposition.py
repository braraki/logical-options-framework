import numpy as np

class Proposition(object):

    def __init__(self, name):
        self.name = name
        self.value = None

    # returns True or False based on the obj_state of the environment
    # eval cannot depend on the values of other propositions or on
    # information from previous timesteps
    def eval(self, obj_dict):
        raise NotImplementedError

# this prop is kind of an exception to how I've defined props
# because it's value does not depend on the state but it rather
# random at every time step. However, this could be accounted for by
# adding an extra dimension to the state space and attaching this
# prop's value to it
class RandomProp(Proposition):

    def __init__(self, name, prob):
        super().__init__(name)
        self.prob = prob

    def eval(self, obj_dict):
        self.value = np.random.uniform() < self.prob

        return self.value

# this is like an AND statement over two props
# if this is true, then when you return the overall prop state,
# list the children props as 0 and this prop as 1
class CombinedProp(Proposition):

    def __init__(self, name, prop1, prop2, prop_idxs):
        super().__init__(name)
        self.prop1 = prop1
        self.prop2 = prop2
        self.prop_idxs = prop_idxs

    def eval(self, obj_dict):
        self.value = (self.prop1.value and self.prop2.value)

        return self.value

# true if ball_a_location == basket_location
# neither obj can be a StaticObj
class SameLocationProp(Proposition):

    def __init__(self, name, obj_name1, obj_name2):
        super().__init__(name)

        self.obj1 = obj_name1
        self.obj2 = obj_name2

    def eval(self, obj_dict):
        obj_state1 = obj_dict[self.obj1].state
        obj_state2 = obj_dict[self.obj2].state

        self.value = (obj_state1 == obj_state2)

        return self.value

# when the car decides to trigger a maneuver
# true if state[2] == the number associated w/ the maneuver
class ManeuverProp(Proposition):

    def __init__(self, name, obj_name, option_number):
        super().__init__(name)

        self.obj = obj_name # typically is the agent
        self.option_number = option_number

    def eval(self, obj_dict):
        obj_state = obj_dict[self.obj].state

        # THIS IS A BIG HACK AND I NEED TO FIX IT!
        self.value = (obj_state[2] == self.option_number) and (obj_state[:2] == [4, 0])

        if self.value is not True and self.value is not False:
            print("f")

        return self.value

class OnObstacleProp(Proposition):

    def __init__(self, name, obj_name, obstacle_name):
        super().__init__(name)

        self.obj = obj_name
        self.obstacle = obstacle_name

    def eval(self, obj_dict):


        obj_state = obj_dict[self.obj].state
        obstacle_state = obj_dict[self.obstacle].state

        self.value = (obstacle_state[obj_state[0], obj_state[1]].item() == 1)

        if self.value is not False and self.value is not True:
            print('f')

        return self.value

class HoldingBallProp(Proposition):

    # the holder must have a gripper state
    def __init__(self, name, ball_name):
        self.name = name
        self.value = False
        # names, not the actual objects
        self.ball = ball_name

    # the ball's state[2] is the prop basically
    def eval(self, obj_dict):
        ball = obj_dict[self.ball]
        self.value = bool(ball.state[2])

        return self.value

