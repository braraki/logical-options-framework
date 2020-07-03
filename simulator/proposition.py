class Proposition(object):

    def __init__(self, name):
        self.name = name
        self.value = None

    # returns True or False based on the obj_state of the environment
    # eval cannot depend on the values of other propositions or on
    # information from previous timesteps
    def eval(self, obj_dict):
        raise NotImplementedError

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

class OnObstacleProp(Proposition):

    def __init__(self, name, obj_name, obstacle_name):
        super().__init__(name)

        self.obj = obj_name
        self.obstacle = obstacle_name

    def eval(self, obj_dict):


        obj_state = obj_dict[self.obj].state
        obstacle_state = obj_dict[self.obstacle].state

        self.value = (obstacle_state[obj_state[0], obj_state[1]] == 1)

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

