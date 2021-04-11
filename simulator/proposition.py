import numpy as np

"""
Author: Brandon Araki
These classes define propositions. Propositions must have
a name and a value (true or false), as well as an 'eval'
function that evaluates true or false based on the
current state of the environment.
"""

class Proposition(object):

    def __init__(self, name):
        self.name = name
        self.value = None


    def eval(self, obj_dict):
        """
        Returns True or False based on the obj_state of the environment.
        Eval cannot depend on the values of other propositions or on
        information from previous timesteps.
        """
        raise NotImplementedError

class ExternalProp(Proposition):
    """ExternalProp
    This prop is kind of an exception to how I've defined props
    because it's value does not depend on the state but is rather
    random at every time step. However, this could be accounted for by
    adding an extra dimension to the state space and attaching this
    prop's value to it
    """

    def __init__(self, name, value):
        super().__init__(name)
        self.value = value

    def eval(self, obj_dict):
        return self.value


class CombinedProp(Proposition):
    """CombinedProp
    This is like an AND statement over two props
    if this is true, then when you return the overall prop state,
    list the children props as 0 and this prop as 1
    """
    def __init__(self, name, prop1, prop2, prop_idxs):
        super().__init__(name)
        self.prop1 = prop1
        self.prop2 = prop2
        self.prop_idxs = prop_idxs

    def eval(self, obj_dict):
        self.value = (self.prop1.value and self.prop2.value)

        return self.value


class SameLocationProp(Proposition):
    """SameLocationProp
    True if obj1_location == obj2_location.
    Neither obj can be a StaticObj
    """

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
    """
    True if the location of self.obj intersects
    with the locations of the obstacles.
    """

    def __init__(self, name, obj_name, obstacle_name):
        super().__init__(name)

        self.obj = obj_name
        self.obstacle = obstacle_name

    def eval(self, obj_dict):


        obj_state = obj_dict[self.obj].get_state()
        obstacle_state = obj_dict[self.obstacle].state

        self.value = (obstacle_state[obj_state[0], obj_state[1]].item() == 1)

        if self.value is not False and self.value is not True:
            print('f')

        return self.value
