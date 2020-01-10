class Proposition(object):

    def __init__(self, name):
        self.name = name
        self.value = None

    # returns True or False based on the obj_state of the environment
    # eval cannot depend on the values of other propositions or on
    # information from previous timesteps
    def eval(self, obj_dict):
        raise NotImplementedError

# true if ball_a_location == basket_location
# neither obj can be a StaticObj
class SameLocationProp(Proposition):

    def __init__(self, name, obj_name1, obj_name2):
        self.name = name
        self.value = None

        self.obj1 = obj_name1
        self.obj2 = obj_name2

    def eval(self, obj_dict):
        obj_state1 = obj_dict[self.obj1].state
        obj_state2 = obj_dict[self.obj2].state

        self.value = (obj_state1 == obj_state2)

        return self.value