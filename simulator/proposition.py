class Proposition(object):

    def __init__(self, name):
        self.name = name
        self.value = None

    # returns True or False based on the obj_state of the environment
    # eval cannot depend on the values of other propositions or on
    # information from previous timesteps
    def eval(self, obj_dict, prop_dict, action):
        raise NotImplementedError

# true if ball_a_location == basket_location
# neither obj can be a StaticObj
class SameLocationProp(Proposition):

    def __init__(self, name, obj_name1, obj_name2):
        self.name = name
        self.value = None

        self.obj1 = obj_name1
        self.obj2 = obj_name2

    def eval(self, obj_dict, prop_dict, action):
        obj_state1 = obj_dict[self.obj1].state
        obj_state2 = obj_dict[self.obj2].state

        self.value = (obj_state1 == obj_state2)

        return self.value

class HoldingBallProp(Proposition):

    # the holder must have a gripper state
    def __init__(self, name, holdee_name, holder_name='agent'):
        self.name = name
        self.value = False

        # names, not the actual objects
        self.holder = holder_name
        self.holdee = holdee_name

    # if the holder is directly above the holdee
    # and the action is 'grip' then set to true
    # if the gripper is open
    def eval(self, obj_dict, prop_dict, action):
        holder = obj_dict[self.holder]
        holdee = obj_dict[self.holdee]

        holding_other_ball = False
        for prop_name in prop_dict.keys():
            # check if the agent is holding the other ball
            if 'hb' in prop_name and prop_name != self.name:
                if prop_dict[prop_name].value:
                    holding_other_ball = True

        if action == 3: # grip
            # if the gripper is above the holdee
            if holder.state[1] == holdee.state[1] + 1 and holder.state[0] == holdee.state[0]:
                if not holding_other_ball:
                    self.value = True
        elif action == 4: # drop
            self.value = False
        # else, do not change value

        return self.value

