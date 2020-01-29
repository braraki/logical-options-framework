import numpy as np

class PolicyBase(object):

    def get_action(self, env):
        raise NotImplementedError

class VIPolicy(PolicyBase):

    def make_policy(self, env):
        return
    
    def get_action(self, env):
        return

class HardCodedPolicy(PolicyBase):

    # def __init__(self, actions):
    #     self.actions = actions

    def get_action(self, env):
        ball_starting_height = env.dom_size[1] - 2

        agent_x = env.obj_dict['agent'].state[0]

        balla_x = env.obj_dict['ball_a'].state[0]
        balla_y = env.obj_dict['ball_a'].state[1]
        balla_held = env.prop_dict['hba'].value

        ballb_x = env.obj_dict['ball_b'].state[0]
        ballb_y = env.obj_dict['ball_b'].state[1]
        ballb_held = env.prop_dict['hbb'].value

        basket_x = env.obj_dict['basket'].state[0]

        # if ball a is NOT yet held, move the agent to ball a
        if not balla_held and balla_y == ball_starting_height:
            if balla_x < agent_x:
                return 'left'
            elif balla_x > agent_x:
                return 'right'
            elif balla_x == agent_x:
                return 'grip'
        # if ball a is held, move to be above the basket
        elif balla_held:
            if basket_x < agent_x:
                return 'left'
            elif basket_x > agent_x:
                return 'right'
            elif basket_x == agent_x:
                return 'drop'
        elif not balla_held and balla_y != ball_starting_height: #aka the ball has been dropped
            if not ballb_held and ballb_y == ball_starting_height:
                if ballb_x < agent_x:
                    return 'left'
                elif ballb_x > agent_x:
                    return 'right'
                elif ballb_x == agent_x:
                    return 'grip'
            elif ballb_held:
                if basket_x < agent_x:
                    return 'left'
                elif basket_x > agent_x:
                    return 'right'
                elif basket_x == agent_x:
                    return 'drop'

        return 'nothing'
