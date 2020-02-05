import numpy as np
import scipy.sparse as sparse
from pathlib import Path

class PolicyBase(object):

    def load_reward_function(self, env):
        file_name = Path(__file__).parent.parent / 'storage' \
            / env.name / 'reward' / 'R.npz'
        R = np.load(file_name)['R']
        return R

    def load_transitions(self, env):
        T = []
        directory = Path(__file__).parent.parent / 'storage' / env.name / 'transitions'
        files = []
        for file in directory.iterdir():
            files.append(file)
        files = sorted(files)
        for file in files:
            T.append(sparse.load_npz(file))
        return T

    def get_action(self, env):
        raise NotImplementedError

# note: this class is for pure plain VI
# no logic stuff
class VIPolicy(PolicyBase):

    def __init__(self):
        self.Q = None
        self.num_iter = 60
        self.T = None
    
    def init_value_function(self, ss_size):
        V = np.zeros((ss_size,))
        return V

    def make_transition_function(self, env):
        initial_state = env.get_state()
        T = env.make_transition_function()
        self.T = T
        env.set_state(initial_state)

    def make_policy(self, env):
        # NOTE: need to change this so that reward func is
        # calculated for every new environment
        R = self.load_reward_function(env)

        self.make_transition_function(env)

        ss_size = np.prod(env.get_full_state_space())
        V = self.init_value_function(ss_size)

        nA = len(self.T) # number of actions
        gamma = 0.99

        Q = np.zeros((ss_size, nA))
        for k in range(self.num_iter):
            for i, t in enumerate(self.T):
                Q[:, i] = t.dot(R + gamma * V)
            V = np.max(Q, axis=1)

        self.Q = Q
    
    def get_action(self, env):
        if self.Q is None:
            print("policy not yet calculated!")
            return 0
        
        state = env.get_state()
        s_idx = env.state_to_idx(state)
        
        action = np.argmax(self.Q[s_idx])

        for a_n, a in env.action_dict.items():
            if a == action:
                action_name = a_n

        return action_name

class LVIPolicy(VIPolicy):

    def __init__(self):
        super().__init__() 
        self.P = None

    def make_prop_map(self, env):
        initial_state = env.get_state()
        P = env.make_prop_map()
        self.P = P
        env.set_state(initial_state)

    def make_reward_function(self, ss_size, nF):
        R = np.zeros((nF, ss_size))
        # trap has negative reward
        R[-1, :] = -10
        # goal has positive
        R[-2, :] = 10
        return R

    def init_value_function(self, ss_size, nF):
        V = np.zeros((nF, ss_size))
        return V

    # this function shouldn't be in the Policy class
    # should be in Sim or Env
    def get_fsa_state(self, env, f, tm):
        state = env.get_state()
        s_idx = env.state_to_idx(state)
        p = np.argmax(self.P[:, s_idx])
        print("props: {}".format(self.P[:, s_idx]))

        next_f = np.argmax(tm[f, :, p])
        print(tm[f, :, p])
        
        return next_f

    def get_action(self, env, f):
        if self.Q is None:
            print("policy not yet calculated!")
            return 0
        
        state = env.get_state()
        s_idx = env.state_to_idx(state)
        
        action = np.argmax(self.Q[f, s_idx])

        for a_n, a in env.action_dict.items():
            if a == action:
                action_name = a_n

        return action_name

    def make_policy(self, env, tm):
        # TM: f x f x p
        # P: p x s
        self.make_prop_map(env)

        # T: a x s x s (a list over a)
        self.make_transition_function(env)

        nF = tm.shape[0]
        ss_size = np.prod(env.get_full_state_space())
        
        # R: f x s
        R = self.make_reward_function(ss_size, nF)
        # V: f x s
        V = self.init_value_function(ss_size, nF)

        # transform P to be a new shape
        # p x s ===> f x f x p x s
        PM = self.P[None, None, ...]
        PM = np.tile(PM, (nF, nF, 1, 1))

        # assign the prob of f' given p to every state s
        # (f x f x p x 1) times (f x f x p x s)
        preC = tm[..., None]*PM

        # sum over props so that you assign total prob
        # of f' for every state s
        # (f x f x p x s) ==> (f x f x s)
        C = np.sum(preC, axis=2)

        nA = len(self.T) # number of actions
        gamma = 0.99

        # Q: f x s x a
        Q = np.zeros((nF, ss_size, nA))
        for k in range(self.num_iter):
            # NOTE: it is very UNFORTUNATE that I have to
            # iterate over every FSA state, due to the sparse
            # matrix operation.... I wonder if it's worth it
            for f in range(nF):
                for i, t in enumerate(self.T):
                    # (s x 1) = (s x s) times (s x 1 + s x 1)
                    Q[f, :, i] = t.dot(R[f] + gamma * V[f])
            V = np.max(Q, axis=2)
            preV = np.tile(V[None, ...], (nF, 1, 1))
            V = np.sum(preV*C, axis=1)
            # print('hi')


        self.Q = Q

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
