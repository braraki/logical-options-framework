import numpy as np
import scipy.sparse as sparse
from pathlib import Path

# efficient sparse matrix construction:
from scipy.sparse import dok_matrix
# efficient matrix-vector multiplication:
from scipy.sparse import csr_matrix

import random

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

    def make_policy(self, env):
        # NOTE: need to change this so that reward func is
        # calculated for every new environment
        R = env.make_reward_function()

        self.T = env.make_transition_function()

        ss_size = np.prod(env.get_full_state_space())
        V = self.init_value_function(ss_size)

        nA = len(self.T) # number of actions
        gamma = 0.99

        Q = np.zeros((ss_size, nA))
        for k in range(self.num_iter):
            RplusV = R + gamma*V
            for i, t in enumerate(self.T):
                Q[:, i] = t.dot(RplusV)
            V = np.max(Q, axis=1)

        print(V)

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

class OptionsPolicy(PolicyBase):

    def __init__(self):
        self.Q = None
        self.subQ = None
        self.high_num_iter = 50
        self.low_num_iter = 50
        self.T = None
        self.poss = None # p(s' | s, o)

    def init_value_function(self, ss_size, nF):
        V = np.zeros((nF, ss_size))
        return V

    # this function can be greatly simplified
    # since I'm assuming that there is only one instance
    # of every goal, so poss is known virtually automatically
    def make_poss(self, goal_ids):
        # goal_ids: (o x s)
        # each value is the ID of g = o(s)
        # (aka the end state of option o starting from state s)

        nO = goal_ids.shape[0]
        ss_size = goal_ids.shape[1]

        # define a transition matrix for poss, p(s' | s, o)
        poss = [dok_matrix((ss_size, ss_size)) for o in np.arange(nO)]

        for state in np.arange(ss_size):
            for o in np.arange(nO):
                goal_state = goal_ids[o, state]
                if goal_state != -1:
                    poss[o][state, goal_state] = 1

        self.poss = poss

    def init_low_level_value_and_q_functions_and_poss(self, ss_size, nO):
        # the value func is 2 columns
        # the first column is the value, and
        # the second column is the state index of the goal from the given state
        V = np.zeros((nO, ss_size, 2))

        # the goal index, if unknown, is -1
        V[:, :, 1] = -1

        # if the state is a goal, then its goal idx is its own index

        # list of indices corresponding to state idxs of props
        goalIdxList = np.nonzero(self.P[:-2] == 1) # need to change this to only use SUBGOALS
        goalIdxList = tuple(list(goalIdxList) + [np.ones_like(goalIdxList[0], dtype=int)])
        # array of indices corresponding to state idxs of props
        goalIdxs = np.argwhere(self.P[:-2] == 1) # ONLY SUBGOALS
        # set the goalStates to equal the idxs of the props
        V[goalIdxList] = goalIdxs[:, 1]

        nA = len(self.T)
        Q = np.zeros((nO, ss_size, nA, 2))

        # I believe this sets the 'goal index' column of Q
        # to be the next state given the current state and action
        for i, t in enumerate(self.T):
            # V[0, :, 1] --> need next state IDs, which are the same
            # for every option as the actions apply the same to all options
            next_goals = t.dot(V[0, :, 1])
            Q[:, :, i, 1] = np.tile(next_goals[None, ...], (nO, 1))

        return V, Q

    def make_transition_function(self, env):
        initial_state = env.get_state()
        T = env.make_transition_function()
        self.T = T
        env.set_state(initial_state)

    # for options, need to make a reward function
    # for each option - in other words,
    # take the option/prop and use the prop map
    # to determine which states
    # (the states where the prop is true) to reward
    def make_low_level_reward_functions(self):
        # the reward function for each state is basically
        # the same as the prop map - for a given prop,
        # states where the prop is true have +0 reward
        # all other states have -1 reward
        # -1 bc each time step is penalized
        R = np.copy(self.P) - 1
        # R[np.where(R == 0)] = 1

        # R should not include the empty prop
        # nor any safety props
        R = R[:-2]

        # the main difference is that R should not include
        # the empty prop
        # R = R[:-1]

        # for every subgoal proposition in self.P, take the safety props
        # and add them as highly negative rewards
        for i, prop in enumerate(self.P[:-2]): # ONLY SUBGOALS
            R[i] += -1000*self.P[-2] # ALL SAFETY PROPS

        # for every proposition in self.P, take the other props
        # and add them as highly negative rewards
        # for i, prop in enumerate(self.P[:-1]):
        #     R[i] += -1000*np.sum(np.delete(self.P[:-1], i, axis=0), axis=0)

        # the reward function should apparently only be 0
        # with the 'do nothing' action... actually maybe
        # it doesn't :c
        # R = np.tile(R, (len(self.T), 1, 1))
        # R[1:] = -1
        # for i, t in enumerate(self.T):
        #     # V[0, :, 1] --> need next state IDs, which are the same
        #     # for every option as the actions apply the same to all options
        #     R[i] = t.dot(R[i].T).T

        return R

    # R: f x s x o
    # NEEDS TO BE SIMPLIFIED (for example, each  option
    # and FSA state combo has a unique
    # cost that is applied at every low-level state)
    def make_reward_function(self, ss_size, nF, nO):
        R = np.zeros((nF, ss_size, nO)) + 1
        # trap has negative reward
        # R[-1] = -1
        # goal has positive
        R[-2] = 0
        return R

    def make_prop_map(self, env):
        initial_state = env.get_state()
        P = env.make_prop_map()
        self.P = P
        env.set_state(initial_state)

    # this function shouldn't be in the Policy class
    # should be in Sim or Env
    def get_fsa_state(self, env, f, tm):
        state = env.get_state()
        s_idx = env.state_to_idx(state)

        # if more than 2 props are active
        # if np.sum(self.P[:, s_idx]) > 1:
        ps = np.argwhere(1 == self.P[:, s_idx])
        # next_fs = np.argmax(tm[f, :, ps])
        # next_f_coord = np.argwhere(next_fs != f)[0]
        # next_f = next_fs[next_f_coord]

        # p = np.argmax(self.P[:, s_idx])
        # print("props: {}".format(self.P[:, s_idx]))

        next_f = np.argmax(tm[f, :, ps])
        # # print(tm[f, :, p])
        
        return next_f

    def get_option(self, env, f):
        if self.Q is None:
            print("policy not yet calculated!")
            return 0
        
        # Q: f x s x o
        state = env.get_state()
        s_idx = env.state_to_idx(state)

        option = np.argmax(self.Q[f, s_idx])

        return option

    def get_action(self, env, o):
        if self.subQ is None:
            print("subpolicies not yet calculated!")
            return 0
        
        state = env.get_state()
        s_idx = env.state_to_idx(state)
        
        action = np.argmax(self.subQ[o, s_idx, :, 0])

        for a_n, a in env.action_dict.items():
            if a == action:
                action_name = a_n

        return action_name

    def make_low_level_policies(self, env):
        # assume you have access to
        # self.P: p x s
        # self.T: a x s x s (a list over a)

        # number of options is # of non-empty props (-1)
        # but need to add obstacle prop (+1)
        nO = len(env.props) - 1 # ONLY THE NUMBER OF SUBGOAL PROPS
        ss_size = np.prod(env.get_full_state_space())

        nA = len(self.T)

        # these are the low-level reward funcs
        # reward for achieving a given option
        # lowR: o x s (o = p - 2 since empty prop and safety props not included)
        lowR = self.make_low_level_reward_functions()
        # low-level value funcs:
        # basically a matrix where each row is the value function
        # for a given option
        # the second value is the index of the goal state from the given state
        # lowV: o x s x 2
        # lowQ: o x s x a x 2
        lowV, lowQ = self.init_low_level_value_and_q_functions_and_poss(ss_size, nO)

        # using NO discounting right now, as in SoRB
        gamma = 1. # 0.99


        # NOTE: the 2nd column, which is supposed to store
        # the final state that the policy ends up in from the
        # current state, DOES NOT WORK. need to fix that here
        for k in range(self.low_num_iter):
            for i, t in enumerate(self.T):

                # (s x o) = (s x s) times (s x o + s x o)
                preQ = t.dot(lowR.T + gamma * lowV[:, :, 0].T)

                # (o x s) = (o x s)
                lowQ[:, :, i, 0] = preQ.T
                lowQ[:, :, i, 1] = t.dot(lowV[..., 1].T).T          

            # (o x s x a x 2) ==> (o x s x 2)
            preLowV = np.max(lowQ, axis=2)
            lowV[..., 0] = preLowV[..., 0]
            # these are the states where the final state value
            # is not -1 and therefore these positions in lowV
            # should be updated
            updateFinalStateIdxs = list(np.where(preLowV[..., 1] != -1))
            updateFinalStateIdxs = tuple(updateFinalStateIdxs + [np.ones_like(updateFinalStateIdxs[0])])
            lowV[updateFinalStateIdxs] = preLowV[updateFinalStateIdxs]


        return lowQ, lowV

    def make_policy(self, env, tm):
        # TM: f x f x p
        # P: p x s
        self.make_prop_map(env)

        # T: a x s x s (a list over a)
        self.make_transition_function(env)

        # first column is the values
        # second column is the goal ID of the given state
        # lowQ: o x s x a x 2
        # lowV: o x s x 2
        # poss: o x s x s
        lowQ, lowV = self.make_low_level_policies(env)
        self.make_poss(lowV[..., 1])
        self.subQ = lowQ

        nF = tm.shape[0]
        ss_size = np.prod(env.get_full_state_space())
        nO = self.P.shape[0] - 2 # need to get rid of empty prop AND SAFETY PROPS
        
        # R: f x s
        R = self.make_reward_function(ss_size, nF, nO)
        # R: (f x o) times (o x s) ==> (f x s)
        # R = R.dot(lowV[..., 0])
        # V: f x s
        V = self.init_value_function(ss_size, nF)

        # # P(s' | s, o) ---> probability of s' given
        # # option o and current state s
        # # (o x s) .times (o x s)
        # pre_poss = self.P[:-1] * lowV
        # # get the indices of the max value in each row
        # # this is the end state of each option
        # poss_max_indices = np.argmax(pre_poss, axis=1)
        # poss = np.zeros_like(pre_poss)
        # poss[np.arange(pre_poss.shape[0]), poss_max_indices] = 1
        # # poss is "probability of s' given o and s"

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
        # gamma = 0.99 the gamma is baked in to poss

        # Q: f x s x o
        Q = np.zeros((nF, ss_size, nO))
        for k in range(self.high_num_iter):
            # (f x s x o) = (o x f x f) times (f x s)
            # Q = T * (R + gamma*V) -- OLD
            # Q = R + T * V

            for i, o in enumerate(self.poss):
                # (s x f) = (s x s) times (s x f + s x f)
                # preQ =  R[..., i].T + o.dot(np.tile(lowV[i, :, 0, None], [1, nF]) + V.T)
                preQ = np.multiply(R[..., i].T, np.tile(lowV[i, :, 0, None], [1, nF]) - 1) + o.dot(V.T)
                # (f x s)
                Q[..., i] = preQ.T

            V = np.max(Q, axis=2)
            preV = np.tile(V[None, ...], (nF, 1, 1))

            V = np.sum(preV*C, axis=1)
            # print('hi')


        self.Q = Q

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
    # NOTE: this function only works when there is a
    # single proposition true at any given state, for now
    def get_fsa_state(self, env, f, tm):
        state = env.get_state()
        s_idx = env.state_to_idx(state)

        # if more than 2 props are active
        # if np.sum(self.P[:, s_idx]) > 1:
        # ps = np.argwhere(1 == self.P[:, s_idx])
        # next_fs = np.argmax(tm[f, :, ps])
        # next_f_coord = np.argwhere(next_fs != f)[0]
        # next_f = next_fs[next_f_coord]

        # list of which props are on (1) and off (0)
        # props = self.P[:, s_idx]
        # for prop in props:
        #     if prop

        p = np.argmax(self.P[:, s_idx])
        # print("props: {}".format(self.P[:, s_idx]))

        next_f = np.argmax(tm[f, :, p])
        # # print(tm[f, :, p])
        
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
        self.T = env.make_transition_function()

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

class HardCodedLineWorldPolicy(PolicyBase):
    def get_action(self, env):
        return random.choice(['left', 'right', 'nothing'])

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
