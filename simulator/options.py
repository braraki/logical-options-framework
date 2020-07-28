import numpy as np
import scipy.sparse as sparse
from pathlib import Path

# efficient sparse matrix construction:
from scipy.sparse import dok_matrix
# efficient matrix-vector multiplication:
from scipy.sparse import csr_matrix

import random
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/RRT/")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/RRTStar/")

try:
    from rrt import RRT
    # from rrt_with_pathsmoothing import path_smoothing
    from rrt_star import RRTStar
except ImportError:
    raise


class Subgoal(object):

    def __init__(self, name, prop_index, subgoal_index, state_index):
        self.name = name
        # index in the overall list of props
        self.prop_index = prop_index
        # index in the list of subgoals (theoretically
        # the same as the prop_index)
        self.subgoal_index = subgoal_index
        self.state_index = state_index

class SafetySpec(object):

    def __init__(self, subgoal, safety_spec, tm, safety_costs):
        self.subgoal = subgoal
        self.spec = safety_spec
        self.tm = tm
        self.nF = tm.shape[0]
        self.safety_costs = safety_costs

class TaskSpec(object):

    def __init__(self, task_spec, tm, task_state_costs):
        self.spec = task_spec
        self.tm = tm
        self.nF = tm.shape[0]
        self.task_state_costs = task_state_costs

class OptionBase(object):

    def __init__(self, safety_spec, subgoal):
        # safety_spec is an instance of SafetySpec
        # subgoal is an instance of Subgoal
        self.safety_spec = safety_spec
        self.subgoal = subgoal
        self.nF = safety_spec.nF
        self.tm = safety_spec.tm

        self.initiation_set = None
        self.policy = None
        self.termination_condition = None
        self.poss = None
        self.reward = None

    def is_terminated(self, env):
        raise NotImplementedError()

    def get_action(self, env):
        raise NotImplementedError()

class VIOption(OptionBase):

    def __init__(self, safety_spec, subgoal, T, P, ss_size):
        # subgoal is the index of the subgoal prop
        super().__init__(safety_spec, subgoal)

        self.ss_size = ss_size

        num_iter = 90
        V, Q = self.make_policy(safety_spec, T, P, num_iter)

        self.policy = Q

        self.reward = V

        self.poss = self.init_poss(subgoal)

        self.termination_condition = lambda state : subgoal.state_index == state

    def is_terminated(self, env):
        if self.termination_condition is None:
            print("termination condition for subgoal {} is undefined!".format(self.subgoal.name))

        state = env.get_state()
        s_idx = env.state_to_idx(state)

        return self.termination_condition(s_idx)

    def get_action(self, env):
        if self.policy is None:
            print("policy for subgoal {} not yet calculated!".format(self.subgoal.name))
            return 0
        
        state = env.get_state()
        s_idx = env.state_to_idx(state)
        
        action = np.argmax(self.policy[s_idx, :])

        for a_n, a in env.action_dict.items():
            if a == action:
                action_name = a_n

        return action_name

    def init_poss(self, subgoal):
        # define a transition matrix for poss, p(s' | s, o)
        # poss = poss[option][current state, end state] = prob of end state given current state and option
        poss = dok_matrix((self.ss_size, self.ss_size))

        # assume when initializing that given an option, every state is guaranteed to
        # reach the subgoal. While calculating the actual policies, if the policy
        # doesn't reach the subgoal for a given state, then set
        # poss[o][state, subgoal] = 0
        poss[:, subgoal.state_index] = 1

        return poss

    def make_reward_function(self, ss_size, safety_costs, P):
        R = np.zeros((ss_size))
        for i, cost in enumerate(safety_costs):
            R += cost * P[i]
        return R

    def init_value_function(self, ss_size):
        V = np.zeros((ss_size))
        return V

    def make_policy(self, safety_spec, T, P, num_iter):
        # P: p x s
        # T: a x s x s (a list over a)

        # self.nO
        # self.ss_size

        nA = len(T)

        # R: s
        R = self.make_reward_function(self.ss_size,
                safety_spec.safety_costs, P)
        # V: s
        V = self.init_value_function(self.ss_size)

        # using NO discounting right now, as in SoRB
        gamma = 1. # 0.99

        # Q: s x a
        Q = np.zeros((self.ss_size, nA))
        for k in range(num_iter):
            for i, t in enumerate(T):
                Q[:, i] = t.dot(R + gamma * V)

            V = np.max(Q, axis=1)
        return V, Q    

class LVIOption(OptionBase):

    def __init__(self, safety_spec, subgoal, T, P, ss_size):
        # subgoal is the index of the subgoal prop
        super().__init__(safety_spec, subgoal)

        self.ss_size = ss_size

        num_iter = 30
        V, Q = self.make_policy(safety_spec, T, P, num_iter)
        # the option reward function is just the value
        # function of the INITIAL safety FSA state
        # the other states don't matter I believe
        # we should be able to assume the option's reward
        # is always queried from the initial state
        self.policy = Q

        self.reward = V[0] # f x s ==> s

        self.poss = self.init_poss(subgoal)

        self.termination_condition = lambda state : P[subgoal.state_index, state] == 1

    def is_terminated(self, env):
        if self.termination_condition is None:
            print("termination condition for subgoal {} is undefined!".format(self.subgoal.name))

        state = env.get_state()
        s_idx = env.state_to_idx(state)

        return self.termination_condition(s_idx)

    def get_action(self, env):
        if self.policy is None:
            print("policy for subgoal {} not yet calculated!".format(self.subgoal.name))
            return 0
        
        state = env.get_state()
        s_idx = env.state_to_idx(state)
        
        action = np.argmax(self.policy[s_idx, :])

        for a_n, a in env.action_dict.items():
            if a == action:
                action_name = a_n

        return action_name

    def init_poss(self, subgoal):
        # define a transition matrix for poss, p(s' | s, o)
        # poss = poss[option][current state, end state] = prob of end state given current state and option
        poss = dok_matrix((self.ss_size, self.ss_size))

        # assume when initializing that given an option, every state is guaranteed to
        # reach the subgoal. While calculating the actual policies, if the policy
        # doesn't reach the subgoal for a given state, then set
        # poss[o][state, subgoal] = 0
        poss[:, subgoal.state_index] = 1

        return poss

    def make_reward_function(self, ss_size, nF, safety_state_costs):
        R = np.zeros((nF, ss_size))
        for i, cost in enumerate(safety_state_costs):
            R[i, :] = cost
        return R

    def init_value_function(self, ss_size, nF):
        V = np.zeros((nF, ss_size))
        return V

    def make_policy(self, safety_spec, T, P, num_iter=30):
        # tm: f x f x p
        tm = safety_spec.tm
        # P: p x s
        # T: a x s x s (a list over a)

        nF = safety_spec.nF
        # self.ss_size
        
        # self.R: f x s
        R = self.make_reward_function(self.ss_size, nF,
                safety_spec.safety_state_costs)
        # V: f x s
        V = self.init_value_function(self.ss_size, nF)

        # transform P to be a new shape
        # p x s ===> f x f x p x s
        PM = P[None, None, ...]
        PM = np.tile(PM, (nF, nF, 1, 1))

        # assign the prob of f' given p to every state s
        # (f x f x p x 1) times (f x f x p x s)
        preC = tm[..., None]*PM

        # sum over props so that you assign total prob
        # of f' for every state s
        # (f x f x p x s) ==> (f x f x s)
        C = np.sum(preC, axis=2)

        nA = len(T) # number of actions
        gamma = 0.99

        # Q: f x s x a
        Q = np.zeros((nF, self.ss_size, nA))
        for k in range(num_iter):
            # NOTE: it is very UNFORTUNATE that I have to
            # iterate over every FSA state, due to the sparse
            # matrix operation.... I wonder if it's worth it
            for f in range(nF):
                for i, t in enumerate(T):
                    # (s x 1) = (s x s) times (s x 1 + s x 1)
                    Q[f, :, i] = t.dot(R[f] + gamma * V[f])
            V = np.max(Q, axis=2)
            preV = np.tile(V[None, ...], (nF, 1, 1))

            V = np.sum(preV*C, axis=1)

        return V, Q    

class MetaPolicyBase(object):

    def __init__(self, subgoals, task_spec, safety_props, safety_specs, env):
        self.task_spec = task_spec
        self.tm = task_spec.tm
        # instances of Subgoal
        self.subgoals = subgoals
        # assume it's the index of each safety prop in the list of props
        self.safety_props = safety_props
        # a list of safety specs, one for each subgoal
        self.safety_specs = safety_specs

        # number of options
        self.nO = len(self.subgoals)
        # number of safety props
        self.nS = len(self.safety_props)

        self.options = None
        self.reward = None
        self.poss = None

    def init_poss(self, subgoals):
        raise NotImplementedError

class VIMetaPolicy(MetaPolicyBase):

    def __init__(self, subgoals, task_spec, safety_props, safety_specs, env):
        super().__init__(subgoals, task_spec, safety_props, safety_specs, env)

        self.T = self.make_transition_function(env)
        # assume that the order of props is always
        # [subgoals, safety props, empty]
        self.P = self.make_prop_map(env)

        # state space size
        self.ss_size = np.prod(env.get_full_state_space())
        self.nF = task_spec.nF
        # number of actions
        self.nA = len(self.T)

        self.options = self.make_logical_options(subgoals, safety_specs)
        self.poss = self.make_poss()
        self.option_rewards = self.make_option_rewards_function()

        self.Q = self.make_policy(env, task_spec)

    def make_transition_function(self, env):
        initial_state = env.get_state()
        T = env.make_transition_function()
        env.set_state(initial_state)
        return T

    def make_prop_map(self, env):
        initial_state = env.get_state()
        P = env.make_prop_map()
        env.set_state(initial_state)
        return P

    def make_logical_options(self, subgoals, safety_specs):
        options = []
        for subgoal, safety_spec in zip(subgoals, safety_specs):
            option = VIOption(safety_spec, subgoal, self.T, self.P, self.ss_size)
            options.append(option)
        return options

    def make_option_rewards_function(self):
        reward_list = [option.reward for option in self.options]
        reward = np.vstack(reward_list)

        return reward

    def make_poss(self):
        # define a transition matrix for poss, p(s' | s, o)
        # poss = poss[option][current state, end state] = prob of end state given current state and option
        poss = [option.poss for option in self.options]

        return poss

    def get_option(self, env, f):
        if self.Q is None:
            print("policy not yet calculated!")
            return 0
        
        # Q: f x s x o
        state = env.get_state()
        s_idx = env.state_to_idx(state)

        option = np.argmax(self.Q[f, s_idx])

        return option

    def get_action(self, env, option):
        return self.options[option].get_action(env)

    def is_terminated(self, env, option):
        return self.options[option].is_terminated(env)

    def get_fsa_state(self, env, f):
        # state = env.get_state()
        # s_idx = env.state_to_idx(state)

        p = env.get_proposition()

        # p = np.argmax(self.P[:, s_idx])
        # print("props: {}".format(self.P[:, s_idx]))

        next_f = np.argmax(self.tm[f, :, p])
        # # print(tm[f, :, p])
        
        return next_f

    def make_reward_function(self, task_spec):
        R = np.zeros((self.nF, self.ss_size, self.nO))

        for i, cost in enumerate(task_spec.task_state_costs):
            R[i] = cost
        return R

    def init_value_function(self, ss_size, nF):
        V = np.zeros((nF, ss_size))
        return V

    def make_policy(self, env, task_spec, num_iter=50):
        # TM: f x f x p
        # P: p x s

        # T: a x s x s (a list over a)

        # self.option_reward: o x s
        # poss : o x s x s

        # self.nF
        # self. ss_size
        # self.nO

        # R: f x s x o
        R = self.make_reward_function(task_spec)
        # R: (f x o) times (o x s) ==> (f x s)
        # R = R.dot(lowV[..., 0])
        # V: f x s
        V = self.init_value_function(self.ss_size, self.nF)

        # transform P to be a new shape
        # p x s ===> f x f x p x s
        PM = self.P[None, None, ...]
        PM = np.tile(PM, (self.nF, self.nF, 1, 1))

        # assign the prob of f' given p to every state s
        # (f x f x p x 1) times (f x f x p x s)
        preC = self.tm[..., None]*PM

        # sum over props so that you assign total prob
        # of f' for every state s
        # (f x f x p x s) ==> (f x f x s)
        C = np.sum(preC, axis=2)

        # self.nA
        # gamma = 0.99 the gamma is baked in to poss

        # Q: f x s x o
        Q = np.zeros((self.nF, self.ss_size, self.nO))
        for k in range(num_iter):
            # (f x s x o) = (o x f x f) times (f x s)
            # Q = T * (R + gamma*V) -- OLD
            # Q = R + T * V

            for i, o in enumerate(self.poss):
                # (s x f) = (s x s) times (s x f + s x f)
                # preQ =  R[..., i].T + o.dot(np.tile(lowV[i, :, 0, None], [1, nF]) + V.T)
                preQ = np.multiply(R[..., i].T, np.tile(self.option_rewards[i, :, None], [1, self.nF]) - 1) + o.dot(V.T)
                # (f x s)
                Q[..., i] = preQ.T

            V = np.max(Q, axis=2)
            preV = np.tile(V[None, ...], (self.nF, 1, 1))

            V = np.sum(preV*C, axis=1)

        return Q


class LVIMetaPolicy(MetaPolicyBase):

    def __init__(self, subgoals, safety_specs, safety_props, env, tm):
        super().__init__(subgoals, safety_specs, safety_props, env, tm)

        self.T = self.make_transition_function(env)
        # assume that the order of props is always
        # [subgoals, safety props, empty]
        self.P = self.make_prop_map(env)

        # state space size
        self.ss_size = np.prod(env.get_full_state_space())
        # number of actions
        self.nA = len(self.T)

        self.options = self.make_logical_options(subgoals, safety_specs)
        self.poss = self.make_poss()
        self.option_rewards = self.make_option_rewards_function()

        self.Q = self.make_policy(env, tm)

    def make_transition_function(self, env):
        initial_state = env.get_state()
        T = env.make_transition_function()
        env.set_state(initial_state)
        return T

    def make_prop_map(self, env):
        initial_state = env.get_state()
        P = env.make_prop_map()
        env.set_state(initial_state)
        return P

    def make_logical_options(self, subgoals, safety_specs):
        options = []
        for subgoal, safety_spec in zip(subgoals, safety_specs):
            option = LVIOption(safety_spec, subgoal, self.T, self.P, self.ss_size)
            options.append(option)
        return options

    def make_option_rewards_function(self):
        reward_list = [option.reward for option in self.options]
        reward = np.vstack(reward_list)

        return reward

    def make_poss(self):
        # define a transition matrix for poss, p(s' | s, o)
        # poss = poss[option][current state, end state] = prob of end state given current state and option
        poss = [option.poss for option in self.options]

        return poss

    def get_option(self, env, f):
        if self.Q is None:
            print("policy not yet calculated!")
            return 0
        
        # Q: f x s x o
        state = env.get_state()
        s_idx = env.state_to_idx(state)

        option = np.argmax(self.Q[f, s_idx])

        return option

    def make_reward_function(self, ss_size, nF, nO):
        R = np.zeros((nF, ss_size, nO)) + 1
        # trap has negative reward
        R[-1] = 10
        # penalize (?) left lane states
        R[1] += 1
        R[6] += 1
        R[2] += 1
        R[7] += 1
        # goal has positive
        R[-2] = 0
        return R

    def init_value_function(self, ss_size, nF):
        V = np.zeros((nF, ss_size))
        return V

    def make_policy(self, env, tm):
        # TM: f x f x p
        # P: p x s

        # T: a x s x s (a list over a)

        # self.option_reward: o x s
        # poss : o x s x s

        # self.nF
        # self. ss_size
        # self.nO

        # R: f x s x o
        R = self.make_reward_function(self.ss_size, self.nF, self.nO)
        # R: (f x o) times (o x s) ==> (f x s)
        # R = R.dot(lowV[..., 0])
        # V: f x s
        V = self.init_value_function(self.ss_size, self.nF)

        # transform P to be a new shape
        # p x s ===> f x f x p x s
        PM = self.P[None, None, ...]
        PM = np.tile(PM, (self.nF, self.nF, 1, 1))

        # assign the prob of f' given p to every state s
        # (f x f x p x 1) times (f x f x p x s)
        preC = tm[..., None]*PM

        # sum over props so that you assign total prob
        # of f' for every state s
        # (f x f x p x s) ==> (f x f x s)
        C = np.sum(preC, axis=2)

        # self.nA
        # gamma = 0.99 the gamma is baked in to poss

        # Q: f x s x o
        Q = np.zeros((self.nF, self.ss_size, self.nO))
        for k in range(self.high_num_iter):
            # (f x s x o) = (o x f x f) times (f x s)
            # Q = T * (R + gamma*V) -- OLD
            # Q = R + T * V

            for i, o in enumerate(self.poss):
                # (s x f) = (s x s) times (s x f + s x f)
                # preQ =  R[..., i].T + o.dot(np.tile(lowV[i, :, 0, None], [1, nF]) + V.T)
                preQ = np.multiply(R[..., i].T, np.tile(self.option_rewards[i, :, None], [1, self.nF]) - 1) + o.dot(V.T)
                # (f x s)
                Q[..., i] = preQ.T

            V = np.max(Q, axis=2)
            preV = np.tile(V[None, ...], (self.nF, 1, 1))

            V = np.sum(preV*C, axis=1)
            # print('hi')


        return Q
