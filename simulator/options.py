import numpy as np
import scipy.sparse as sparse
from pathlib import Path
import copy

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
                "/RRTDubins/")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/RRTStar/")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/RRTStarReedsShepp/")

try:
    from rrt import RRT
    # from rrt_with_pathsmoothing import path_smoothing
    from rrt_star import RRTStar
    from rrt_star_reeds_shepp import RRTStarReedsShepp
    from rrt_dubins import RRTDubins
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

class RRTOption(OptionBase):

    def __init__(self, env, states, safety_spec, subgoal, T, P, ss_size, vi_option):
        # subgoal is the index of the subgoal prop
        super().__init__(safety_spec, subgoal)

        self.vi_option = vi_option

        self.ss_size = ss_size
        self.policy = self.vi_option.policy
        self.reward = self.vi_option.reward
        self.termination_condition = self.vi_option.termination_condition

        num_iter = 10 # currently unused
        # 'states' are the states (indices) to be considered, in this case the initial state
        # should also probably include all subgoals in the future
        self.rrt_policies, self.costs, self.poss = self.make_policy(env, states, safety_spec, T, P, num_iter)

        def termination_condition(state):
            subgoal_state = env.idx_to_state(subgoal.state_index)
            if subgoal_state == state:
                print('f ffffff')
                print('ffffff fff ff f f')
            return subgoal_state == state

        self.termination_condition = lambda state : termination_condition(state)

    def is_terminated(self, env):
        if self.termination_condition is None:
            print("termination condition for subgoal {} is undefined!".format(self.subgoal.name))

        state = env.get_rrt_state()

        return self.termination_condition(state)

    def get_action(self, env, steps):
        start_state = env.option_start
        state = env.get_rrt_state()
        path = self.rrt_policies[tuple(start_state)][:-1]
        if steps < len(path):
            return path[-1-steps]
        else:
            return tuple(state)

    def round_state_to_idx(self, env, state):
        discrete_state = list(np.around([state[0], state[1]]).astype(int))
        idx = env.state_to_idx(discrete_state)
        return idx

    def make_policy(self, env, states, safety_spec, T, P, num_iter):
        poss = {}
        costs = {}
        paths = {}
        all_states = set([env.idx_to_state(s) for s in range(self.ss_size)])

        # note that state is a CONTINUOUS (x, y) VARIABLE
        # not an index
        for state in states:
            poss[tuple(state)] = []

            # if P[-2, s] == 1:
            #     state = env.idx_to_state(s)
            #     path = [tuple(state), tuple(state)]
            #     paths.append(path)
            #     continue

            # s is the index of the state rounded to the nearest discrete state
            s = self.round_state_to_idx(env, state)
            ns = s

            # this is for building up the discrete path
            start = env.idx_to_state(ns)
            discrete_path = [start]
            path_set = set([start])

            for i in range(30): # need to fix this arbitrary number
                action = np.argmax(self.vi_option.policy[ns])
                ns = np.argmax(T[action][ns])
                next_state = env.idx_to_state(ns)
                if state not in discrete_path:
                    discrete_path.append(next_state)
                path_set.add(next_state)
                if ns == self.subgoal.state_index:
                    break
        
            goal_idx = ns

            # let's assume that the policy is deterministic, so starting 
            # in state is guaranteed to end up in goal
            # this can be changed to be probabilistic later if needed
            goal_state = env.idx_to_state(goal_idx)
            poss[tuple(state)] = goal_state
            poss[tuple(goal_state)] = goal_state

            obstacles = all_states.difference(path_set)
            obstacle_sizes = self.process_obstacles(path_set, obstacles)
            obstacles = [(*ob, size) for ob, size in zip(obstacles, obstacle_sizes)]
        
            # idk why but the RRT's path is backwards
            # so i've started start and goal
            cost = 0
            if ns != self.subgoal.state_index:
                print('didnt reach goal')
                cost = -10000
            if start == goal_state:
                path = [state, goal_state]
            else:
                # rrt = RRTStar(start=start,
                #         goal=goal,
                #         rand_area=[0, env.dom_size[0]], # NEED TO VARIABILIZE THIS
                #         obstacle_list=list(obstacles),
                #         max_iter=200,
                #         path=discrete_path
                #         )
                start = [*state, np.deg2rad(90.)]
                goal = [*goal_state, np.deg2rad(90.)]
                print(start, goal)
                rrt = RRTStarReedsShepp(start=start,
                                        goal=goal,
                                        rand_area=[0, env.dom_size[0]],
                                        obstacle_list=list(obstacles),
                                        max_iter=10,
                                        path=discrete_path
                                        )
                path = rrt.planning(animation=False, search_until_max_iter=True)
                if path is None:
                    path = [start[:2], start[:2]]
                    cost = -10000
                    print('f')
                else:
                    path = [[x, y] for [x, y, yaw] in path]
                    for i, (x, y) in enumerate(path[1:]):
                        discrete_state = list(np.around([x, y]).astype(int))
                        idx = env.state_to_idx(discrete_state)
                        p = np.argmax(P[:, idx])
                        nP = P.shape[0]
                        # if the active prop is the obstacle, then
                        # set it to be the empty prop instead
                        if p == nP - 2:
                            p = nP - 1
                        
                        dist = np.sqrt((x - path[i][0])**2 + (y - path[i][1])**2)
                        print(x, y, path[i][0], path[i][1], dist, p, safety_spec.safety_costs[p])
                        cost += safety_spec.safety_costs[p] * dist

            costs[tuple(state)] = cost
            paths[tuple(state)] = path


        print(poss, costs, paths)
        return paths, costs, poss

    def process_obstacles(self, path, obstacles):

        sizes = []

        for obstacle in obstacles:
            #        du   dd   dl   dr
            size = [0.5, 0.5, 0.5, 0.5]

            for step in path:
                above = (step[0], step[1]+1)
                below = (step[0], step[1]-1)
                left = (step[0]-1, step[1])
                right = (step[0]+1, step[1])

                change = 0.2
                if above == obstacle:   # up (du)
                    size[1] -= change
                elif below == obstacle: # down (dd)
                    size[0] -= change
                elif left == obstacle:  # left (dl)
                    size[3] -= change
                elif right == obstacle: # right (dr)
                    size[2] -= change

            sizes.append(size)

        return sizes

class QLearningOption(OptionBase):

    def __init__(self, safety_spec, subgoal, T, P, ss_size):
        # subgoal is the index of the subgoal prop
        super().__init__(safety_spec, subgoal)

        self.ss_size = ss_size

        nA = len(T)
        V, Q = self.init_vq(self.ss_size, nA)

        self.policy = Q

        self.reward = V

        # this is the low-level reward function
        self.R = self.make_reward_function(ss_size, safety_spec.safety_costs, P)

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

    def init_vq(self, ss_size, nA):
        V = np.zeros((ss_size))

        Q = np.zeros((ss_size, nA))
        return V, Q

    def make_reward_function(self, ss_size, safety_costs, P):
        R = np.zeros((ss_size))
        for i, cost in enumerate(safety_costs):
            R += cost * P[i]
        return R

    def update_qfunction(self, state_index, action, next_state_index, gamma=1, alpha=0.5):
        # self.nO
        # self.ss_size

        # note: self.reward is the Value function
        # and self.policy is the Q function

        # self.R is the low-level reward function

        reward = self.R[state_index]
        q_update = reward + gamma * self.reward[next_state_index] - self.policy[state_index, action]
        self.policy[state_index, action] += alpha * (q_update)
        self.reward = np.max(self.policy, axis=1) 

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

    def __init__(self, subgoals, task_spec, safety_props, safety_specs, env,
                 num_episodes=1000, episode_length=100, gamma=1., alpha=0.5, epsilon=0.3):
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

        # training parameters
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

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

class RRTMetaPolicy(VIMetaPolicy):
    def __init__(self, subgoals, task_spec, safety_props, safety_specs, env):
        super().__init__(subgoals, task_spec, safety_props, safety_specs, env)

        # the list of starting states for options
        # right now, I'm just using the initial state of the env
        # should also probably include all subgoals, at least
        self.states = [env.get_rrt_state()]

        self.rrt_options = self.make_rrt_options(subgoals, safety_specs, env)

        # self.poss = [option]{start state : end state}
        self.rrt_poss = [o.poss for o in self.rrt_options]
        # self.rewards = [option][start state : reward]
        self.rrt_rewards = [o.costs for o in self.rrt_options]
        # self.rrt_policies = [option][start state: path]
        self.rrt_policies = [o.rrt_policies for o in self.rrt_options]

        self.rrt_Q = self.make_rrt_policy(env, task_spec)

    def is_terminated(self, env, option):
        return self.rrt_options[option].is_terminated(env)

    def get_option(self, env, f):
        if self.rrt_Q is None:
            print("policy not yet calculated!")
            return 0
        
        # self.rrt_Q: {(f, (x, y), o)}
        state = env.get_rrt_state()

        nO = len(self.subgoals)

        best_option = 0
        best_state = (f, tuple(state), best_option)
        for o in range(nO):
            Q_state = (f, tuple(state), o)
            if self.rrt_Q[Q_state] > self.rrt_Q[best_state]:
                best_option = o
                best_state = (f, tuple(state), best_option)

        return best_option

    def get_action(self, env, option, steps):
        return self.rrt_options[option].get_action(env, steps)

    def make_rrt_options(self, subgoals, safety_specs, env):
        options = []
        for subgoal, safety_spec, vi_option in zip(subgoals, safety_specs, self.options):
            option = RRTOption(env, self.states, safety_spec, subgoal, self.T, self.P, self.ss_size, vi_option)
            options.append(option)
        return options

    def make_rrt_reward_function(self, task_spec):
        return task_spec.task_state_costs

    def init_rrt_value_function(self, nF):
        V = {}
        for i in range(nF):
            for s in self.states:
                V[(i, tuple(s))] = 0.
                for poss in self.rrt_poss:
                    ns = poss[tuple(s)]
                    V[(i, tuple(ns))] = 0.
        return V

    def round_state_to_idx(self, env, state):
        discrete_state = list(np.around([state[0], state[1]]).astype(int))
        idx = env.state_to_idx(discrete_state)
        return idx

    def make_rrt_policy(self, env, task_spec, num_iter=10):
        # TM: f x f x p
        # P: p x s

        # T: a x s x s (a list over a)

        # self.rrt_rewards: [option][start state: reward]
        # poss : o x s x s

        # self.nF
        # self.nO

        # R: [f] = reward
        R = self.make_rrt_reward_function(task_spec)
        # V: {(f x s)} = value
        V = self.init_rrt_value_function(self.nF)

        # all of the states to consider, aka
        # all start + end states
        all_states = set()
        for s in self.states:
            all_states.add(tuple(s))
            for poss in self.rrt_poss:
                all_states.add(poss[tuple(s)])
        all_states = list(all_states)

        Q = {}
        for k in range(num_iter):
            # poss: [option]{start state : end state}
            for f in range(self.nF):
                for s in all_states:
                    best_o = 0
                    best_o_value = -100000000.
                    for o, poss in enumerate(self.rrt_poss):
                        if s not in poss.keys():
                            poss[s] = s
                        if s not in self.rrt_rewards[o].keys():
                            self.rrt_rewards[o][s] = -1.
                        

                        ns = poss[s]
                        Q[(f, tuple(s), o)] = float(R[f])*float(self.rrt_rewards[o][s]) + float(V[(f, tuple(ns))])
                        if Q[(f, tuple(s), o)] > best_o_value:
                            best_o_value = Q[(f, tuple(s), o)]
                            best_o = o
                    V[(f, tuple(s))] = Q[(f, tuple(s), best_o)]
            
            for s in all_states:
                for f in range(self.nF):
                    idx = self.round_state_to_idx(env, s)
                    p = np.argmax(self.P[:, idx])
                    nf = np.argmax(self.tm[f, :, p])
                    V[(f, tuple(s))] = V[(nf, tuple(s))]

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

# abstract class for metapolicies on discrete spaces
class DiscreteMetaPolicy(MetaPolicyBase):
    def __init__(self, subgoals, task_spec, safety_props, safety_specs, env,
                 num_episodes=1000, episode_length=100, gamma=1., alpha=0.5, epsilon=0.3,
                 record_training=False, recording_frequency=100, experiment_num=0):
        super().__init__(subgoals, task_spec, safety_props, safety_specs, env,
                         num_episodes, episode_length, gamma, alpha, epsilon)

        self.experiment_num = experiment_num

        self.record_training = record_training
        self.recording_frequency = recording_frequency
        if record_training:
            self.training_steps = []
            self.training_reward = []
            self.training_success = []
            self.training_last_state = []

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
        self.q_learning(env)

        self.Q = None

    def q_learning(self, env):
        raise NotImplementedError()

    def evaluate_policy(self, env, episode_num, experiment_num=0, task_spec=None):
        raise NotImplementedError()

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
            option = QLearningOption(safety_spec, subgoal, self.T, self.P, self.ss_size)
            options.append(option)
        return options

    def get_epsilon_greed_action(self, env, option_index, state_index, epsilon):
        if np.random.uniform() < epsilon:
            action_index = np.random.choice(range(self.nA))
        else:
            action_index = np.argmax(self.options[option_index].policy[state_index])

        action_name = None
        for a_n, a in env.action_dict.items():
            if a == action_index:
                action_name = a_n

        return action_name

    def get_random_state(self, env):
        dom = env.dom_size
        x = np.random.choice(range(dom[0]))
        y = np.random.choice(range(dom[1]))
        return [x, y]

    def is_terminated(self, env, option):
        return self.options[option].is_terminated(env)

    def get_fsa_state(self, env, f, tm=None):
        # if a tm is given, use that one. otherwise use the tm
        # used during training
        if tm is None:
            tm = self.tm
        p = env.get_proposition()
        next_f = np.argmax(tm[f, :, p])        
        return next_f

    def make_poss(self):
        # define a transition matrix for poss, p(s' | s, o)
        # poss = poss[option][current state, end state] = prob of end state given current state and option
        poss = [option.poss for option in self.options]

        return poss

    def get_action(self, env, option):
        return self.options[option].get_action(env)

    # used in the make_policy function to do high-level value iteration
    def init_value_function(self, ss_size, nF):
        V = np.zeros((nF, ss_size))
        return V

    def make_option_rewards_function(self):
        reward_list = [option.reward for option in self.options]
        reward = np.vstack(reward_list) - 1

        return reward

    def make_reward_function(self, task_spec):
        R = np.zeros((task_spec.nF, self.ss_size, self.nO))

        for i, cost in enumerate(task_spec.task_state_costs):
            R[i] = cost
        return R

    def get_option(self, env, f):
        if self.Q is None:
            print("policy not yet calculated!")
            return 0
        
        # Q: f x s x o
        state = env.get_state()
        s_idx = env.state_to_idx(state)

        option = np.argmax(self.Q[f, s_idx])

        return option

    def record_results(self, i, num_steps, initial_state, env):
        if i % self.recording_frequency == 0:
            env.set_state(initial_state)
            episode_reward, success, final_f = self.evaluate_policy(env, i/self.recording_frequency, self.experiment_num)
            if self.record_training:
                training_steps = i * self.episode_length + (num_steps - 1)
                self.training_reward.append(episode_reward)
                self.training_steps.append(training_steps)
                self.training_success.append(success)
                self.training_last_state.append(final_f)
            print("Episode: {}\t| Reward: {}\t| Success: {}".format(i, episode_reward, success))

    def record_composability_results(self, env, task_spec, num_iter):
        initial_state = env.get_state()
        rewards, successes, final_fs = self.make_and_evaluate_policy(env, task_spec, self.experiment_num, num_iter=num_iter)
        env.set_state(initial_state)
        self.composability_reward = rewards
        self.composability_success = successes
        self.composability_last_state = final_fs
        self.composability_steps = num_iter
        # print("Episode: {}\t| Reward: {}\t| Success: {}".format(i, episode_reward, success))

    def get_results(self):
        if self.record_training:
            return {'reward': self.training_reward, 'steps': self.training_steps,
                    'success': self.training_success, 'last_state': self.training_last_state}
        else:
            return None

    def get_composability_results(self, task_name):
        return {'reward': self.composability_reward, 'steps': self.composability_steps,
                'success': self.composability_success, 'last_state': self.composability_last_state,
                'task_name': task_name}

    def if_experiment_modification(self, env, f, experiment_num, episode_num, task_spec=None):
        # if task_spec is None, use the training task_spec. Otherwise use
        # the given task_spec. this is for testing composability
        if task_spec is None:
            task_spec = self.task_spec
        
        evens = 0
        if experiment_num % 2 == 0:
            evens = 1
        if task_spec.spec == '(F((a|b) & F(c & F home)) & G ! can) | (F((a|b) & F home) & F can) & G ! o':
            if f == 0:
                if episode_num % 2 == evens:
                    env.prop_dict['canceled'].value = True
                else:
                    env.prop_dict['canceled'].value = False
            else:
                env.prop_dict['canceled'].value = False
        elif task_spec.spec == '(F (c & F a) & G ! can) | (F a & F can) & G ! o':
            if f == 0:
                if episode_num % 2 == evens:
                    env.prop_dict['canceled'].value = True
                else:
                    env.prop_dict['canceled'].value = False
            else:
                env.prop_dict['canceled'].value = False

class QLearningMetaPolicy(DiscreteMetaPolicy):

    def __init__(self, subgoals, task_spec, safety_props, safety_specs, env,
                 num_episodes=1000, episode_length=100, gamma=1., alpha=0.5, epsilon=0.3,
                 record_training=False, recording_frequency=100, experiment_num=0):
        super().__init__(subgoals, task_spec, safety_props, safety_specs, env,
                         num_episodes, episode_length, gamma, alpha, epsilon,
                         record_training, recording_frequency, experiment_num)

        self.Q = self.make_policy(env, task_spec)

    def q_learning(self, env):
        nO = len(self.options)

        initial_state = env.get_state()

        for i in range(self.num_episodes):
            random_state = self.get_random_state(env)
            env.set_state(random_state)
            option_index = i % nO # episodes loop thru options

            num_steps = 0
            # while not self.is_terminated(env, option_index) and num_steps < self.episode_length:
            while num_steps < self.episode_length:
                current_state = env.state_to_idx(env.get_state())
                action = self.get_epsilon_greed_action(env, option_index, current_state, self.epsilon)
                obs = env.step(action)                    
                next_state = env.state_to_idx(env.get_state())

                for option in self.options:
                    option.update_qfunction(current_state, env.action_dict[action], next_state,
                    gamma=self.gamma, alpha=self.alpha)

                num_steps += 1
            
            self.option_rewards = self.make_option_rewards_function()

            self.record_results(i, num_steps, initial_state, env)

        env.set_state(initial_state)

    def evaluate_policy(self, env, episode_num, experiment_num=0, task_spec=None, Q=None):
        # if a task_spec is given, use that one (for composability experiments)
        # otherwise use the one used during training
        if task_spec is None:
            task_spec = self.task_spec

        starting_state = env.get_state()

        if Q is None:
            self.Q = self.make_policy(env, task_spec)
        else:
            self.Q = Q
        f = 0
        goal_state = task_spec.nF - 1
        max_steps_in_option = 30
        max_steps = 100

        reward = 0
        success = False
        num_steps = 0

        while num_steps < max_steps:
            option = self.get_option(env, f)
            f_prev = f
            steps_in_option = 0
            if self.is_terminated(env, option):
                num_steps += 1
                state_index = env.state_to_idx(env.get_state())
                this_reward = task_spec.task_state_costs[f]*self.options[option].R[state_index]
                if this_reward == 0:
                    this_reward = -1
                reward += this_reward
            while not self.is_terminated(env, option) and f_prev == f and steps_in_option < max_steps_in_option and num_steps < max_steps and f != goal_state:
                state_index = env.state_to_idx(env.get_state())

                self.if_experiment_modification(env, f, experiment_num, episode_num, task_spec=task_spec)

                # print(option, task_spec.task_state_costs[f]*self.options[option].R[state_index])

                reward += task_spec.task_state_costs[f]*self.options[option].R[state_index]

                action = self.get_action(env, option)
                obs = env.step(action)
                f_prev = f
                f = self.get_fsa_state(env, f, task_spec.tm)
                steps_in_option += 1
                num_steps += 1
                
            if f == goal_state:
                success = True
                break
            # reward += task_spec.task_state_costs[f]*self.options[option].R[state_index]

        self.Q = None
        env.set_state(starting_state)
        return reward, success, f

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
        V = self.init_value_function(self.ss_size, task_spec.nF)

        # transform P to be a new shape
        # p x s ===> f x f x p x s
        PM = self.P[None, None, ...]
        PM = np.tile(PM, (task_spec.nF, task_spec.nF, 1, 1))

        # assign the prob of f' given p to every state s
        # (f x f x p x 1) times (f x f x p x s)
        preC = task_spec.tm[..., None]*PM

        # sum over props so that you assign total prob
        # of f' for every state s
        # (f x f x p x s) ==> (f x f x s)
        C = np.sum(preC, axis=2)

        # self.nA
        # gamma = 0.99 the gamma is baked in to poss

        # Q: f x s x o
        Q = np.zeros((task_spec.nF, self.ss_size, self.nO))
        for k in range(num_iter):
            # (f x s x o) = (o x f x f) times (f x s)
            # Q = R-1 + T * V
            #                                Q = T * (R + gamma*V) -- OLD
            for i, o in enumerate(self.poss):
                # (s x f) = (s x s) times (s x f + s x f)
                # preQ =  R[..., i].T + o.dot(np.tile(lowV[i, :, 0, None], [1, nF]) + V.T)
                preQ = np.multiply(R[..., i].T, np.tile(self.option_rewards[i, :, None], [1, task_spec.nF]) - 1) \
                       + o.dot(V.T)
                # (f x s)
                Q[..., i] = preQ.T

            V = np.max(Q, axis=2)
            preV = np.tile(V[None, ...], (task_spec.nF, 1, 1))

            V = np.sum(preV*C, axis=1)

        return Q

    def make_and_evaluate_policy(self, env, task_spec, experiment_num, num_iter=50):
        rewards = []
        successes = []
        fs = []

        R = self.make_reward_function(task_spec)
        V = self.init_value_function(self.ss_size, task_spec.nF)
        PM = self.P[None, None, ...]
        PM = np.tile(PM, (task_spec.nF, task_spec.nF, 1, 1))

        preC = task_spec.tm[..., None]*PM
        C = np.sum(preC, axis=2)
        # Q: f x s x o
        Q = np.zeros((task_spec.nF, self.ss_size, self.nO))
        for k in range(num_iter):
            for i, o in enumerate(self.poss):
                preQ = np.multiply(R[..., i].T, np.tile(self.option_rewards[i, :, None], [1, task_spec.nF]) - 1) \
                       + o.dot(V.T)
                # (f x s)
                Q[..., i] = preQ.T

            V = np.max(Q, axis=2)
            preV = np.tile(V[None, ...], (task_spec.nF, 1, 1))

            V = np.sum(preV*C, axis=1)
            reward, success, f = self.evaluate_policy(env, k, experiment_num, task_spec=task_spec, Q=Q)
            rewards.append(reward)
            successes.append(success)
            fs.append(f)

        return rewards, successes, fs

# NO HIGH-LEVEL FSA USED HERE
# (fsa stuff still used at the low level for safety stuff)
class FlatQLearningMetaPolicy(DiscreteMetaPolicy):

    def __init__(self, subgoals, task_spec, safety_props, safety_specs, env,
                 num_episodes=1000, episode_length=100, gamma=1., alpha=0.5, epsilon=0.3,
                 record_training=False, recording_frequency=20, experiment_num=0):
        super().__init__(subgoals, task_spec, safety_props, safety_specs, env,
                         num_episodes, episode_length, gamma, alpha, epsilon,
                         record_training, recording_frequency, experiment_num)

        self.Q = self.high_level_q_learning(env, env.get_state())

    def q_learning(self, env):
        nO = len(self.options)

        initial_state = env.get_state()

        for i in range(self.num_episodes):
            random_state = self.get_random_state(env)
            env.set_state(random_state)
            
            option_index = i % nO # episodes loop thru options

            num_steps = 0
            # while not self.is_terminated(env, option_index) and num_steps < self.episode_length:
            while num_steps < self.episode_length:
                current_state = env.state_to_idx(env.get_state())
                action = self.get_epsilon_greed_action(env, option_index, current_state, self.epsilon)
                obs = env.step(action)
                next_state = env.state_to_idx(env.get_state())

                for option in self.options:
                    option.update_qfunction(current_state, env.action_dict[action], next_state,
                                            gamma=self.gamma, alpha=self.alpha)

                num_steps += 1

            self.record_results(i, num_steps, initial_state, env)

        env.set_state(initial_state)

    def high_level_q_learning(self, env, start_state, alpha=0.5):
        previous_start_state = env.get_state()

        Q = np.zeros((self.ss_size, self.nO))
        V = np.zeros((self.ss_size,))

        num_episodes = 100
        episode_length = 30
        gamma = 1
        epsilon = 0.3
        
        # need to keep track of the true FSA state so that a goal reward can be made
        # and so you can stop learning when the goal state is reached
        f = 0
        goal_state = self.task_spec.nF - 1
        for i in range(num_episodes):
            if i % 2 == 0:
                if self.task_spec.spec == '(F (c & F a) & G ! can) | (F a & F can) & G ! o':
                    f = 3
                if self.task_spec.spec == '(F((a|b) & F(c & F home)) & G ! can) | (F((a|b) & F home) & F can) & G ! o':
                    f = 2
            env.set_state(start_state)
            # self.if_experiment_modification(env, f, 0, i)
            num_steps = 0
            while num_steps < episode_length and f != goal_state:
                current_state = env.state_to_idx(env.get_state())

                if np.random.uniform() < epsilon:
                    option_index = np.random.choice(range(self.nO))
                else:
                    option_index = np.argmax(Q[current_state])

                next_state = np.argmax(self.poss[option_index][current_state].toarray())

                env.set_state(env.idx_to_state(next_state))

                f = self.get_fsa_state(env, f)

                if f == goal_state:
                    reward = 0
                else:
                    reward = self.options[option_index].reward[current_state]
                    # to compensate for the fact that option reward is 0 at the goal
                    # but should be -1 for the overall policy
                    if reward == 0:
                        reward = -1

                q_update = reward + gamma * V[next_state] - Q[current_state, option_index]
                Q[current_state, option_index] += alpha * q_update
                V = np.max(Q, axis=1)

                num_steps += 1

        env.set_state(previous_start_state)

        return Q

    def evaluate_policy(self, env, episode_num, experiment_num=0):
        self.Q = self.high_level_q_learning(env, env.get_state())
        f = 0
        goal_state = self.task_spec.nF - 1
        max_steps_in_option = 30
        max_steps = 100

        reward = 0
        success = False

        num_steps = 0
        option = self.get_option(env)
        while num_steps < max_steps:
            f_prev = f
            steps_in_option = 0


            # this is for the case when the policy decides to squat on a goal
            if self.is_terminated(env, option):
                self.if_experiment_modification(env, f, experiment_num, episode_num)
                option = self.get_option(env)
                state_index = env.state_to_idx(env.get_state())

                state_reward = self.options[option].R[state_index]
                # this is to compensate for the fact that the reward for an option is 0 at the goal
                # but for the overall policy the reward at the option should be -1
                if state_reward == 0:
                    state_reward = -1.0
                # print(option, self.task_spec.task_state_costs[f]*state_reward)
                reward += self.task_spec.task_state_costs[f]*state_reward
                action = self.get_action(env, option)
                obs = env.step(action)
                f_prev = f
                f = self.get_fsa_state(env, f)
                steps_in_option +=1
                num_steps += 1

            while not self.is_terminated(env, option) and steps_in_option < max_steps_in_option and num_steps < max_steps:
                self.if_experiment_modification(env, f, experiment_num, episode_num)
                state_index = env.state_to_idx(env.get_state())
                state_reward = self.options[option].R[state_index]
                # this is to compensate for the fact that the reward for an option is 0 at the goal
                # but for the overall policy the reward at the option should be -1
                if state_reward == 0:
                    state_reward = -1.0
                # print(option, self.task_spec.task_state_costs[f]*state_reward)
                reward += self.task_spec.task_state_costs[f]*state_reward

                action = self.get_action(env, option)
                obs = env.step(action)
                f_prev = f
                f = self.get_fsa_state(env, f)
                steps_in_option += 1
                num_steps += 1
                if f == 1:
                    if np.random.uniform() < 0.0:
                        env.prop_dict['canceled'].value = True
                    else:
                        env.prop_dict['canceled'].value = False
                else:
                    env.prop_dict['canceled'].value = False
            if f == goal_state:
                success = True
                break
            state_index = env.state_to_idx(env.get_state())
            if state_reward == 0:
                state_reward = -1.0
        # reward += self.task_spec.task_state_costs[f]*state_reward

        self.Q = None

        return reward, success, f

    def get_option(self, env):
        if self.Q is None:
            print("policy not yet calculated!")
            return 0
        
        # Q: f x s x o
        state = env.get_state()
        s_idx = env.state_to_idx(state)

        option = np.argmax(self.Q[s_idx])

        return option

# NO ACCESS TO TM HERE
# basically, it's model-free LOF where q-learning
# must be used to learn the high-level policy
# performance should be about the same as normal LOF,
# but this alg is not composable
class FSAQLearningMetaPolicy(DiscreteMetaPolicy):

    def __init__(self, subgoals, task_spec, safety_props, safety_specs, env,
                 num_episodes=1000, episode_length=100, gamma=1., alpha=0.5, epsilon=0.3,
                 record_training=False, recording_frequency=100, experiment_num=0):
        super().__init__(subgoals, task_spec, safety_props, safety_specs, env,
                         num_episodes, episode_length, gamma, alpha, epsilon,
                         record_training, recording_frequency, experiment_num)

        self.Q = self.high_level_q_learning(env, env.get_state())

    def q_learning(self, env):
        nO = len(self.options)

        initial_state = env.get_state()

        for i in range(self.num_episodes):
            random_state = self.get_random_state(env)
            env.set_state(random_state)
            
            option_index = i % nO # episodes loop thru options

            num_steps = 0
            # while not self.is_terminated(env, option_index) and num_steps < self.episode_length:
            while num_steps < self.episode_length:
                current_state = env.state_to_idx(env.get_state())
                action = self.get_epsilon_greed_action(env, option_index, current_state, self.epsilon)
                obs = env.step(action)
                next_state = env.state_to_idx(env.get_state())

                for option in self.options:
                    option.update_qfunction(current_state, env.action_dict[action], next_state,
                                            gamma=self.gamma, alpha=self.alpha)

                num_steps += 1

            self.record_results(i, num_steps, initial_state, env)
            
        env.set_state(initial_state)

    def high_level_q_learning(self, env, start_state, alpha=0.5, task_spec=None):
        if task_spec is None:
            task_spec = self.task_spec

        previous_start_state = env.get_state()

        nF = task_spec.nF

        Q = np.zeros((nF, self.ss_size, self.nO))
        V = np.zeros((nF, self.ss_size))

        num_episodes = 500
        episode_length = 10
        gamma = 1.
        epsilon = 0.4
        
        goal_state = task_spec.nF - 1
        for i in range(num_episodes):
            f = 0 # i % (nF - 1) # cycle through the FSA states except the goal state
            if i % 2 == 0:
                if task_spec.spec == '(F (c & F a) & G ! can) | (F a & F can) & G ! o':
                    f = 3
                if task_spec.spec == '(F((a|b) & F(c & F home)) & G ! can) | (F((a|b) & F home) & F can) & G ! o':
                    f = 2
            env.set_state(start_state)
            num_steps = 0
            while num_steps < episode_length:
                current_state = env.state_to_idx(env.get_state())

                if np.random.uniform() < epsilon:
                    option_index = np.random.choice(range(self.nO))
                else:
                    option_index = np.argmax(Q[f, current_state])

                next_state = np.argmax(self.poss[option_index][current_state].toarray())
                env.set_state(env.idx_to_state(next_state))
                # self.if_experiment_modification(env, f, 0, i)
                next_f = self.get_fsa_state(env, f, task_spec.tm)

                reward = (task_spec.task_state_costs[f]) * (self.options[option_index].reward[current_state]-1)

                q_update = reward + gamma * V[next_f, next_state] - Q[f, current_state, option_index]
                Q[f, current_state, option_index] += alpha * q_update
                V = np.max(Q, axis=2)

                f = next_f
                # print(f)
                num_steps += 1

        env.set_state(previous_start_state)

        return Q

    def evaluate_policy(self, env, episode_num, experiment_num=0, task_spec=None, Q=None):
        if task_spec is None:
            task_spec = self.task_spec

        starting_state = env.get_state()

        if Q is None:
            self.Q = self.high_level_q_learning(env, env.get_state(), task_spec=task_spec)
        else:
            self.Q = Q

        f = 0
        goal_state = task_spec.nF - 1
        max_steps_in_option = 30
        max_steps = 100

        reward = 0
        success = False
        num_steps = 0
        prev_option = 0
        f_prev = 0

        while num_steps < max_steps:
            if f_prev != f and not self.is_terminated(env, prev_option):
                print('redoing high-level Q. f_prev: {}, f: {}, prev_option: {}, state: {}'.format(f_prev, f, prev_option, env.state_to_idx(env.get_state())))
                self.Q = self.high_level_q_learning(env, env.get_state())
            option = self.get_option(env, f)
            prev_option = option
            f_prev = f
            steps_in_option = 0
            if self.is_terminated(env, option):
                num_steps += 1
                state_index = env.state_to_idx(env.get_state())
                this_reward = task_spec.task_state_costs[f]*self.options[option].R[state_index]
                if this_reward == 0:
                    this_reward = -1
                reward += this_reward
            while not self.is_terminated(env, option) and f_prev == f and steps_in_option < max_steps_in_option and num_steps < max_steps:
                state_index = env.state_to_idx(env.get_state())

                self.if_experiment_modification(env, f, experiment_num, episode_num, task_spec=task_spec)


                reward += task_spec.task_state_costs[f]*self.options[option].R[state_index]

                action = self.get_action(env, option)
                obs = env.step(action)
                f_prev = f
                f = self.get_fsa_state(env, f, task_spec.tm)
                steps_in_option += 1
                num_steps += 1

            if f == goal_state:
                success = True
                break
            # state_index = env.state_to_idx(env.get_state())
            # reward += self.task_spec.task_state_costs[f]*self.options[option].R[state_index]

        self.Q = None
        env.set_state(starting_state)

        return reward, success, f

    def make_and_evaluate_policy(self, env, task_spec, experiment_num, alpha=0.5, num_iter=300):
        if task_spec is None:
            task_spec = self.task_spec

        rewards = []
        successes = []
        fs = []

        previous_start_state = env.get_state()
        start_state = previous_start_state

        nF = task_spec.nF

        Q = np.zeros((nF, self.ss_size, self.nO))
        V = np.zeros((nF, self.ss_size))

        num_episodes = num_iter
        episode_length = 10
        gamma = 1.
        epsilon = 0.4
        
        goal_state = task_spec.nF - 1
        for i in range(num_episodes):
            f = 0 # i % (nF - 1) # cycle through the FSA states except the goal state
            if i % 2 == 0:
                if task_spec.spec == '(F (c & F a) & G ! can) | (F a & F can) & G ! o':
                    f = 3
                if task_spec.spec == '(F((a|b) & F(c & F home)) & G ! can) | (F((a|b) & F home) & F can) & G ! o':
                    f = 2
            env.set_state(start_state)
            num_steps = 0
            while num_steps < episode_length:
                current_state = env.state_to_idx(env.get_state())

                if np.random.uniform() < epsilon:
                    option_index = np.random.choice(range(self.nO))
                else:
                    option_index = np.argmax(Q[f, current_state])

                next_state = np.argmax(self.poss[option_index][current_state].toarray())
                env.set_state(env.idx_to_state(next_state))
                # self.if_experiment_modification(env, f, 0, i)
                next_f = self.get_fsa_state(env, f, task_spec.tm)

                reward = (task_spec.task_state_costs[f]) * (self.options[option_index].reward[current_state]-1)

                q_update = reward + gamma * V[next_f, next_state] - Q[f, current_state, option_index]
                Q[f, current_state, option_index] += alpha * q_update
                V = np.max(Q, axis=2)

                f = next_f
                # print(f)
                num_steps += 1
            
            env.set_state(start_state)
            reward, success, final_f = self.evaluate_policy(env, i, experiment_num=experiment_num, task_spec=task_spec, Q=Q)
            rewards.append(reward)
            successes.append(success)
            fs.append(final_f)

        env.set_state(previous_start_state)

        return rewards, successes, fs

# same as QLearningMetaPolicy except instead of using value iteration
# to find the shortest path thru the FSA, just greedily choose the next option
# with the highest reward (lowest cost)
class GreedyQLearningMetaPolicy(DiscreteMetaPolicy):

    def __init__(self, subgoals, task_spec, safety_props, safety_specs, env,
                 num_episodes=1000, episode_length=100, gamma=1., alpha=0.5, epsilon=0.3,
                 record_training=False, recording_frequency=100, experiment_num=0):
        super().__init__(subgoals, task_spec, safety_props, safety_specs, env,
                         num_episodes, episode_length, gamma, alpha, epsilon,
                         record_training, recording_frequency, experiment_num)

        # self.Q = self.make_policy(env, task_spec)

    def q_learning(self, env):
        nO = len(self.options)

        initial_state = env.get_state()

        for i in range(self.num_episodes):
            random_state = self.get_random_state(env)
            env.set_state(random_state)
            option_index = i % nO # episodes loop thru options

            num_steps = 0
            # while not self.is_terminated(env, option_index) and num_steps < self.episode_length:
            while num_steps < self.episode_length:
                current_state = env.state_to_idx(env.get_state())
                action = self.get_epsilon_greed_action(env, option_index, current_state, self.epsilon)
                obs = env.step(action)
                next_state = env.state_to_idx(env.get_state())

                for option in self.options:
                    option.update_qfunction(current_state, env.action_dict[action], next_state,
                    gamma=self.gamma, alpha=self.alpha)

                num_steps += 1
            
            self.option_rewards = self.make_option_rewards_function()

            self.record_results(i, num_steps, initial_state, env)

        env.set_state(initial_state)

    def get_option(self, env, f, task_spec=None):
        if task_spec is None:
            task_spec = self.task_spec

        state = env.get_state()
        s_idx = env.state_to_idx(state)

        tm = task_spec.tm[f]

        highest_reward = -np.inf
        highest_reward_option = 0
        for i, option in enumerate(self.options):
            # only pick among options that lead to new states,
            # not back to the current state
            if tm[f, i] != 1:
                reward = option.reward[s_idx]
                if reward > highest_reward:
                    highest_reward = reward
                    highest_reward_option = i

        return highest_reward_option

    def evaluate_policy(self, env, episode_num, experiment_num=0, task_spec=None):
        if task_spec is None:
            task_spec = self.task_spec

        starting_state = env.get_state()
        # self.Q = self.make_policy(env, self.task_spec)
        f = 0
        goal_state = task_spec.nF - 1
        max_steps_in_option = 30
        max_steps = 100

        reward = 0
        success = False
        num_steps = 0

        while num_steps < max_steps:
            option = self.get_option(env, f, task_spec=task_spec)
            f_prev = f
            steps_in_option = 0
            if self.is_terminated(env, option):
                num_steps += 1
                state_index = env.state_to_idx(env.get_state())
                this_reward = task_spec.task_state_costs[f]*self.options[option].R[state_index]
                if this_reward == 0:
                    this_reward = -1
                reward += this_reward
            while not self.is_terminated(env, option) and f_prev == f and steps_in_option < max_steps_in_option and num_steps < max_steps and f != goal_state:
                state_index = env.state_to_idx(env.get_state())

                self.if_experiment_modification(env, f, experiment_num, episode_num, task_spec=task_spec)

                reward += task_spec.task_state_costs[f]*self.options[option].R[state_index]

                action = self.get_action(env, option)
                obs = env.step(action)
                f_prev = f
                f = self.get_fsa_state(env, f, task_spec.tm)
                steps_in_option += 1
                num_steps += 1
                if f == 1:
                    if np.random.uniform() < 0.0:
                        env.prop_dict['canceled'].value = True
                    else:
                        env.prop_dict['canceled'].value = False
                else:
                    env.prop_dict['canceled'].value = False
            if f == goal_state:
                success = True
                break
            # reward += self.task_spec.task_state_costs[f]*self.options[option].R[state_index]

        # self.Q = None
        env.set_state(starting_state)

        return reward, success, f

    def make_and_evaluate_policy(self, env, task_spec, experiment_num, num_iter):
        rewards = []
        successes = []
        fs = []
        for i in range(num_iter):
            reward, success, f = self.evaluate_policy(env, i, experiment_num, task_spec=task_spec)
            rewards.append(reward)
            successes.append(success)
            fs.append(f)
        return rewards, successes, fs