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

from .options import *

class RewardMachinePolicy(OptionBase):

    def __init__(self, task_spec, ss_size, nA, alpha=0.5):
        # subgoal is the index of the subgoal prop
        self.task_spec = task_spec

        self.alpha = alpha
        self.ss_size = ss_size
        self.nA = nA
        V, Q = self.init_vq(self.ss_size, nA)
        self.policy = Q
        self.V = V

        # self.R = [0...nF]
        self.R = task_spec.task_state_costs

    def get_action(self, env):
        if self.policy is None:
            print("policy not yet created")
            return 0
        
        state = env.get_state()
        s_idx = env.state_to_idx(state)
        
        action = np.argmax(self.policy[s_idx, :])

        for a_n, a in env.action_dict.items():
            if a == action:
                action_name = a_n

        return action_name

    def init_vq(self, ss_size, nA):
        V = np.zeros((ss_size,))

        Q = np.zeros((ss_size, nA))
        return V, Q

class RewardMachineMetaPolicy(MetaPolicyBase):
    # there are NO SAFETY SPECS - everything must be embodied in the task_spec
    def __init__(self, subgoals, task_spec, safety_props, env,
                 num_episodes=1000, episode_length=100, gamma=1., alpha=0.5, epsilon=0.3,
                 record_training=False, recording_frequency=20):
        super().__init__(subgoals, task_spec, safety_props, None, env,
                         num_episodes, episode_length, gamma, alpha, epsilon)

        self.record_training = record_training
        self.recording_frequency = recording_frequency
        if record_training:
            self.training_steps = []
            self.training_reward = []
            self.training_success = []
            self.training_last_state = []

        # state space size
        self.ss_size = np.prod(env.get_full_state_space())
        # number of actions
        self.nA = len(env.action_dict)

        self.R = task_spec.task_state_costs

        self.policies = self.init_policies(task_spec)
        self.q_learning(env)

    def init_policies(self, task_spec):
        policies = []
        for i in range(task_spec.nF):
            policies.append(RewardMachinePolicy(self.task_spec, self.ss_size, self.nA))
        return policies

    def get_epsilon_greed_action(self, env, policy_index, state_index, epsilon):
        if np.random.uniform() < epsilon:
            action_index = np.random.choice(range(self.nA))
        else:
            action_index = np.argmax(self.policies[policy_index].policy[state_index])

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

    def record_results(self, i, num_steps, initial_state, env):
        if i % self.recording_frequency == 0:
            env.set_state(initial_state)
            episode_reward, success, final_f = self.evaluate_policy(env)
            if self.record_training:
                training_steps = i * self.episode_length + (num_steps - 1)
                self.training_reward.append(episode_reward)
                self.training_steps.append(training_steps)
                self.training_success.append(success)
                self.training_last_state.append(final_f)
            print("Episode: {}\t| Reward: {}\t| Success: {}".format(i, episode_reward, success))

    def get_results(self):
        if self.record_training:
            return {'reward': self.training_reward, 'steps': self.training_steps,
                    'success': self.training_success, 'last_state': self.training_last_state}
        else:
            return None

    def q_learning(self, env):
        num_policies = self.task_spec.nF

        initial_state = env.get_state()

        for i in range(self.num_episodes):
            random_state = self.get_random_state(env)
            env.set_state(random_state)
            
            subpolicy_index = i % num_policies # episodes loop thru FSA states and there is one subpolicy per FSA state

            num_steps = 0
            while num_steps < self.episode_length:
                
                current_state = env.state_to_idx(env.get_state())
                action = self.get_epsilon_greed_action(env, subpolicy_index, current_state, self.epsilon)
                action_index = env.action_dict[action]
                obs = env.step(action)
                next_state = env.state_to_idx(env.get_state())

                for current_f, policy in enumerate(self.policies):
                    next_f = self.get_fsa_state(env, current_f)

                    reward = self.R[next_f]
                    next_policy = self.policies[next_f]
                    q_update = reward + self.gamma * (next_policy.V[next_state] - policy.policy[current_state, action_index])
                    policy.policy[current_state, action_index] += self.alpha * (q_update)

                    policy.V = np.max(policy.policy, axis=1)

                num_steps += 1

            self.record_results(i, num_steps, initial_state, env)

        env.set_state(initial_state)

    def evaluate_policy(self, env):
        f = 0
        goal_state = self.task_spec.nF - 2
        trap_state = self.task_spec.nF - 1

        reward = 0
        success = False

        for i in range(100):
            reward += self.task_spec.task_state_costs[f]
            action = self.get_action(env, f)
            obs = env.step(action)
            f = self.get_fsa_state(env, f)

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
            elif f == trap_state:
                break
            # reward += self.task_spec.task_state_costs[f]

        return reward, success, f

    def get_action(self, env, f):
        return self.policies[f].get_action(env)

    def get_fsa_state(self, env, f):
        p = env.get_proposition()
        next_f = np.argmax(self.tm[f, :, p])
        
        return next_f