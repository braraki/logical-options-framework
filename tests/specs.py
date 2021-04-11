import numpy as np
from simulator.options import *


# RMs need a slightly different Task Spec
def make_taskspec_delivery1_rm():
    # go to A or B, then C, then HOME, unless C is CANceled in which case just go to A or B then HOME
    spec = '(F((a|b) & F(c & F home)) & G ! can) | (F((a|b) & F home) & F can) & G ! o'

    # prop order:
    # a b c home can cana canb canc canh o e

    nF = 8
    nP = 11
    tm = np.zeros((nF, nF, nP))

    # S0
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  1  1  0  0  0  0  0  0  1
    # 1  1  1  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  1  0  0  1  1  0  0
    # 3  0  0  0  0  0  1  1  0  0  0  0
    # 4  0  0  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  0  0  0  0  0  0  0
    # T  0  0  0  0  0  0  0  0  0  1  0
    tm[0, 1, 0] = 1
    tm[0, 1, 1] = 1
    tm[0, 0, 2] = 1
    tm[0, 0, 3] = 1
    tm[0, 2, 4] = 1
    tm[0, 3, 5] = 1
    tm[0, 3, 6] = 1
    tm[0, 2, 7] = 1
    tm[0, 2, 8] = 1
    tm[0, 7, 9] = 1
    tm[0, 0, 10] = 1
    # S1
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  1  1  0  0  0  0  0  0  0  0  1
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  1  0  1  1  1  1  0  0  0
    # 4  0  0  0  1  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  0  0  0  0  1  0  0
    # T  0  0  0  0  0  0  0  0  0  1  0
    tm[1, 1, 0] = 1
    tm[1, 1, 1] = 1
    tm[1, 3, 2] = 1
    tm[1, 4, 3] = 1
    tm[1, 3, 4] = 1
    tm[1, 3, 5] = 1
    tm[1, 3, 6] = 1
    tm[1, 3, 7] = 1
    tm[1, 6, 8] = 1
    tm[1, 7, 9] = 1
    tm[1, 1, 10] = 1
    # S2
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  1  1  1  0  0  1  1  0  1
    # 3  1  1  0  0  0  1  1  0  0  0  0
    # 4  0  0  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  0  0  0  0  0  0  0
    # T  0  0  0  0  0  0  0  0  0  1  0
    tm[2, 3, 0] = 1
    tm[2, 3, 1] = 1
    tm[2, 2, 2] = 1
    tm[2, 2, 3] = 1
    tm[2, 2, 4] = 1
    tm[2, 3, 5] = 1
    tm[2, 3, 6] = 1
    tm[2, 2, 7] = 1
    tm[2, 2, 8] = 1
    tm[2, 7, 9] = 1
    tm[2, 2, 10] = 1
    # S3
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  1  1  1  0  1  1  1  1  0  0  1
    # 4  0  0  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  1  0  0  0  0  1  0  0
    # T  0  0  0  0  0  0  0  0  0  1  0
    tm[3, 3, 0] = 1
    tm[3, 3, 1] = 1
    tm[3, 3, 2] = 1
    tm[3, 6, 3] = 1
    tm[3, 3, 4] = 1
    tm[3, 3, 5] = 1
    tm[3, 3, 6] = 1
    tm[3, 3, 7] = 1
    tm[3, 6, 8] = 1
    tm[3, 7, 9] = 1
    tm[3, 3, 10] = 1
    # S4
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0  0
    # 4  1  1  0  1  0  0  0  0  0  0  1
    # 5  0  0  1  0  0  0  0  0  0  0  0
    # G  0  0  0  0  1  1  1  1  1  0  0
    # T  0  0  0  0  0  0  0  0  0  1  0
    tm[4, 4, 0] = 1
    tm[4, 4, 1] = 1
    tm[4, 5, 2] = 1
    tm[4, 4, 3] = 1
    tm[4, 6, 4] = 1
    tm[4, 6, 5] = 1
    tm[4, 6, 6] = 1
    tm[4, 6, 7] = 1
    tm[4, 6, 8] = 1
    tm[4, 7, 9] = 1
    tm[4, 4, 10] = 1
    # S5
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0  0
    # 4  0  0  0  0  0  0  0  0  0  0  0
    # 5  1  1  1  0  0  0  0  0  0  0  1
    # G  0  0  0  1  1  1  1  1  1  0  0
    # T  0  0  0  0  0  0  0  0  0  1  0
    tm[5, 5, 0] = 1
    tm[5, 5, 1] = 1
    tm[5, 5, 2] = 1
    tm[5, 6, 3] = 1
    tm[5, 6, 4] = 1
    tm[5, 6, 5] = 1
    tm[5, 6, 6] = 1
    tm[5, 6, 7] = 1
    tm[5, 6, 8] = 1
    tm[5, 7, 9] = 1
    tm[5, 5, 10] = 1
    # G
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0  0
    # 4  0  0  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0  0  0
    # G  1  1  1  1  1  1  1  1  1  1  1
    # T  0  0  0  0  0  0  0  0  0  0  0
    tm[6, 6, :] = 1
    # T
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0  0
    # 4  0  0  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  0  0  0  0  0  0  0
    # T  1  1  1  1  1  1  1  1  1  1  1
    tm[7, 7, :] = 1

    # remember that these are multiplicative
    task_state_costs = [-1, -1, -1, -1, -1, -1, 0, -1000]

    safety_props = [4, 5, 6, 7, 8, 9]
    task_spec = TaskSpec(spec, tm, task_state_costs)

    return task_spec, safety_props

# sequential task
# F(a & F (b & (F c & F h))) & G ! o
def make_taskspec_delivery2_rm():
    # go to A, then B, then C, then HOME
    spec = 'F(a & F (b & (F c & F h))) & G ! o'

    # prop order:
    # a b c home can cana canb canc canh o e

    nF = 6
    nP = 11
    tm = np.zeros((nF, nF, nP))

    # S0
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  1  1  1  1  0  1  1  1  0  1
    # 1  1  0  0  0  0  1  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  0  0  0  0  0  0  0
    # T  0  0  0  0  0  0  0  0  0  1  0
    tm[0, 1, 0] = 1
    tm[0, 0, 1] = 1
    tm[0, 0, 2] = 1
    tm[0, 0, 3] = 1
    tm[0, 0, 4] = 1
    tm[0, 1, 5] = 1
    tm[0, 0, 6] = 1
    tm[0, 0, 7] = 1
    tm[0, 0, 8] = 1
    tm[0, 5, 9] = 1
    tm[0, 0, 10] = 1
    # S1
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  1  0  1  1  1  1  0  1  1  0  1
    # 2  0  1  0  0  0  0  1  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  0  0  0  0  0  0  0
    # T  0  0  0  0  0  0  0  0  0  1  0
    tm[1, 1, 0] = 1
    tm[1, 2, 1] = 1
    tm[1, 1, 2] = 1
    tm[1, 1, 3] = 1
    tm[1, 1, 4] = 1
    tm[1, 1, 5] = 1
    tm[1, 2, 6] = 1
    tm[1, 1, 7] = 1
    tm[1, 1, 8] = 1
    tm[1, 5, 9] = 1
    tm[1, 1, 10] = 1
    # S2
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  1  1  0  1  1  1  1  0  1  0  1
    # 3  0  0  1  0  0  0  0  1  0  0  0
    # G  0  0  0  0  0  0  0  0  0  0  0
    # T  0  0  0  0  0  0  0  0  0  1  0
    tm[2, 2, 0] = 1
    tm[2, 2, 1] = 1
    tm[2, 3, 2] = 1
    tm[2, 2, 3] = 1
    tm[2, 2, 4] = 1
    tm[2, 2, 5] = 1
    tm[2, 2, 6] = 1
    tm[2, 3, 7] = 1
    tm[2, 2, 8] = 1
    tm[2, 5, 9] = 1
    tm[2, 2, 10] = 1
    # S3
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  1  1  1  0  1  1  1  1  0  0  1
    # G  0  0  0  1  0  0  0  0  1  0  0
    # T  0  0  0  0  0  0  0  0  0  1  0
    tm[3, 3, 0] = 1
    tm[3, 3, 1] = 1
    tm[3, 3, 2] = 1
    tm[3, 4, 3] = 1
    tm[3, 3, 4] = 1
    tm[3, 3, 5] = 1
    tm[3, 3, 6] = 1
    tm[3, 3, 7] = 1
    tm[3, 4, 8] = 1
    tm[3, 5, 9] = 1
    tm[3, 3, 10] = 1
    # G
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0  0
    # G  1  1  1  1  1  1  1  1  1  1  1
    tm[4, 4, :] = 1
    # T
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  0  0  0  0  0  0  0
    # T  1  1  1  1  1  1  1  1  1  1  1
    tm[5, 5, :] = 1

    # remember that these are multiplicative
    task_state_costs = [-1, -1, -1, -1, 0, -1000]

    safety_props = [4, 5, 6, 7, 8, 9]
    task_spec = TaskSpec(spec, tm, task_state_costs)

    return task_spec, safety_props

# OR task
# F ((a | b) & F c) & G ! o
def make_taskspec_delivery3_rm():
    # go to A, then B, then C, then HOME
    spec = 'F ((a | b) & F c) & G ! o'

    # prop order:
    # a b c home can cana canb canc canh o e

    nF = 4
    nP = 11
    tm = np.zeros((nF, nF, nP))

    # S0
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  1  1  1  0  0  1  1  0  1
    # 1  1  1  0  0  0  1  1  0  0  0  0
    # G  0  0  0  0  0  0  0  0  0  0  0
    # T  0  0  0  0  0  0  0  0  0  1  0
    tm[0, 1, 0] = 1
    tm[0, 1, 1] = 1
    tm[0, 0, 2] = 1
    tm[0, 0, 3] = 1
    tm[0, 0, 4] = 1
    tm[0, 1, 5] = 1
    tm[0, 1, 6] = 1
    tm[0, 0, 7] = 1
    tm[0, 0, 8] = 1
    tm[0, 3, 9] = 1
    tm[0, 0, 10] = 1
    # S1
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  1  1  0  1  1  1  1  0  1  0  1
    # G  0  0  1  0  0  0  0  1  0  0  0
    # T  0  0  0  0  0  0  0  0  0  1  0
    tm[1, 1, 0] = 1
    tm[1, 1, 1] = 1
    tm[1, 2, 2] = 1
    tm[1, 1, 3] = 1
    tm[1, 1, 4] = 1
    tm[1, 1, 5] = 1
    tm[1, 1, 6] = 1
    tm[1, 2, 7] = 1
    tm[1, 1, 8] = 1
    tm[1, 3, 9] = 1
    tm[1, 1, 10] = 1
    # G
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # G  1  1  1  1  1  1  1  1  1  1  1
    # T  0  0  0  0  0  0  0  0  0  0  0
    tm[2, 2, :] = 1
    # T
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  0  0  0  0  0  0  0
    # T  1  1  1  1  1  1  1  1  1  1  1
    tm[3, 3, :] = 1

    # remember that these are multiplicative
    task_state_costs = [-1, -1, 0, -1000]

    safety_props = [4, 5, 6, 7, 8, 9]
    task_spec = TaskSpec(spec, tm, task_state_costs)

    return task_spec, safety_props

# IF task
# (F (c & F a) & G ! can) | (F a & F can)
def make_taskspec_delivery4_rm():
    # go to A, then B, then C, then HOME
    spec = '(F (c & F a) & G ! can) | (F a & F can) & G ! o'

    # prop order:
    # a b c home can cana canb canc canh o e

    nF = 6
    nP = 11
    tm = np.zeros((nF, nF, nP))

    # S0
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  1  0  1  0  0  0  0  0  0  1
    # 1  1  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  1  0  1  0  1  1  1  0  0
    # G  0  0  0  0  0  1  0  0  0  0  0
    # T  0  0  0  0  0  0  0  0  0  1  0
    tm[0, 1, 0] = 1
    tm[0, 0, 1] = 1
    tm[0, 3, 2] = 1
    tm[0, 0, 3] = 1
    tm[0, 3, 4] = 1
    tm[0, 4, 5] = 1
    tm[0, 3, 6] = 1
    tm[0, 3, 7] = 1
    tm[0, 3, 8] = 1
    tm[0, 5, 9] = 1
    tm[0, 0, 10] = 1
    # S1
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  1  1  0  1  0  0  0  0  0  0  1
    # 2  0  0  1  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  1  1  1  1  1  0  0
    # T  0  0  0  0  0  0  0  0  0  1  0
    tm[1, 1, 0] = 1
    tm[1, 1, 1] = 1
    tm[1, 2, 2] = 1
    tm[1, 1, 3] = 1
    tm[1, 4, 4] = 1
    tm[1, 4, 5] = 1
    tm[1, 4, 6] = 1
    tm[1, 4, 7] = 1
    tm[1, 4, 8] = 1
    tm[1, 5, 9] = 1
    tm[1, 1, 10] = 1
    # S2
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  1  1  1  0  0  0  0  0  0  1
    # 3  0  0  0  0  0  0  0  0  0  0  0
    # G  1  0  0  0  1  1  1  1  1  0  0
    # T  0  0  0  0  0  0  0  0  0  1  0
    tm[2, 4, 0] = 1
    tm[2, 2, 1] = 1
    tm[2, 2, 2] = 1
    tm[2, 2, 3] = 1
    tm[2, 4, 4] = 1
    tm[2, 4, 5] = 1
    tm[2, 4, 6] = 1
    tm[2, 4, 7] = 1
    tm[2, 4, 8] = 1
    tm[2, 5, 9] = 1
    tm[2, 2, 10] = 1
    # S3
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  0  1  1  1  1  0  1  1  1  0  1
    # G  1  0  0  0  0  1  0  0  0  0  0
    # T  0  0  0  0  0  0  0  0  0  1  0
    tm[3, 4, 0] = 1
    tm[3, 3, 1] = 1
    tm[3, 3, 2] = 1
    tm[3, 3, 3] = 1
    tm[3, 3, 4] = 1
    tm[3, 4, 5] = 1
    tm[3, 3, 6] = 1
    tm[3, 3, 7] = 1
    tm[3, 3, 8] = 1
    tm[3, 5, 9] = 1
    tm[3, 3, 10] = 1
    # G
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0  0
    # G  1  1  1  1  1  1  1  1  1  1  1
    tm[4, 4, :] = 1
    # T
    tm[5, 5, :] = 1

    # remember that these are multiplicative
    task_state_costs = [-1, -1, -1, -1, 0, -1000]

    safety_props = [4, 5, 6, 7, 8, 9]
    task_spec = TaskSpec(spec, tm, task_state_costs)

    return task_spec, safety_props

def make_subgoals_delivery(env):
    initial_state = env.get_state()
    P = env.make_prop_map()
    env.set_state(initial_state)

    def get_subgoal_index(idx, alt_idx):
        if not np.all(P[idx] == 0):
            return np.argmax(P[idx])
        else:
            return np.argmax(P[alt_idx])

    a_index = get_subgoal_index(0, 5)
    b_index = get_subgoal_index(1, 6)
    c_index = get_subgoal_index(2, 7)
    home_index = get_subgoal_index(3, 8)

    subgoal_a = Subgoal('a', 0, 0, a_index)
    subgoal_b = Subgoal('b', 1, 1, b_index)
    subgoal_c = Subgoal('c', 2, 2, c_index)
    subgoal_home = Subgoal('home', 3, 3, home_index)

    return [subgoal_a, subgoal_b, subgoal_c, subgoal_home]

def make_taskspec_delivery1():
    # go to A or B, then C, then HOME, unless C is CANceled in which case just go to A or B then HOME
    spec = '(F((a|b) & F(c & F home)) & G ! can) | (F((a|b) & F home) & F can) & G ! o'

    # prop order:
    # a b c home can cana canb canc canh o e

    nF = 7
    nP = 11
    tm = np.zeros((nF, nF, nP))

    # S0
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  1  1  0  0  0  0  0  0  1
    # 1  1  1  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  1  0  0  1  1  0  0
    # 3  0  0  0  0  0  1  1  0  0  0  0
    # 4  0  0  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  0  0  0  0  0  0  0
    tm[0, 1, 0] = 1
    tm[0, 1, 1] = 1
    tm[0, 0, 2] = 1
    tm[0, 0, 3] = 1
    tm[0, 2, 4] = 1
    tm[0, 3, 5] = 1
    tm[0, 3, 6] = 1
    tm[0, 2, 7] = 1
    tm[0, 2, 8] = 1
    tm[0, 0, 10] = 1
    # S1
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  1  1  0  0  0  0  0  0  0  0  1
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  1  0  1  1  1  1  0  0  0
    # 4  0  0  0  1  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  0  0  0  0  1  0  0
    tm[1, 1, 0] = 1
    tm[1, 1, 1] = 1
    tm[1, 3, 2] = 1
    tm[1, 4, 3] = 1
    tm[1, 3, 4] = 1
    tm[1, 3, 5] = 1
    tm[1, 3, 6] = 1
    tm[1, 3, 7] = 1
    tm[1, 6, 8] = 1
    tm[1, 1, 10] = 1
    # S2
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  1  1  1  0  0  1  1  0  1
    # 3  1  1  0  0  0  1  1  0  0  0  0
    # 4  0  0  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  0  0  0  0  0  0  0
    tm[2, 3, 0] = 1
    tm[2, 3, 1] = 1
    tm[2, 2, 2] = 1
    tm[2, 2, 3] = 1
    tm[2, 2, 4] = 1
    tm[2, 3, 5] = 1
    tm[2, 3, 6] = 1
    tm[2, 2, 7] = 1
    tm[2, 2, 8] = 1
    tm[2, 2, 10] = 1
    # S3
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  1  1  1  0  1  1  1  1  0  0  1
    # 4  0  0  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  1  0  0  0  0  1  0  0
    tm[3, 3, 0] = 1
    tm[3, 3, 1] = 1
    tm[3, 3, 2] = 1
    tm[3, 6, 3] = 1
    tm[3, 3, 4] = 1
    tm[3, 3, 5] = 1
    tm[3, 3, 6] = 1
    tm[3, 3, 7] = 1
    tm[3, 6, 8] = 1
    tm[3, 3, 10] = 1
    # S4
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0  0
    # 4  1  1  0  1  0  0  0  0  0  0  1
    # 5  0  0  1  0  0  0  0  0  0  0  0
    # G  0  0  0  0  1  1  1  1  1  0  0
    tm[4, 4, 0] = 1
    tm[4, 4, 1] = 1
    tm[4, 5, 2] = 1
    tm[4, 4, 3] = 1
    tm[4, 6, 4] = 1
    tm[4, 6, 5] = 1
    tm[4, 6, 6] = 1
    tm[4, 6, 7] = 1
    tm[4, 6, 8] = 1
    tm[4, 4, 10] = 1
    # S5
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0  0
    # 4  0  0  0  0  0  0  0  0  0  0  0
    # 5  1  1  1  0  0  0  0  0  0  0  1
    # G  0  0  0  1  1  1  1  1  1  0  0
    tm[5, 5, 0] = 1
    tm[5, 5, 1] = 1
    tm[5, 5, 2] = 1
    tm[5, 6, 3] = 1
    tm[5, 6, 4] = 1
    tm[5, 6, 5] = 1
    tm[5, 6, 6] = 1
    tm[5, 6, 7] = 1
    tm[5, 6, 8] = 1
    tm[5, 5, 10] = 1
    # G
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0  0
    # 4  0  0  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0  0  0
    # G  1  1  1  1  1  1  1  1  1  1  1
    tm[6, 6, :] = 1

    # remember that these are multiplicative
    task_state_costs = [1, 1, 1, 1, 1, 1, 0]

    safety_props = [4, 5, 6, 7, 8, 9]
    task_spec = TaskSpec(spec, tm, task_state_costs)

    return task_spec, safety_props

# sequential task
# F(a & F (b & (F c & F h))) & G ! o
def make_taskspec_delivery2():
    # go to A, then B, then C, then HOME
    spec = 'F(a & F (b & (F c & F h))) & G ! o'

    # prop order:
    # a b c home can cana canb canc canh o e

    nF = 5
    nP = 11
    tm = np.zeros((nF, nF, nP))

    # S0
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  1  1  1  1  0  1  1  1  0  1
    # 1  1  0  0  0  0  1  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  0  0  0  0  0  0  0
    tm[0, 1, 0] = 1
    tm[0, 0, 1] = 1
    tm[0, 0, 2] = 1
    tm[0, 0, 3] = 1
    tm[0, 0, 4] = 1
    tm[0, 1, 5] = 1
    tm[0, 0, 6] = 1
    tm[0, 0, 7] = 1
    tm[0, 0, 8] = 1
    tm[0, 0, 10] = 1
    # S1
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  1  0  1  1  1  1  0  1  1  0  1
    # 2  0  1  0  0  0  0  1  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  0  0  0  0  0  0  0
    tm[1, 1, 0] = 1
    tm[1, 2, 1] = 1
    tm[1, 1, 2] = 1
    tm[1, 1, 3] = 1
    tm[1, 1, 4] = 1
    tm[1, 1, 5] = 1
    tm[1, 2, 6] = 1
    tm[1, 1, 7] = 1
    tm[1, 1, 8] = 1
    tm[1, 1, 10] = 1
    # S2
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  1  1  0  1  1  1  1  0  1  0  1
    # 3  0  0  1  0  0  0  0  1  0  0  0
    # G  0  0  0  0  0  0  0  0  0  0  0
    tm[2, 2, 0] = 1
    tm[2, 2, 1] = 1
    tm[2, 3, 2] = 1
    tm[2, 2, 3] = 1
    tm[2, 2, 4] = 1
    tm[2, 2, 5] = 1
    tm[2, 2, 6] = 1
    tm[2, 3, 7] = 1
    tm[2, 2, 8] = 1
    tm[2, 2, 10] = 1
    # S3
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  1  1  1  0  1  1  1  1  0  0  1
    # G  0  0  0  1  0  0  0  0  1  0  0
    tm[3, 3, 0] = 1
    tm[3, 3, 1] = 1
    tm[3, 3, 2] = 1
    tm[3, 4, 3] = 1
    tm[3, 3, 4] = 1
    tm[3, 3, 5] = 1
    tm[3, 3, 6] = 1
    tm[3, 3, 7] = 1
    tm[3, 4, 8] = 1
    tm[3, 3, 10] = 1
    # G
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0  0
    # G  1  1  1  1  1  1  1  1  1  1  1
    tm[4, 4, :] = 1

    # remember that these are multiplicative
    task_state_costs = [1, 1, 1, 1, 0]

    safety_props = [4, 5, 6, 7, 8, 9]
    task_spec = TaskSpec(spec, tm, task_state_costs)

    return task_spec, safety_props

# OR task
# F ((a | b) & F c) & G ! o
def make_taskspec_delivery3():
    # go to A, then B, then C, then HOME
    spec = 'F ((a | b) & F c) & G ! o'

    # prop order:
    # a b c home can cana canb canc canh o e

    nF = 3
    nP = 11
    tm = np.zeros((nF, nF, nP))

    # S0
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  1  1  1  0  0  1  1  0  1
    # 1  1  1  0  0  0  1  1  0  0  0  0
    # G  0  0  0  0  0  0  0  0  0  0  0
    tm[0, 1, 0] = 1
    tm[0, 1, 1] = 1
    tm[0, 0, 2] = 1
    tm[0, 0, 3] = 1
    tm[0, 0, 4] = 1
    tm[0, 1, 5] = 1
    tm[0, 1, 6] = 1
    tm[0, 0, 7] = 1
    tm[0, 0, 8] = 1
    tm[0, 0, 10] = 1
    # S1
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  1  1  0  1  1  1  1  0  1  0  1
    # G  0  0  1  0  0  0  0  1  0  0  0
    tm[1, 1, 0] = 1
    tm[1, 1, 1] = 1
    tm[1, 2, 2] = 1
    tm[1, 1, 3] = 1
    tm[1, 1, 4] = 1
    tm[1, 1, 5] = 1
    tm[1, 1, 6] = 1
    tm[1, 2, 7] = 1
    tm[1, 1, 8] = 1
    tm[1, 1, 10] = 1
    # G
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # G  1  1  1  1  1  1  1  1  1  1  1
    tm[2, 2, :] = 1

    # remember that these are multiplicative
    task_state_costs = [1, 1, 0]

    safety_props = [4, 5, 6, 7, 8, 9]
    task_spec = TaskSpec(spec, tm, task_state_costs)

    return task_spec, safety_props

# IF task
# (F (c & F a) & G ! can) | (F a & F can)
def make_taskspec_delivery4():
    # go to A, then B, then C, then HOME
    spec = '(F (c & F a) & G ! can) | (F a & F can) & G ! o'

    # prop order:
    # a b c home can cana canb canc canh o e

    nF = 5
    nP = 11
    tm = np.zeros((nF, nF, nP))

    # S0
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  1  0  1  0  0  0  0  0  0  1
    # 1  1  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  1  0  1  0  1  1  1  0  0
    # G  0  0  0  0  0  1  0  0  0  0  0
    tm[0, 1, 0] = 1
    tm[0, 0, 1] = 1
    tm[0, 3, 2] = 1
    tm[0, 0, 3] = 1
    tm[0, 3, 4] = 1
    tm[0, 4, 5] = 1
    tm[0, 3, 6] = 1
    tm[0, 3, 7] = 1
    tm[0, 3, 8] = 1
    tm[0, 0, 10] = 1
    # S1
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  1  1  0  1  0  0  0  0  0  0  1
    # 2  0  0  1  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  1  1  1  1  1  0  0
    tm[1, 1, 0] = 1
    tm[1, 1, 1] = 1
    tm[1, 2, 2] = 1
    tm[1, 1, 3] = 1
    tm[1, 4, 4] = 1
    tm[1, 4, 5] = 1
    tm[1, 4, 6] = 1
    tm[1, 4, 7] = 1
    tm[1, 4, 8] = 1
    tm[1, 1, 10] = 1
    # S2
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  1  1  1  0  0  0  0  0  0  1
    # 3  0  0  0  0  0  0  0  0  0  0  0
    # G  1  0  0  0  1  1  1  1  1  0  0
    tm[2, 4, 0] = 1
    tm[2, 2, 1] = 1
    tm[2, 2, 2] = 1
    tm[2, 2, 3] = 1
    tm[2, 4, 4] = 1
    tm[2, 4, 5] = 1
    tm[2, 4, 6] = 1
    tm[2, 4, 7] = 1
    tm[2, 4, 8] = 1
    tm[2, 2, 10] = 1
    # S3
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  0  1  1  1  1  0  1  1  1  0  1
    # G  1  0  0  0  0  1  0  0  0  0  0
    tm[3, 4, 0] = 1
    tm[3, 3, 1] = 1
    tm[3, 3, 2] = 1
    tm[3, 3, 3] = 1
    tm[3, 3, 4] = 1
    tm[3, 4, 5] = 1
    tm[3, 3, 6] = 1
    tm[3, 3, 7] = 1
    tm[3, 3, 8] = 1
    tm[3, 3, 10] = 1
    # G
    #    a  b  c  h  c ca cb cc ch  o  e
    # 0  0  0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0  0  0
    # G  1  1  1  1  1  1  1  1  1  1  1
    tm[4, 4, :] = 1

    # remember that these are multiplicative
    task_state_costs = [1, 1, 1, 1, 0]

    safety_props = [4, 5, 6, 7, 8, 9]
    task_spec = TaskSpec(spec, tm, task_state_costs)

    return task_spec, safety_props

def make_safetyspecs_delivery():

    ##### SAFETY SPEC A #####

    spec_a = 'G!o & Fa'
    nP = 7

    nF_a = 3
    tm_a = np.zeros((nF_a, nF_a, nP))
    # S1
    tm_a[0, 1, 0] = 1
    tm_a[0, 0, 1] = 1
    tm_a[0, 0, 2] = 1
    tm_a[0, 0, 3] = 1
    tm_a[0, 0, 4] = 1
    tm_a[0, 2, 5] = 1
    tm_a[0, 0, 6] = 1
    # G
    tm_a[1, 1, :] = 1
    # T
    tm_a[2, 2, :] = 1

    # multiplicative       a   b   c   h   c  ca  cb  cc  ch     o   e
    safety_prop_costs_a = [0, -1, -1, -1, -1,  0, -1, -1, -1, -1000, -1]

    safety_spec_a = SafetySpec('a', spec_a, tm_a, safety_prop_costs_a)

    ##### SAFETY SPEC B #####
    spec_b = 'G!o & Fb'
    nF_b = 3
    tm_b = np.zeros((nF_b, nF_b, nP))
    tm_b[0, 0, 0] = 1
    tm_b[0, 1, 1] = 1
    tm_b[0, 0, 2] = 1
    tm_b[0, 0, 3] = 1
    tm_b[0, 0, 4] = 1
    tm_b[0, 2, 5] = 1
    tm_b[0, 0, 6] = 1

    safety_prop_costs_b = [-1, 0, -1, -1, -1, -1, 0, -1, -1, -1000, -1]

    safety_spec_b = SafetySpec('b', spec_b, tm_b, safety_prop_costs_b)

    ##### SAFETY SPEC C #####
    spec_c = 'G!o & Fc'
    nF_c = 3
    tm_c = np.zeros((nF_c, nF_c, nP))
    tm_c[0, 0, 0] = 1
    tm_c[0, 0, 1] = 1
    tm_c[0, 1, 2] = 1
    tm_c[0, 0, 3] = 1
    tm_c[0, 0, 4] = 1
    tm_c[0, 2, 5] = 1
    tm_c[0, 0, 6] = 1

    safety_prop_costs_c = [-1, -1, 0, -1, -1, -1, -1, 0, -1, -1000, -1]

    safety_spec_c = SafetySpec('c', spec_c, tm_c, safety_prop_costs_c)

    ##### SAFETY SPEC H #####
    spec_h = 'G!o & Fh'
    nF_h = 3
    tm_h = np.zeros((nF_h, nF_h, nP))
    tm_h[0, 0, 0] = 1
    tm_h[0, 0, 1] = 1
    tm_h[0, 0, 2] = 1
    tm_h[0, 1, 3] = 1
    tm_h[0, 0, 4] = 1
    tm_h[0, 2, 5] = 1
    tm_h[0, 0, 6] = 1

    safety_prop_costs_h = [-1, -1, -1, 0, -1, -1, -1, -1, 0, -1000, -1]

    safety_spec_h = SafetySpec('h', spec_h, tm_h, safety_prop_costs_h)

    safety_specs = [safety_spec_a, safety_spec_b, safety_spec_c, safety_spec_h]

    return safety_specs