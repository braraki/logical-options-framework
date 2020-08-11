import time
from simulator.rendering import Viewer
from simulator.balldrop import BallDropSim
from simulator.lineworld import LineWorldSim
from simulator.gridworld import GridWorldSim
from simulator.delivery import DeliverySim
from simulator.driveworld import DriveWorldSim
from simulator.options import *
from celluloid import Camera

# human: show plots
# anim: don't show plots but save a gif
# note: the first call to 'render' sets the render mode
render_mode = 'anim'

# 'vi', 'lvi', 'hardcoded', 'options'
policy_mode = 'options'

def make_subgoals_gridworld(env):
    initial_state = env.get_state()
    P = env.make_prop_map()
    env.set_state(initial_state)

    a_index = np.argmax(P[0])
    b_index = np.argmax(P[1])
    c_index = np.argmax(P[2])

    subgoal_a = Subgoal('a', 0, 0, a_index)
    subgoal_b = Subgoal('b', 1, 1, b_index)
    subgoal_c = Subgoal('c', 2, 2, c_index)

    return [subgoal_a, subgoal_b, subgoal_c]

def make_taskspec_gridworld():
    # go to goal A, then B, then A
    spec = 'F (a & F (b & F a))) | F c'

    # prop order:
    # goal_a, goal_b

    nF = 4
    nP = 5
    tm = np.zeros((nF, nF, nP))

    # initial state
    #   a b c o e
    # 0 0 1 0 0 1
    # 1 1 0 0 0 0
    # 2 0 0 0 0 0
    # 3 0 0 1 0 0
    tm[0, 1, 0] = 1
    tm[0, 0, 1] = 1
    tm[0, 3, 2] = 1
    tm[0, 0, 3] = 1
    tm[0, 0, 4] = 1
    # S1
    #   a b c o e
    # 0 0 0 0 0 0
    # 1 1 0 1 0 1
    # 2 0 1 0 0 0
    # 3 0 0 0 0 0
    # 4 0 0 0 1 0
    tm[1, 1, 0] = 1
    tm[1, 2, 1] = 1
    tm[1, 1, 2] = 1
    tm[1, 1, 3] = 1
    tm[1, 1, 4] = 1
    # S2
    #   a b c o e
    # 0 0 0 0 0 0
    # 1 0 0 0 0 0
    # 2 0 1 0 0 1
    # 3 1 0 1 0 0
    # 4 0 0 0 1 0
    tm[2, 3, 0] = 1
    tm[2, 2, 1] = 1
    tm[2, 2, 2] = 1
    tm[2, 2, 3] = 1
    tm[2, 2, 4] = 1
    # G
    #   a b c o e
    # 0 0 0 0 0 0
    # 1 0 0 0 0 0
    # 2 0 0 0 0 0
    # 3 1 1 1 1 1
    tm[3, 3, :] = 1

    # remember that these are multiplicative
    task_state_costs = [1, 1, 1, 0]

    safety_props = [3]
    task_spec = TaskSpec(spec, tm, task_state_costs)

    return task_spec, safety_props

def make_safetyspecs_gridworld():

    ##### SAFETY SPEC A #####

    spec_a = 'G!o & Fa'
    nP = 5

    nF_a = 3
    tm_a = np.zeros((nF_a, nF_a, nP))
    # S1
    tm_a[0, 1, 0] = 1
    tm_a[0, 0, 1] = 1
    tm_a[0, 0, 2] = 1
    tm_a[0, 2, 3] = 1
    tm_a[0, 0, 4] = 1
    # G
    tm_a[1, 1, :] = 1
    # T
    tm_a[2, 2, :] = 1

    # multiplicative       a   b   c      o   e
    safety_prop_costs_a = [0, -1, -1, -1000, -1]

    safety_spec_a = SafetySpec('a', spec_a, tm_a, safety_prop_costs_a)

    ##### SAFETY SPEC B #####
    spec_b = 'G!o & Fb'
    nF_b = 3
    tm_b = np.zeros((nF_b, nF_b, nP))
    tm_b[0, 0, 0] = 1
    tm_b[0, 1, 1] = 1
    tm_b[0, 0, 2] = 1
    tm_b[0, 2, 3] = 1
    tm_b[0, 0, 4] = 1

    safety_prop_costs_b = [-1, 0, -1, -1000, -1]

    safety_spec_b = SafetySpec('b', spec_b, tm_b, safety_prop_costs_b)

    ##### SAFETY SPEC C #####
    spec_c = 'G!o & Fc'
    nF_c = 3
    tm_c = np.zeros((nF_c, nF_c, nP))
    tm_c[0, 0, 0] = 1
    tm_c[0, 0, 1] = 1
    tm_c[0, 1, 2] = 1
    tm_c[0, 2, 3] = 1
    tm_c[0, 0, 4] = 1

    safety_prop_costs_c = [-1, -1, 0, -1000, -1]

    safety_spec_c = SafetySpec('c', spec_c, tm_c, safety_prop_costs_c)

    safety_specs = [safety_spec_a, safety_spec_b, safety_spec_c]

    return safety_specs

def make_subgoals_delivery(env):
    initial_state = env.get_state()
    P = env.make_prop_map()
    env.set_state(initial_state)

    def get_subgoal_index(idx, alt_idx):
        if not np.all(P[idx] == 0):
            return np.argmax(P[idx])
        else:
            return np.argmax(P[alt_idx])

    a_index = get_subgoal_index(0, 4)
    b_index = get_subgoal_index(1, 5)
    home_index = get_subgoal_index(2, 6)

    subgoal_a = Subgoal('a', 0, 0, a_index)
    subgoal_b = Subgoal('b', 1, 1, b_index)
    subgoal_home = Subgoal('home', 2, 2, home_index)

    return [subgoal_a, subgoal_b, subgoal_home]

def make_taskspec_delivery():
    # go to goal A, then B, then A
    spec = '(F(a &F(b & F home)) & G ! can) | (F(a & F home) & F can) & G ! o'

    # prop order:
    # a b home can o e

    nF = 7
    nP = 9
    tm = np.zeros((nF, nF, nP))

    # S0
    #    a  b  h  c ca cb ch  o  e
    # 0  0  1  1  0  0  0  0  0  1
    # 1  1  0  0  0  0  0  0  0  0
    # 2  0  0  0  1  0  1  1  0  0
    # 3  0  0  0  0  1  0  0  0  0
    # 4  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  0  0  0  0  0
    tm[0, 1, 0] = 1
    tm[0, 0, 1] = 1
    tm[0, 0, 2] = 1
    tm[0, 2, 3] = 1
    tm[0, 3, 4] = 1
    tm[0, 2, 5] = 1
    tm[0, 2, 6] = 1
    tm[0, 0, 8] = 1
    # S1
    #    a  b  h  c ca cb ch  o  e
    # 0  0  0  0  0  0  0  0  0  0
    # 1  1  0  0  0  0  0  0  0  1
    # 2  0  0  0  0  0  0  0  0  0
    # 3  0  1  0  1  1  1  0  0  0
    # 4  0  0  1  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  0  0  1  0  0
    tm[1, 1, 0] = 1
    tm[1, 3, 1] = 1
    tm[1, 4, 2] = 1
    tm[1, 3, 3] = 1
    tm[1, 3, 4] = 1
    tm[1, 3, 5] = 1
    tm[1, 6, 6] = 1
    tm[1, 1, 8] = 1
    # S2
    #    a  b  h  c ca cb ch  o  e
    # 0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0
    # 2  0  1  1  1  0  1  1  0  1
    # 3  1  0  0  0  1  0  0  0  0
    # 4  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0
    # G  0  0  0  0  0  0  0  0  0
    tm[2, 3, 0] = 1
    tm[2, 2, 1] = 1
    tm[2, 2, 2] = 1
    tm[2, 2, 3] = 1
    tm[2, 3, 4] = 1
    tm[2, 2, 5] = 1
    tm[2, 2, 6] = 1
    tm[2, 2, 8] = 1
    # S3
    #    a  b  h  c ca cb ch  o  e
    # 0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0
    # 3  1  1  0  1  1  1  0  0  1
    # 4  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0
    # G  0  0  1  0  0  0  1  0  0
    tm[3, 3, 0] = 1
    tm[3, 3, 1] = 1
    tm[3, 6, 2] = 1
    tm[3, 3, 3] = 1
    tm[3, 3, 4] = 1
    tm[3, 3, 5] = 1
    tm[3, 6, 6] = 1
    tm[3, 3, 8] = 1
    # S4
    #    a  b  h  c ca cb ch  o  e
    # 0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0
    # 4  1  0  1  0  0  0  0  0  1
    # 5  0  1  0  0  0  0  0  0  0
    # G  0  0  0  1  1  1  1  0  0
    tm[4, 4, 0] = 1
    tm[4, 5, 1] = 1
    tm[4, 4, 2] = 1
    tm[4, 6, 3] = 1
    tm[4, 6, 4] = 1
    tm[4, 6, 5] = 1
    tm[4, 6, 6] = 1
    tm[4, 4, 8] = 1
    # S5
    #    a  b  h  c ca cb ch  o  e
    # 0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0
    # 4  0  0  0  0  0  0  0  0  0
    # 5  1  1  0  0  0  0  0  0  1
    # G  0  0  1  1  1  1  1  0  0
    tm[5, 5, 0] = 1
    tm[5, 5, 1] = 1
    tm[5, 6, 2] = 1
    tm[5, 6, 3] = 1
    tm[5, 6, 4] = 1
    tm[5, 6, 5] = 1
    tm[5, 6, 6] = 1
    tm[5, 5, 8] = 1
    # G
    #    a  b  h  c ca cb ch  o  e
    # 0  0  0  0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0  0  0  0
    # 2  0  0  0  0  0  0  0  0  0
    # 3  0  0  0  0  0  0  0  0  0
    # 4  0  0  0  0  0  0  0  0  0
    # 5  0  0  0  0  0  0  0  0  0
    # G  1  1  1  1  1  1  1  1  1
    tm[6, 6, :] = 1

    # remember that these are multiplicative
    task_state_costs = [1, 1, 1, 1, 1, 1, 0]

    safety_props = [3, 4, 5, 6, 7]
    task_spec = TaskSpec(spec, tm, task_state_costs)

    return task_spec, safety_props

def make_safetyspecs_delivery():

    ##### SAFETY SPEC A #####

    spec_a = 'G!o & Fa'
    nP = 6

    nF_a = 3
    tm_a = np.zeros((nF_a, nF_a, nP))
    # S1
    tm_a[0, 1, 0] = 1
    tm_a[0, 0, 1] = 1
    tm_a[0, 0, 2] = 1
    tm_a[0, 0, 3] = 1
    tm_a[0, 2, 4] = 1
    tm_a[0, 0, 5] = 1
    # G
    tm_a[1, 1, :] = 1
    # T
    tm_a[2, 2, :] = 1

    # multiplicative       a   b   h   c  ca  cb  ch     o   e
    safety_prop_costs_a = [0, -1, -1, -1,  0, -1, -1, -1000, -1]

    safety_spec_a = SafetySpec('a', spec_a, tm_a, safety_prop_costs_a)

    ##### SAFETY SPEC B #####
    spec_b = 'G!o & Fb'
    nF_b = 3
    tm_b = np.zeros((nF_b, nF_b, nP))
    tm_b[0, 0, 0] = 1
    tm_b[0, 1, 1] = 1
    tm_b[0, 0, 2] = 1
    tm_b[0, 0, 3] = 1
    tm_b[0, 2, 4] = 1
    tm_b[0, 0, 5] = 1

    safety_prop_costs_b = [-1, 0, -1, -1, -1, 0, -1, -1000, -1]

    safety_spec_b = SafetySpec('b', spec_b, tm_b, safety_prop_costs_b)

    ##### SAFETY SPEC C #####
    spec_h = 'G!o & Fh'
    nF_h = 3
    tm_h = np.zeros((nF_h, nF_h, nP))
    tm_h[0, 0, 0] = 1
    tm_h[0, 0, 1] = 1
    tm_h[0, 1, 2] = 1
    tm_h[0, 0, 3] = 1
    tm_h[0, 2, 4] = 1
    tm_h[0, 0, 5] = 1

    safety_prop_costs_h = [-1, -1, 0, -1, -1, -1, 0, -1000, -1]

    safety_spec_h = SafetySpec('h', spec_h, tm_h, safety_prop_costs_h)

    safety_specs = [safety_spec_a, safety_spec_b, safety_spec_h]

    return safety_specs

def make_subgoals_driveworld(env):
    initial_state = env.get_state()
    P = env.make_prop_map()
    env.set_state(initial_state)

    def get_subgoal_index(idx, alt_idx):
        if not np.all(P[idx] == 0):
            return np.argmax(P[idx])
        else:
            return np.argmax(P[alt_idx])

    o_index = get_subgoal_index(0, 0)
    s_index = get_subgoal_index(1, 1)
    c_index = get_subgoal_index(2, 2)

    subgoal_o = Subgoal('go', 0, 0, o_index)
    subgoal_s = Subgoal('gs', 1, 1, s_index)
    subgoal_c = Subgoal('gc', 2, 2, c_index)

    return [subgoal_o, subgoal_s, subgoal_c]

def make_taskspec_driveworld():

    spec = 'F(gs|gc|go) & G( !obstacle & (s => !l) & (c => Fl) & (o => F!l) )'

    nF = 2
    nP = 6
    tm = np.zeros((nF, nF, nP))

    # initial state
    #   go gs gc  l ob  e
    # 0  0  0  0  1  1  1
    # 1  1  1  1  0  0  0
    tm[0, 1, :3] = 1 # go, gs, gc
    tm[0, 0, 3:] = 1 # l, ob, e
    # G
    #   o  s  c  l ob  e
    # 0 0  0  0  0  0  0
    # 1 1  1  1  1  1  1
    tm[1, 1, :] = 1

    # remember that these are multiplicative
    task_state_costs = [1, 0]

    safety_props = [3, 4]
    task_spec = TaskSpec(spec, tm, task_state_costs)

    return task_spec, safety_props

def make_safetyspecs_driveworld():

    nP = 6

    ##### SAFETY SPEC OVERTAKING #####

    spec_o = '(F!l & !o) U go'
    nF_o = 4
    tm_o = np.zeros((nF_o, nF_o, nP))
    # S0
    #   go gs gc  l ob  e
    # 0  0  1  1  0  0  1
    # 1  0  0  0  1  0  0
    # 2  1  0  0  0  0  0
    # 3  0  0  0  0  1  0
    tm_o[0, 2, 0] = 1
    tm_o[0, 0, 1] = 1
    tm_o[0, 0, 2] = 1
    tm_o[0, 1, 3] = 1
    tm_o[0, 3, 4] = 1
    tm_o[0, 0, 5] = 1
    # S1
    #   go gs gc  l ob  e
    # 0  0  0  0  0  0  1
    # 1  0  1  1  1  0  0
    # 2  1  0  0  0  0  0
    # 3  0  0  0  0  1  0
    tm_o[1, 2, 0] = 1
    tm_o[1, 1, 1] = 1
    tm_o[1, 1, 2] = 1
    tm_o[1, 1, 3] = 1
    tm_o[1, 3, 4] = 1
    tm_o[1, 0, 5] = 1
    # G
    #   go gs gc  l ob  e
    # 0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0
    # 2  1  1  1  1  1  1
    # 3  0  0  0  0  0  0
    tm_o[2, 2, :] = 1
    # T
    #   go gs gc  l ob  e
    # 0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0
    # 2  0  0  0  0  0  0
    # 3  1  1  1  1  1  1
    tm_o[3, 3, :] = 1

    # multiplicative       go  gs  gc   l      o   e
    safety_prop_costs_o = [ 0, -1, -2, -2, -1000, -1]

    safety_spec_o = SafetySpec('go', spec_o, tm_o, safety_prop_costs_o)

    ##### SAFETY SPEC STRAIGHT #####

    spec_s = '(!s & !o) U gs'
    nF_s = 3
    tm_s = np.zeros((nF_s, nF_s, nP))
    # S0
    #   go gs gc  l ob  e
    # 0  1  0  1  0  0  1
    # 1  0  1  0  0  0  0
    # 2  0  0  0  1  1  0
    tm_s[0, 0, 0] = 1
    tm_s[0, 1, 1] = 1
    tm_s[0, 0, 2] = 1
    tm_s[0, 2, 3] = 1
    tm_s[0, 2, 4] = 1
    tm_s[0, 0, 5] = 1
    # S1
    #   go gs gc  l ob  e
    # 0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0
    # 2  1  1  1  1  1  1
    # 3  0  0  0  0  0  0
    tm_s[1, 1, :] = 1
    # G
    #   go gs gc  l ob  e
    # 0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0
    # 2  0  0  0  0  0  0
    # 3  1  1  1  1  1  1
    tm_s[2, 2, :] = 1

    # multiplicative       go  gs     gc      l      o   e
    safety_prop_costs_s = [-1,  0, -1000, -1000, -1000, -1]

    safety_spec_s = SafetySpec('gs', spec_s, tm_s, safety_prop_costs_s)

    
    ##### SAFETY SPEC CHANGE LANE #####

    spec_c = '(F l & !o) U gc'
    nF_c = 4
    tm_c = np.zeros((nF_c, nF_c, nP))
    # S0 [in the left lane]
    #   go gs gc  l ob  e
    # 0  0  0  0  1  0  0
    # 1  1  1  0  0  0  1
    # 2  0  0  1  0  0  0
    # 3  0  0  0  0  1  0
    tm_c[0, 1, 0] = 1
    tm_c[0, 1, 1] = 1
    tm_c[0, 2, 2] = 1
    tm_c[0, 0, 3] = 1
    tm_c[0, 3, 4] = 1
    tm_c[0, 1, 5] = 1
    # S1 [out of the left lane]
    #   go gs gc  l ob  e
    # 0  0  0  0  1  0  0
    # 1  1  1  0  0  0  1
    # 2  0  0  1  0  0  0
    # 3  0  0  0  0  1  0
    tm_c[1, 1, 0] = 1
    tm_c[1, 1, 1] = 1
    tm_c[1, 2, 2] = 1
    tm_c[1, 0, 3] = 1
    tm_c[1, 3, 4] = 1
    tm_c[1, 1, 5] = 1
    # G
    #   go gs gc  l ob  e
    # 0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0
    # 2  1  1  1  1  1  1
    # 3  0  0  0  0  0  0
    tm_c[2, 2, :] = 1
    # T
    #   go gs gc  l ob  e
    # 0  0  0  0  0  0  0
    # 1  0  0  0  0  0  0
    # 2  0  0  0  0  0  0
    # 3  1  1  1  1  1  1
    tm_c[3, 3, :] = 1

    # multiplicative       go  gs  gc   l      o   e
    safety_prop_costs_c = [-2, -2,  0, -1, -1000, -2]

    safety_spec_c = SafetySpec('gc', spec_c, tm_c, safety_prop_costs_c)

    safety_specs = [safety_spec_o, safety_spec_s, safety_spec_c]

    return safety_specs

def test_options(sim, task_spec=None):
    sim.reset()

    task_spec, safety_props = make_taskspec_delivery()
    safety_specs = make_safetyspecs_delivery()
    subgoals = make_subgoals_delivery(sim.env)


    policy = VIMetaPolicy(subgoals, task_spec, safety_props, safety_specs, sim.env)
    # policy.make_policy(sim.env, task_spec)
    f = 0
    goal_state = 6
    max_steps_in_option = 30

    for i in range(10):
        sim.render(mode=render_mode)
        camera = sim.render()
        option = policy.get_option(sim.env, f)
        f_prev = f
        steps_in_option = 0
        while not policy.is_terminated(sim.env, option) and f_prev == f and steps_in_option < max_steps_in_option:
            action = policy.get_action(sim.env, option)
            obs = sim.step(action)
            f_prev = f
            f = policy.get_fsa_state(sim.env, f)
            print("option: {} | FSA state: {}".format(option, f))
            camera = sim.render()
            steps_in_option += 1
            if f == 1:
                if np.random.uniform() < 0.0:
                    sim.env.prop_dict['canceled'].value = True
                else:
                    sim.env.prop_dict['canceled'].value = False
            else:
                sim.env.prop_dict['canceled'].value = False
        if f == goal_state:
            break
        

    if render_mode == 'anim':
        animation = camera.animate()
        animation.save(sim.env.name + '_metapolicy.gif', writer='imagemagick')
    return 0

def test_options_driveworld(sim, task_spec=None):
    sim.reset()

    task_spec, safety_props = make_taskspec_driveworld()
    safety_specs = make_safetyspecs_driveworld()
    subgoals = make_subgoals_driveworld(sim.env)


    policy = VIMetaPolicy(subgoals, task_spec, safety_props, safety_specs, sim.env)
    # policy.make_policy(sim.env, task_spec)
    f = 0
    goal_state = 1
    max_steps_in_option = 30

    for i in range(3):
        sim.render(mode=render_mode)
        camera = sim.render()
        option = policy.get_option(sim.env, f)
        f_prev = f
        steps_in_option = 0
        while not policy.is_terminated(sim.env, option) and f_prev == f and steps_in_option < max_steps_in_option:
            action = policy.get_action(sim.env, option)
            obs = sim.step(action)
            f_prev = f
            f = policy.get_fsa_state(sim.env, f)
            print("option: {} | FSA state: {}".format(option, f))
            camera = sim.render()
            steps_in_option += 1
        if f == goal_state:
            break
        

    if render_mode == 'anim':
        animation = camera.animate()
        animation.save(sim.env.name + '_metapolicy.gif', writer='imagemagick')
    return 0

def test_rrtoptions_driveworld(sim, task_spec=None):
    sim.reset()

    task_spec, safety_props = make_taskspec_driveworld()
    safety_specs = make_safetyspecs_driveworld()
    subgoals = make_subgoals_driveworld(sim.env)


    policy = RRTMetaPolicy(subgoals, task_spec, safety_props, safety_specs, sim.env)
    # policy.make_policy(sim.env, task_spec)
    options = policy.rrt_policies
    f = 0
    goal_state = 1
    max_steps_in_option = 400

    for i in range(4):
        sim.render(mode=render_mode)
        # camera = sim.render()
        option = policy.get_option(sim.env, f)
        state = tuple(sim.env.get_state())
        sim.env.option_start = state
        path = options[option][state]
        other_paths = []
        for i in range(len(subgoals)):
            if i != option:
                other_paths.append(options[i][state])
        f_prev = f
        steps_in_option = 0
        camera = sim.render_rrt(path, other_paths)
        print("option: {}, state: {}, goal: {}, path: {}".format(option, sim.env.get_state(), path[-1], path))
        while not policy.is_terminated(sim.env, option) and f_prev == f and steps_in_option < max_steps_in_option:
            # steps_in_option is fed to get_action to indicate how far
            # along the path the agent is
            action = policy.get_action(sim.env, option, steps_in_option)
            print(action)
            obs = sim.step(action)
            # s_idx = sim.env.state_to_idx(sim.env.get_state())
            # path = all_paths[option].rrt_policy[s_idx]
            # print("option: {}, state: {}, s_idx: {}, g_idx: {}".format(option, sim.env.get_state(), s_idx, sim.env.state_to_idx(path[-1])))
            f_prev = f
            f = policy.get_fsa_state(sim.env, f)
            print("option: {} | FSA state: {}".format(option, f))
            camera = sim.render_rrt(path, other_paths)
            # camera = sim.render()
            steps_in_option += 1
        if f == goal_state:
            break
        

    if render_mode == 'anim':
        animation = camera.animate()
        animation.save(sim.env.name + '_rrtmetapolicy.gif', writer='imagemagick')
    return 0

if __name__ == '__main__':
    # sim = BallDropSim()
    # tm = make_tm_balldrop()

    # sim = LineWorldSim()
    # tm = make_tm_lineworld()

    sim = DeliverySim()
    test_options(sim)

    # sim = DriveWorldSim()
    # test_rrtoptions_driveworld(sim)