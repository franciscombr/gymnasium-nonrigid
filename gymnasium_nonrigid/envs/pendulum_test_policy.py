import sys
sys.path.append('../spinningup/')
from spinup.utils.test_policy import load_policy_and_env, run_policy

import gymnasium as gym

env = gym.make('InvertedPendulum-v4',render_mode = 'human')

_, get_action = load_policy_and_env('/home/francisco/Desktop/DIGI2/gymnasium_tests/gymnasium-nonrigid/gymnasium_nonrigid/envs/path/to/output_dir/')


run_policy(env,get_action)