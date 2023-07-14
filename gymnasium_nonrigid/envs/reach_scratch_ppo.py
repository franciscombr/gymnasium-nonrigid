import sys
sys.path.append('../spinningup/')
from spinup.algos.pytorch.ppo.ppo import  ppo

import torch

import gymnasium as gym

class ObservationSpaceWrapper(gym.ObservationWrapper):
    def __init__(self, env, new_observation_space):
        super().__init__(env)
        self.observation_space = new_observation_space

    def observation(self, observation):
        # Modify the observation here according to your requirements
        #print(observation)
        obs_p_obj = observation['observation']
        #print(obs_p_obj)
        return obs_p_obj

env = gym.make('HandReachDense-v1')


new_obs_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(63,))
env_fn = ObservationSpaceWrapper(env, new_obs_space)


ac_kwargs = dict(hidden_sizes=[64,64], activation=torch.nn.ReLU)

logger_kwargs = dict(output_dir='path/to/output_dir', exp_name='experiment_name')

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=300, logger_kwargs=logger_kwargs)