import gymnasium as gym
import numpy as np

env = gym.make('HandReachDense-v1', render_mode='human')
obs = env.reset()

obs1 = env.step(env.action_space.sample())


obs_space_dims = env.observation_space

action_space_dims = env.action_space.shape
print(obs1[0])
