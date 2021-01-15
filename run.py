# import torch.nn as nn
# from preset import ppo
# from all.experiments import run_experiment
# from all.environments import GymEnvironment
import supersuit as ss
from pettingzoo.butterfly import pistonball_v3
from pettingzoo.utils import save_observation

env = pistonball_v3.env(n_pistons=6, local_ratio=0.2, time_penalty=-0.1, continuous=True, random_drop=True, random_rotate=True, ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5, max_cycles=900)

#env = ss.frame_stack_v1(ss.normalize_obs_v0(ss.resize(ss.dtype_v0(ss.color_reduction_v0(env, mode='B'),'float32'), x_size=20, y_size=76)), 3)

env = ss.normalize_obs_v0(ss.resize(ss.dtype_v0(ss.color_reduction_v0(env, mode='B'),'float32'), x_size=20, y_size=76)), 3

save_observation(env, all_agents=True)