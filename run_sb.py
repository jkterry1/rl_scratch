from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2
from pettingzoo.butterfly import pistonball_v4
import supersuit as ss
from stable_baselines.common.callbacks import CheckpointCallback
from pathlib import Path

env = pistonball_v4.parallel_env()
env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v0(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 3)
env = ss.pettingzoo_env_to_vec_env_v0(env)
env = ss.concat_vec_envs_v0(env, 2, num_cpus=4, base_class='stable_baselines')


checkpoint_callback = CheckpointCallback(save_freq=400, save_path=str(Path.home())+'/policy_logs')
model = PPO2(CnnPolicy, env, verbose=2, gamma=.3, n_steps=75, ent_coef=.07, learning_rate=.0002, vf_coef=.5, max_grad_norm=.25, lam=.9, nminibatches=6, noptepochs=15, cliprange_vf=.15)
model.learn(total_timesteps=6000000, callback=checkpoint_callback)
