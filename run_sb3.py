from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2
from pettingzoo.butterfly import pistonball_v3
import supersuit as ss

env = pistonball_v3.parallel_env(n_pistons=20, local_ratio=0, time_penalty=-0.1, continuous=True, random_drop=True, random_rotate=True, ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5, max_cycles=125)
env = ss.color_reduction_v0(env, mode='B')
env = ss.dtype_v0(env, 'float32')
env = ss.resize_v0(env, x_size=84, y_size=84)
env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
env = ss.frame_stack_v1(env, 3)
env = ss.pettingzoo_env_to_vec_env_v0(env)
env = ss.concat_vec_envs_v0(env, 8, num_cpus=4, base_class='stable_baselines')

model = PPO2(CnnPolicy, env, verbose=3, gamma=0.99, n_steps=125, ent_coef=0.01, learning_rate=0.00025, vf_coef=0.5, max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=1)
model.learn(total_timesteps=2000000)  # half convergence time in my rllib tests



# model = PPO2(CnnPolicy, env, gamma=.3, n_steps=1, ent_coef=.07, learning_rate=.0002, vf_coef=.5, max_grad_norm=.25, lam=.9, nminibatches=500, noptepochs=15, cliprange_vf=.15, tensorboard_log=('/home/justin_terry/tensorboard_logs/' + name + '/'))

"""
SB rendering code
Turn on logging?


Hook into Tune
Get tune set to run on many GCP instances
Little search


Future:
ent coeff schedule
Orthogonal policy initialization
Check VF sharing is on
LSTMs/GRUs/etc
Adam annealing
KL penalty?
Remove unnecessary preprocessing
Policy compression/lazy frame stacking?
PPG
Policy architecture search

16 -> 3GB GPU, 4GB RAM
128-> maxed GPU, 10GB CPU


magic number:
n_steps = 0-125
nminibatches: 4-4096
n_envs = 2,4,6,8
gamma = .9-.999
ent_coef = 0-.25
lr = 0.003 to 5e-6
vf_coef=0.1 to 1
lam = .9-1
max_grad_norm = 0-1
cliprange_vf = 0-1


cliprange does nothing if you specify cliprange_vf

Multi-agent reinforcement learning in 14 lines of using (with PettingZoo)

Simple RLlib tutorial (general)
RLlib tutorial (chess)

Make them feel smart

"""
