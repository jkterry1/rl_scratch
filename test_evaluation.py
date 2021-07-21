from stable_baselines3 import PPO
from pettingzoo.butterfly import cooperative_pong_v2
import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
import numpy as np

n_evaluations = 20
n_agents = 2
n_envs = 4
n_timesteps = 4000000

env = cooperative_pong_v2.parallel_env()
player1 = env.possible_agents[0]


def invert_agent_indication(obs, agent):
    if len(obs.shape) == 2:
        obs = obs.reshape(obs.shape+(1,))
    obs2 = obs if agent == player1 else 255-obs
    return np.concatenate([obs, obs2], axis=2)


def image_transpose(env):
    if is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space):
        env = VecTransposeImage(env)
    return env


env = cooperative_pong_v2.parallel_env()
env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v0(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 3)
env = ss.observation_lambda_v0(env, invert_agent_indication)
env = ss.pettingzoo_env_to_vec_env_v0(env)
env = ss.concat_vec_envs_v0(env, n_envs, num_cpus=1, base_class='stable_baselines3')
env = VecMonitor(env)
env = image_transpose(env)

eval_env = cooperative_pong_v2.parallel_env()
eval_env = ss.color_reduction_v0(eval_env, mode='B')
eval_env = ss.resize_v0(eval_env, x_size=84, y_size=84)
eval_env = ss.frame_stack_v1(eval_env, 3)
eval_env = ss.observation_lambda_v0(eval_env, invert_agent_indication)
eval_env = ss.pettingzoo_env_to_vec_env_v0(eval_env)
eval_env = ss.concat_vec_envs_v0(eval_env, 1, num_cpus=1, base_class='stable_baselines3')
eval_env = VecMonitor(eval_env)
eval_env = image_transpose(eval_env)

eval_freq = int(n_timesteps / n_evaluations)
eval_freq = max(eval_freq // (n_envs*n_agents), 1)

model = PPO("CnnPolicy", env, verbose=3)
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=eval_freq, deterministic=True, render=False)
model.learn(total_timesteps=n_timesteps, callback=eval_callback)
model.save("policy_optimal")

model = PPO.load("policy_optimal")

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

print(mean_reward)
print(std_reward)
