from pettingzoo.butterfly import pistonball_v4
import supersuit as ss
import os
from stable_baselines import PPO2
from array2gif import write_gif
import numpy as np
from pathlib import Path

env = pistonball_v4.env()
env = pistonball_v4.parallel_env()
env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v0(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 3)


def evaluate_all_policies(folder):
    mean_reward = []

    def evaluate_policy(env, model):
        total_reward = 0
        NUM_RESETS = 5
        for i in range(NUM_RESETS):
            env.reset()
            for agent in env.agent_iter():
                obs, reward, done, info = env.last()
                total_reward += reward
                act = model.predict(obs, deterministic=True)[0] if not done else None
                env.step(act)
        return total_reward/NUM_RESETS

    print(folder)
    policy_files = os.listdir(folder)
    print(policy_files)

    for policy_file in policy_files:
        model = PPO2.load(folder+policy_file)
        mean_reward.append(evaluate_policy(env, model))

    return folder+policy_files[mean_reward.index(max(mean_reward))]


def generate_gif(path, folder):
    reward_sum = 0
    obs_list = []
    i = 0
    model = PPO2.load(path)
    env.reset()

    while True:
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            reward_sum += reward
            action = model.predict(observation, deterministic=True)[0] if not done else None

            env.step(action)
            i += 1
            if i % (len(env.possible_agents)+1) == 0:
                obs_list.append(np.transpose(env.render(mode='rgb_array'), axes=(1, 0, 2)))
        env.close()
        break

    # print(reward)
    write_gif(obs_list, str(Path.home())+'/gifs/'+folder+'.gif', fps=15)


for folder in os.listdir('/home/justin_terry/good_policies'):
    generate_gif(evaluate_all_policies('/home/justin_terry/good_policies'+folder), folder)

"""
is the FPS for the gif right?
"""
