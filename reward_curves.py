from stable_baselines import PPO2
from pettingzoo.butterfly import pistonball_v4
import supersuit as ss
import os
from pathlib import Path
import sys


def make_env(n_envs):
    if n_envs is None:
        env = pistonball_v4.env(time_penalty=-1)
    else:
        env = pistonball_v4.parallel_env(time_penalty=-1)
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v0(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)
    if n_envs is not None:
        env = ss.pettingzoo_env_to_vec_env_v0(env)
        env = ss.concat_vec_envs_v0(env, 2*n_envs, num_cpus=4, base_class='stable_baselines')
    return env


def evaluate_all_policies(name):

    #mean_rewards = []

    def evaluate_policy(env, model):
        total_reward = 0
        NUM_RESETS = 10
        for i in range(NUM_RESETS):
            env.reset()
            for agent in env.agent_iter():
                obs, reward, done, info = env.last()
                total_reward += reward
                act = model.predict(obs, deterministic=True)[0] if not done else None
                env.step(act)
        return total_reward/NUM_RESETS

    env = make_env(None)
    policy_folder = str(Path.home())+'/policy_logs/'+name+'/'
    policy_files = os.listdir(policy_folder)

    policy_file = sorted(policy_files, key=lambda x: int(x[9:-10]))[-1]

    model = PPO2.load(policy_folder+policy_file)

    for policy_file in policy_files:
        model = PPO2.load(policy_folder+policy_file)
        print(evaluate_policy(env, model))
    #max_reward = max(mean_rewards)
    #optimal_policy = policy_folder+policy_files[mean_rewards.index(max(mean_rewards))]
    #os.system('cp ' + optimal_policy + ' ' + policy_folder + 'name')
    #os.system('rsync ' + policy_folder + 'name' + ' ' + 'justin_terry@10.128.0.24:/home/justin_terry/policies')
    #os.system('rm ' + policy_folder + 'name')
    # rewards_path = str(Path.home())+'/reward_logs/'+name
    # with open(rewards_path+'.txt', 'w') as f:
    #     for reward in mean_rewards:
    #         f.write("%s\n" % reward)


evaluate_all_policies(sys.argv[1])
