from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2
from stable_baselines.common.callbacks import CheckpointCallback
from pettingzoo.butterfly import pistonball_v4
import supersuit as ss
import random
import string
from ray import tune
from ray.tune.suggest.ax import AxSearch
from ax.service.ax_client import AxClient
import os

ax = AxClient(enforce_sequential_optimization=False)
ax.create_experiment(
    name="mnist_experiment",
    parameters=[
        {"name": "gamma", "type": "range", "bounds": [.9, .99], "log_scale": True,  "value_type": 'float'},
        {"name": "n_steps", "type": "range", "bounds": [10, 125], "log_scale": False,  "value_type": 'int'},
        {"name": "ent_coef", "type": "range", "bounds": [.0001, .25], "log_scale": True,  "value_type": 'float'},
        {"name": "learning_rate", "type": "range", "bounds": [5e-6, .003], "log_scale": True,  "value_type": 'float'},
        {"name": "vf_coef", "type": "range", "bounds": [.1, 1], "log_scale": False,  "value_type": 'float'},
        {"name": "max_grad_norm", "type": "range", "bounds": [0, 1], "log_scale": False,  "value_type": 'float'},
        {"name": "lam", "type": "range", "bounds": [.9, 1], "log_scale": False,  "value_type": 'float'},
        {"name": "minibatch_scale", "type": "range", "bounds": [.015, .25], "log_scale": False,  "value_type": 'float'},
        {"name": "noptepochs", "type": "range", "bounds": [3, 50], "log_scale": False,  "value_type": 'int'},
        {"name": "cliprange_vf", "type": "range", "bounds": [0, 1], "log_scale": False,  "value_type": 'float'},
        {"name": "n_envs", "type": "range", "bounds": [1, 4], "log_scale": False,  "value_type": 'int'},
    ],
    objective_name="mean_reward",
    minimize=False,
)


def make_env(n_envs):
    if n_envs is None:
        env = pistonball_v4.env()
    else:
        env = pistonball_v4.parallel_env()
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v0(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)
    if n_envs is not None:
        env = ss.pettingzoo_env_to_vec_env_v0(env)
        env = ss.concat_vec_envs_v0(env, 2*n_envs, num_cpus=4, base_class='stable_baselines')
    return env


def evaluate_all_policies(folder):
    env = make_env(None)
    mean_reward = []

    def evaluate_policy(env, model):
        total_reward = 0
        NUM_RESETS = 5
        for i in range(NUM_RESETS):
            env.reset()
            for agent in env.agent_iter():
                obs, reward, done, info = env.last()
                total_reward += reward
                act = model.predict(obs) if not done else None
                env.step(act)
        return total_reward/NUM_RESETS

    policy_files = os.listdir(folder)

    for policy_file in policy_files:
        model = PPO2.load(folder+policy_file)
        mean_reward.append(evaluate_policy(env, model))

    return max(mean_reward)


def train(parameterization):
    letters = string.ascii_lowercase
    folder = ''.join(random.choice(letters) for i in range(10))
    folder = '~/logs/'+folder+'/'
    os.makedirs(folder)
    checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=folder)

    batch_size = 20*2*parameterization['n_envs']*parameterization['n_steps']
    divisors = [i for i in range(1, int(batch_size*parameterization['minibatch_scale'])) if batch_size % i == 0]
    nminibatches = int(batch_size/divisors[-1])

    env = make_env(parameterization['n_envs'])
    model = PPO2(CnnPolicy, env, gamma=parameterization['gamma'], n_steps=parameterization['n_steps'], ent_coef=parameterization['ent_coef'], learning_rate=parameterization['learning_rate'], vf_coef=parameterization['vf_coef'], max_grad_norm=parameterization['max_grad_norm'], lam=parameterization['lam'], nminibatches=nminibatches, noptepochs=parameterization['noptepochs'], cliprange_vf=parameterization['cliprange_vf'])
    #model.learn(total_timesteps=2000000, callback=checkpoint_callback)
    model.learn(total_timesteps=10000, callback=checkpoint_callback)
    mean_reward = evaluate_all_policies(folder)
    tune.report(negative_mean_reward=mean_reward)


analysis = tune.run(
    train,
    num_samples=4,
    search_alg=AxSearch(ax_client=ax, mode="max"),
    verbose=2,
    resources_per_trial={"gpu": 1, "cpu": 5},
)


ax.save_to_json_file()


"""
Single run:
Make sure ax saving works
Make sure logging gives me everything I want
Make sure the reported optimal hyperparameters are in fact optimal
See if run lengths are what the should be

Double run:
Get to work/Make sure nothing crashes

Future upgrades:
Better obs space rescaling
ent coeff schedule
Orthogonal policy initialization
Check VF sharing is on
LSTMs/GRUs/etc (remove frame stacking?)
Adam annealing
KL penalty?
Remove unnecessary preprocessing
Policy compression/lazy frame stacking?
PPG
Policy architecture in search
Penalize cranking up instability in search
Early termination in search?
Parallelize final policy evaluations?
dont save policies to save time saving to disk?
Incentivize learning faster?
DIAYN
"""
