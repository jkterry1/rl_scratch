from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
#from pettingzoo.butterfly import pistonball_v4
import supersuit as ss
from ray import tune
from ray.tune.suggest.ax import AxSearch
from ax.service.ax_client import AxClient
import os
import ray
from pathlib import Path
import gym

ax = AxClient(enforce_sequential_optimization=False)
ax.create_experiment(
    name="mnist_experiment",
    parameters=[
        {"name": "gamma", "type": "range", "bounds": [.9, .999], "log_scale": True,  "value_type": 'float'},
        {"name": "n_steps", "type": "range", "bounds": [10, 300], "log_scale": False,  "value_type": 'int'},  # 125
        {"name": "ent_coef", "type": "range", "bounds": [.0001, .25], "log_scale": True,  "value_type": 'float'},
        {"name": "learning_rate", "type": "range", "bounds": [5e-6, .003], "log_scale": True,  "value_type": 'float'},
        {"name": "vf_coef", "type": "range", "bounds": [.1, 1], "log_scale": False,  "value_type": 'float'},
        {"name": "max_grad_norm", "type": "range", "bounds": [.01, 10], "log_scale": True,  "value_type": 'float'},
        {"name": "gae_lambda", "type": "range", "bounds": [.9, 1], "log_scale": False,  "value_type": 'float'},
        {"name": "n_epochs", "type": "range", "bounds": [3, 50], "log_scale": False,  "value_type": 'int'},
        {"name": "n_envs", "type": "range", "bounds": [1, 4], "log_scale": False,  "value_type": 'int'},
        {"name": "minibatch_scale", "type": "range", "bounds": [.015, .25], "log_scale": False,  "value_type": 'float'},
    ],
    objective_name="mean_reward",
    minimize=False,
)


def make_env(n_envs):
    if n_envs is None:
        #env = pistonball_v4.env(time_penalty=-1)
        env = gym.make('MountainCarContinuous-v0')
    else:
        #env = pistonball_v4.parallel_env(time_penalty=-1)
        env = gym.make('MountainCarContinuous-v0')
        env = ss.stable_baselines3_vec_env_v0(env, 2*n_envs, multiprocessing=False)

    # env = ss.color_reduction_v0(env, mode='B')
    # env = ss.resize_v0(env, x_size=84, y_size=84)
    # env = ss.frame_stack_v1(env, 3)
    # if n_envs is not None:
    #     env = ss.pettingzoo_env_to_vec_env_v0(env)
    #     env = ss.concat_vec_envs_v0(env, 2*n_envs, num_cpus=4, base_class='stable_baselines')

    return env


def evaluate_all_policies(name):

    def evaluate_policy(env, model):
        total_reward = 0
        NUM_RESETS = 100
        """
        for i in range(NUM_RESETS):
            env.reset()
            for agent in env.agent_iter():
                obs, reward, done, info = env.last()
                total_reward += reward
                act = model.predict(obs, deterministic=True)[0] if not done else None
                env.step(act)
            """
        for i in range(NUM_RESETS):
            done = False
            obs = env.reset()
            while not done:
                act = model.predict(obs, deterministic=True)[0] if not done else None
                observation, reward, done, info = env.step(act)
                total_reward += reward

        return total_reward/NUM_RESETS

    env = make_env(None)
    policy_folder = str(Path.home())+'/policy_logs/'+name+'/'
    policy_files = os.listdir(policy_folder)
    policy_file = sorted(policy_files, key=lambda x: int(x[9:-10]))[-1]
    model = PPO.load(policy_folder+policy_file)

    return evaluate_policy(env, model)


def gen_filename(params):
    name = ''
    keys = list(params.keys())

    for key in keys:
        name = name + key+'_'+str(params[key])[0:5]+'_'

    name = name[0:-1]  # removes trailing _
    return name.replace('.', '')


def name_siphon(trial):
    # https://github.com/ray-project/ray/blob/master/python/ray/tune/trial.py
    print(trial.evaluated_params)
    return trial.trial_id


def train(parameterization):
    name = gen_filename(parameterization)
    folder = str(Path.home())+'/policy_logs/'+name+'/'
    checkpoint_callback = CheckpointCallback(save_freq=400, save_path=folder)  # off by factor that I don't understand

    big_batch_size = 2*parameterization['n_envs']*parameterization['n_steps']  # missing factor of 20 for pistonball
    divisors = [i for i in range(1, int(big_batch_size*parameterization['minibatch_scale'])) if big_batch_size % i == 0]
    nminibatches = int(big_batch_size/divisors[-1])
    batch_size = int(big_batch_size / nminibatches)

    # batch_size = int(2*parameterization['n_envs']*parameterization['n_steps']/4)

    env = make_env(parameterization['n_envs'])
    # try:
    model = PPO("MlpPolicy", env, gamma=parameterization['gamma'], n_steps=parameterization['n_steps'], ent_coef=parameterization['ent_coef'], learning_rate=parameterization['learning_rate'], vf_coef=parameterization['vf_coef'], max_grad_norm=parameterization['max_grad_norm'], gae_lambda=parameterization['gae_lambda'], batch_size=batch_size, n_epochs=parameterization['n_epochs'], tensorboard_log=(str(Path.home())+'/tensorboard_logs/'+name+'/'))
    model.learn(total_timesteps=90000, callback=checkpoint_callback)  # time steps steps of each agent; was 4 million
    mean_reward = evaluate_all_policies(name)
    # except:
    #     mean_reward = -250
    tune.report(mean_reward=mean_reward)


ray.init(address='auto')


analysis = tune.run(
    train,
    num_samples=4,
    search_alg=AxSearch(ax_client=ax, max_concurrent=2, mode='max'),
    verbose=2,
    resources_per_trial={"gpu": 1, "cpu": 5},
    trial_name_creator=tune.function(name_siphon)
)


ax.save_to_json_file()


"""
ray start --head
nohup python3 killer_daemon.py &> killer_log.out &
nohup python3 run_hyperparameters.py &> tune_log.out &

Code upgrades:
Test things

stable baselines 3
supersuit -U
gym[box2d]
delete logs

Make batch size an actual hyperparameter
Knockknock
Parallelize evaluations
Add try mkdirs for everything in code or seperate script
unify log naming
Figure out GCP ssh key issue
Use old hyperparameters as seed (?)
Disable fail2ban
Constant n_envs?
Use local and remote machines (docker?)
Have head be GPUless VM so it cant get rebooted on maintenance
Automatically stop using GCP resources

FP16
NaN handling
https://docs.ray.io/en/master/tune/api_docs/suggestion.html#limiter (2.0)
Parallel env evaluations/rendering

https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/callbacks.py#L207

Limit number of gif renders at once (find faster option?)
Future RL Upgrades:
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

5GB of RAM and 1 core per render (pistonball), 2GB buffer ram, 4 extra CPU cores
"""
