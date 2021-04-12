from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
#from pettingzoo.butterfly import pistonball_v4
import supersuit as ss
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch
import optuna
import os
import ray
from pathlib import Path
import gym
from ray.tune.suggest import ConcurrencyLimiter

space = {
    "ent_coef": optuna.distributions.LogUniformDistribution(.001, .1),
}


"""
Changes:
clip range- 0,1
Don't readd vf_coeff
amax grad norm- [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]
"""


optuna_search = OptunaSearch(
    space,
    metric="mean_reward",
    mode="max")


def make_env(n_envs):
    if n_envs is None:
        #env = pistonball_v4.env(time_penalty=-1)
        env = gym.make('LunarLanderContinuous-v2')
    else:
        #env = pistonball_v4.parallel_env(time_penalty=-1)
        env = gym.make('LunarLanderContinuous-v2')
        env = ss.stable_baselines3_vec_env_v0(env, n_envs, multiprocessing=False)

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
        name = name+key+'_'+str(params[key])[0:5]+'_'

    name = name[0:-1]  # removes trailing _
    return name.replace('.', '')


def train(parameterization):
    name = gen_filename(parameterization)
    folder = str(Path.home())+'/policy_logs/'+name+'/'
    checkpoint_callback = CheckpointCallback(save_freq=400, save_path=folder)  # off by factor that I don't understand

    env = make_env(8)
    # try:
    model = PPO("MlpPolicy", env, gamma=.99, n_steps=1024, ent_coef=parameterization['ent_coef'], batch_size=128, tensorboard_log=(str(Path.home())+'/tensorboard_logs/'+name+'/'), policy_kwargs={"net_arch": [256, 256]})
    model.learn(total_timesteps=3000000, callback=checkpoint_callback)  # time steps steps of each agent; was 4 million

    mean_reward = evaluate_all_policies(name)
    # except:
    #     mean_reward = -250
    tune.report(mean_reward=mean_reward)


ray.init(address='auto')

analysis = tune.run(
    train,
    num_samples=100,
    search_alg=ConcurrencyLimiter(optuna_search, max_concurrent=10),
    verbose=2,
    resources_per_trial={"gpu": 1, "cpu": 5},
)


# trial_name_creator=tune.function(name_siphon)

"""
ray start --head
nohup python3 killer_daemon.py &> killer_log.out &
nohup python3 run_hyperparameters.py &> tune_log.out &

To do:
Seeding

Things to worry about:
gamma isn't log
clip range isn't log
Small batch sizes
Not picking the last policy name right

Potential code upgrades:
Proper reward curve logging via SB3 PR
Knockknock
Parallelize evaluations
Add try mkdirs for everything in code or seperate script
unify log naming- Trial 0eeff824 reported mean_reward=-133.8674492857906 with parameters={'gamma': 0.9551699448901668, 'n_steps': 2048, 'ent_coef': 0.0025892462489487634, 'learning_rate': 5.00000000000005e-06, 'vf_coef': 0.51031746414931, 'max_grad_norm': 0.01, 'gae_lambda': 1.0, 'n_epochs': 9, 'n_envs': 4}
Figure out github ssh key issue
Use old hyperparameters as seed (?)
Disable fail2ban
Use local and remote machines (docker?)
Have head be GPUless VM so it cant get rebooted on maintenance
Automatically stop using GCP resources (don't kill master node though)
Pruner
FP16
Sensibly NaN handling
Parallel env evaluations
Proper reward logging in SB


Potential learning upgrades:
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
