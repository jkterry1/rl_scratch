import torch.nn as nn
from preset import ppo
from all.experiments import run_experiment
# from all.experiments.multiagent_env_experiment import MultiagentEnvExperiment
# from all.environments import GymEnvironment
from all.environments import MultiagentPettingZooEnv
import supersuit as ss
from pettingzoo.butterfly import pistonball_v3

env = pistonball_v3.env(n_pistons=6, local_ratio=0.2, time_penalty=-0.1, continuous=True, random_drop=True, random_rotate=True, ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5, max_cycles=900)

env = ss.color_reduction_v0(env, mode='B')
env = ss.dtype_v0(env, 'float32')
env = ss.resize_v0(env, x_size=20, y_size=76)
env = ss.flatten_v0(env)
env = ss.frame_stack_v1(env, 1)

state_dim = env.observation_spaces[env.agents[0]].shape[0]
action_dim = 2  # single channel PPO
hidden1 = 128
hidden2 = 64

v = nn.Sequential(
    nn.Linear(state_dim, hidden1),
    nn.ReLU(),
    nn.Linear(hidden1, hidden2),
    nn.ReLU(),
    nn.Linear(hidden2, 1)
)

policy = nn.Sequential(
    nn.Linear(state_dim, hidden1),
    nn.ReLU(),
    nn.Linear(hidden1, hidden2),
    nn.ReLU(),
    nn.Linear(hidden2, action_dim*2)
    )


hyperparameters = {
    # Common settings
    "discount_factor": 0.99,
    # Adam optimizer settings
    "lr": 2.5e-4,  # Adam learning rate
    "eps": 1e-5,  # Adam stability
    # Loss scaling
    "entropy_loss_scaling": 0.01,
    "value_loss_scaling": 0.5,
    # Training settings
    "clip_grad": 0.5,
    "clip_initial": 0.1,
    "clip_final": 0.01,
    "epochs": 4,
    "minibatches": 4,
    # Batch settings
    "n_envs": 4,
    "n_steps": 128,
    # GAE settings
    "lam": 0.95,
    # Model construction
    "value_model_constructor": v,
    "policy_model_constructor": policy,
}

run_experiment([ppo(hyperparameters=hyperparameters)], [MultiagentPettingZooEnv(env)], frames=5e6)

# env.reset()
# save_observation(env, all_agents=True)
# crop obs w/ lambda function later
