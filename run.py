import torch.nn as nn
from preset2 import sac
from all.experiments.multiagent_env_experiment import MultiagentEnvExperiment
from all.environments import MultiagentPettingZooEnv
import supersuit as ss


from pettingzoo.butterfly import pistonball_v3

env = pistonball_v3.env(n_pistons=10, local_ratio=0.2, time_penalty=-0.1, continuous=True, random_drop=True, random_rotate=True, ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5, max_cycles=900)
env = ss.color_reduction_v0(env, mode='B')
env = ss.dtype_v0(env, 'float32')
env = ss.resize_v0(env, x_size=20, y_size=76)
env = ss.flatten_v0(env)
env = ss.frame_stack_v1(env, 2)

"""
from pettingzoo.butterfly import cooperative_pong_v2

env = cooperative_pong_v2.env()
env = ss.color_reduction_v0(env, mode='B')
env = ss.dtype_v0(env, 'float32')
env = ss.resize_v0(env, x_size=40, y_size=40)
env = ss.flatten_v0(env)
env = ss.frame_stack_v1(env, 2)
"""

state_dim = env.observation_spaces['piston_0'].shape[0]
action_dim = 1  # single channel PPO
hidden1 = 400
hidden2 = 300


def fc_v(env):
    return nn.Sequential(
        nn.Linear(state_dim, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, 1)
    )


def fc_soft_policy(env):
    return nn.Sequential(
        nn.Linear(state_dim, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, action_dim*2)
    )


def fc_q(env):
    return nn.Sequential(
        nn.Linear(state_dim+action_dim, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, 1),
    )


hyperparameters = {
    # Common settings
    "discount_factor": 0.98,
    # Adam optimizer settings
    "lr_q": 1e-3,
    "lr_v": 1e-3,
    "lr_pi": 1e-4,
    # Training settings
    "minibatch_size": 100,
    "update_frequency": 2,
    "polyak_rate": 0.005,
    # Replay Buffer settings
    "replay_start_size": 300,  # was 5000
    "replay_buffer_size": 1e6,
    # Exploration settings
    "temperature_initial": 0.1,
    "lr_temperature": 1e-5,
    "entropy_target_scaling": 1.,
    # Model construction
    "q1_model_constructor": fc_q,
    "q2_model_constructor": fc_q,
    "v_model_constructor": fc_v,
    "policy_model_constructor": fc_soft_policy
}


env = MultiagentPettingZooEnv(env, 'pistonball')
preset = sac.hyperparameters(**hyperparameters).env(env).build()
experiment = MultiagentEnvExperiment(
    preset,
    env,
    write_loss=False,
)
experiment.train(1e6)
experiment.test(10)
experiment.save()
experiment.close()

# supersuit obs size logging
