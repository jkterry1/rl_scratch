from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer
from ray.tune.registry import register_env
from ray.rllib.utils import try_import_tf
from ray.rllib.env import PettingZooEnv
from pettingzoo.butterfly import pistonball_v3
import supersuit as ss

# for APEX-DQN
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

tf1, tf, tfv = try_import_tf()


class MLPModelV2(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name="my_model"):
        super(MLPModelV2, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        # Simplified to one layer.
        input_layer = tf.keras.layers.Input(
                obs_space.shape,
                dtype=obs_space.dtype)
        layer_1 = tf.keras.layers.Dense(
                400,
                activation="relu",
                kernel_initializer=normc_initializer(1.0))(input_layer)
        layer_2 = tf.keras.layers.Dense(
                300,
                activation="relu",
                kernel_initializer=normc_initializer(1.0))(layer_1)
        output = tf.keras.layers.Dense(
                num_outputs,
                activation=None,
                kernel_initializer=normc_initializer(0.01))(layer_2)
        value_out = tf.keras.layers.Dense(
                1,
                activation=None,
                name="value_out",
                kernel_initializer=normc_initializer(0.01))(layer_2)
        self.base_model = tf.keras.Model(input_layer, [output, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


def make_env_creator():
    def env_creator(args):
        env = pistonball_v3.env(n_pistons=10, local_ratio=0.2, time_penalty=-0.1, continuous=True, random_drop=True, random_rotate=True, ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5, max_cycles=900)
        env = ss.color_reduction_v0(env, mode='B')
        env = ss.dtype_v0(env, 'float32')
        env = ss.resize_v0(env, x_size=20, y_size=76)
        env = ss.flatten_v0(env)
        env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
        env = ss.frame_stack_v1(env, 2)
        return env
    return env_creator


env_creator = make_env_creator()

env_name = pistonball_v3

register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))

test_env = PettingZooEnv(env_creator({}))
obs_space = test_env.observation_space
act_space = test_env.action_space

ModelCatalog.register_custom_model("MLPModelV2", MLPModelV2)


def gen_policy(i):
    config = {
        "model": {
            "custom_model": "MLPModelV2",
        },
        "gamma": 0.99,
    }
    return (None, obs_space, act_space, config)


policies = {"policy_0": gen_policy(0)}

# for all methods
policy_ids = list(policies.keys())

tune.run(
    "PPO",
    name="PPO",
    stop={"episodes_total": 60000},
    checkpoint_freq=100,
    local_dir="~/results_unpruned/"+env_name,
    config={
        # Enviroment specific
        "env": env_name,
        # General
        "log_level": "ERROR",
        "num_gpus": 1,
        "num_workers": 8,
        "num_envs_per_worker": 8,
        "compress_observations": False,
        "gamma": .99,

        "lambda": 0.95,
        "kl_coeff": 0.5,
        "clip_rewards": True,
        "clip_param": 0.1,
        "vf_clip_param": 10.0,
        "entropy_coeff": 0.01,
        "train_batch_size": 5000,
        "rollout_fragment_length": 100,
        "sgd_minibatch_size": 500,
        "num_sgd_iter": 10,
        "batch_mode": 'truncate_episodes',

        # Method specific
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": (
                lambda agent_id: policy_ids[0]),
        },
    },
)

"""
Fix env_name stuff
Get to run at all
Watch a saved policy play
Switch to CNN
Start a hyperparameter search
"""